import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.layers.rotary_embedding import RotaryEmbedding

from nanovllm.engine.memory_manager import MemoryManager

import time

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], memory_manager):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        if not dist.is_initialized():
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        with torch.device("meta"):
            self.model = Qwen3ForCausalLM(hf_config)
            if getattr(hf_config, "tie_word_embeddings", False):
                if hasattr(self.model, "tie_weights"):
                    self.model.tie_weights()
                else:
                    # 如果模型没有实现 tie_weights，手动绑定 (针对常见结构)
                    # 注意：根据你的模型结构，路径可能是 self.model.model.embed_tokens
                    if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                         self.model.lm_head.weight = self.model.model.embed_tokens.weight

        self.memory_manager = memory_manager

        param_map = {} # 用于记录已分配的参数，处理权重共享(tie_weights)
        
        for name, param in self.model.named_parameters(remove_duplicate=False):
            if id(param) in param_map:
                self._set_module_parameter(name, param_map[id(param)])
                continue

            numel = param.numel()
            element_size = param.element_size()
            nbytes = numel * element_size
            
            raw_ptr = self.memory_manager.allocate_weights(nbytes)
            param_view = raw_ptr.view(param.dtype).view(param.shape)
            new_param = nn.Parameter(param_view, requires_grad=param.requires_grad)
            if hasattr(param, "weight_loader"):
                new_param.weight_loader = param.weight_loader
            self._set_module_parameter(name, new_param)
            param_map[id(param)] = new_param

        self.fix_buffers()

        start_load = time.perf_counter()
        load_model(self.model, config.model)
        end_load = time.perf_counter()
        if self.rank == 0:
            print(f"Weights loaded in {end_load - start_load:.2f} seconds")
            
        self.sampler = Sampler()
        # self.warmup_model()
        # self.allocate_kv_cache()
        # if not self.enforce_eager:
        #     self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
    def _set_module_parameter(self, param_name, new_param):
        """
        根据参数名称（如 'model.layers.0.self_attn.q_proj.weight'）
        找到对应的模块，并替换其参数属性。
        """
        module = self.model
        parts = param_name.split('.')
        # 遍历找到父模块
        for part in parts[:-1]:
            module = getattr(module, part)
        # 替换属性
        setattr(module, parts[-1], new_param)

    def fix_buffers(self):
        """修复因为 Meta 初始化而丢失或未计算的 Buffers (主要是 RoPE)"""
        hf_config = self.config.hf_config
        # 获取 RoPE 配置
        base = getattr(hf_config, "rope_theta", 10000.0)
        max_position = getattr(hf_config, "max_position_embeddings", 4096)
        
        for module in self.model.modules():
            if isinstance(module, RotaryEmbedding):
                # 重新计算 cos/sin 缓存
                head_size = module.head_size
                inv_freq = 1.0 / (base**(torch.arange(0, head_size, 2, dtype=torch.float, device="cuda") / head_size))
                t = torch.arange(max_position, dtype=torch.float, device="cuda")
                freqs = torch.einsum("i,j -> ij", t, inv_freq)
                cos = freqs.cos()
                sin = freqs.sin()
                cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
                # 覆盖 buffer
                module.cos_sin_cache = cache.to(dtype=hf_config.torch_dtype)


    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        del self.kv_cache
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        
        # 计算单个 Block 的大小 (字节)
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        if self.memory_manager.kv_buffer is None:
            # 尚未分配，我是第一个初始化的 Runner (或者唯一一个)
            # manager 会计算能放多少个 block，并打印内存报告
            config.num_kvcache_blocks = self.memory_manager.init_kv_pool(block_bytes)
        else:
            # 已经分配过了（说明这是第二个模型，或者被手动调用过）
            # 直接根据现有的 buffer 大小计算我的 config 能分多少个 block
            # 注意：这里假设所有模型复用同一块物理内存，且总是占满剩余显存
            kv_buffer_size = len(self.memory_manager.kv_buffer)
            config.num_kvcache_blocks = kv_buffer_size // block_bytes
            print(f"Reuse KV Pool: {kv_buffer_size / 1024**3:.2f} GB available, assigned {config.num_kvcache_blocks} blocks to current model.")
        
        # 获取切分好的 KV 缓存 Buffer
        kv_storage = self.memory_manager.get_kv_buffer()

        needed_bytes = config.num_kvcache_blocks * block_bytes
        kv_storage = kv_storage[:needed_bytes]
        
        # View 成 KV Cache 的形状
        # 形状: [2, layers, blocks, block_size, kv_heads, head_dim]
        self.kv_cache = kv_storage.view(hf_config.torch_dtype).view(
            2, 
            hf_config.num_hidden_layers, 
            config.num_kvcache_blocks, 
            self.block_size, 
            num_kv_heads, 
            head_dim
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
