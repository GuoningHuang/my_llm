import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context,set_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache  # (num_blocks, block_size, Hkv, D)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # o = flash_attn_varlen_func(q, k, v,
            #                            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
            #                            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                    #   softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            # #手工实现
            # 因为prefill的时候没有prefix cache，所以不启用block table也没事
            q = q.permute(1, 0, 2).unsqueeze(0)
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
            Hq = q.size(1)
            Hkv = k.size(1)
            if Hq != Hkv:
                assert Hq % Hkv == 0, f"GQA mismatch: Hq={Hq}, Hkv={Hkv}"
                group = Hq // Hkv
                k = k.repeat_interleave(group, dim=1)
                v = v.repeat_interleave(group, dim=1)
            o = F.scaled_dot_product_attention(q, k, v,is_causal=True, scale=self.scale)
            o = o.squeeze(0).permute(1, 0, 2)
            # print(o.shape)
            # [seq_len, H, D] = o.shape

        else:    # decode
            # print(q.shape, k_cache.shape, v_cache.shape)
            # o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            #                             cache_seqlens=context.context_lens, block_table=context.block_tables, 
            #                             softmax_scale=self.scale, causal=True)
            # print(context.context_lens,context.block_tables)
            # print(o.shape)
            # 打印信息：
            # torch.Size([1, 16, 128]) torch.Size([156, 256, 2, 128]) torch.Size([156, 256, 2, 128])
            # tensor([57], device='cuda:0', dtype=torch.int32) tensor([[0]], device='cuda:0', dtype=torch.int32)
            # torch.Size([1, 1, 16, 128])

                                    
            B, Hq, D = q.shape
            Hkv = k_cache.size(2)
            T = context.context_lens[0].item()

            # 1. 提取有效的 blocks
            block_table = context.block_tables[0] # [num_blocks_used]
            k_blocks = k_cache[block_table] # (num_used_blocks, Block_size, Hkv, D)

            # 2. 还原成连续序列并截断
            # 先转置成 (num_used_blocks, Hkv, Block_size, D) 
            # 这样 reshape 之后，Block_size 的维度才是连续的时间轴
            k_seq = k_blocks.transpose(1, 2).reshape(Hkv, -1, D) # (Hkv, Total_capacity, D)
            k_hist = k_seq[:, :T, :] # (Hkv, T, D)

            v_seq = v_cache[block_table].transpose(1, 2).reshape(Hkv, -1, D)
            v_hist = v_seq[:, :T, :] # (Hkv, T, D)

            # 3. 准备 SDPA 的输入
            # Q: (1, Hq, 1, D)
            q_input = q.view(B, Hq, 1, D) 

            # K/V: (1, Hq, T, D) - 处理 GQA
            # 使用 repeat_interleave 而不是手动 index 往往更稳妥
            group = Hq // Hkv
            k_input = k_hist.repeat_interleave(group, dim=0).unsqueeze(0) # (1, Hq, T, D)
            v_input = v_hist.repeat_interleave(group, dim=0).unsqueeze(0) # (1, Hq, T, D)

            # 4. 计算
            o = F.scaled_dot_product_attention(q_input, k_input, v_input, is_causal=False, scale=self.scale)

            # 5. 转回 Flash Attention 默认的 (B, S, H, D)
            o = o.transpose(1, 2) # (1, 1, 16, 128)

        return o


def main():
    torch.manual_seed(0)
    device = "cuda"
    
    # --- 1. 配置参数 ---
    BATCH_SIZE = 1
    NUM_HEADS = 32
    NUM_KV_HEADS = 32  # 假设 GQA=1
    HEAD_DIM = 128
    BLOCK_SIZE = 256
    MAX_SEQ_LEN = 2048
    
    # --- 2. 初始化模型 ---
    print(f"初始化模型: Batch={BATCH_SIZE}, HeadDim={HEAD_DIM}, MaxSeq={MAX_SEQ_LEN}")
    model = Attention(
        num_heads=NUM_HEADS, 
        head_dim=HEAD_DIM, 
        scale=HEAD_DIM**-0.5, 
        num_kv_heads=NUM_KV_HEADS
    ).to(device).half()

    # --- 3. 初始化 KV Cache ---
    # 根据 store_kvcache 中的 assert，我们需要满足 stride(1) == D
    # 使用形状 [Num_Blocks, Block_Size, Num_Heads, Head_Dim] 符合要求
    total_tokens = BATCH_SIZE * MAX_SEQ_LEN
    num_blocks = (total_tokens // BLOCK_SIZE) * 2 # 预留足够空间
    
    k_cache = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    v_cache = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    
    # 将 Cache 注入模型
    model.k_cache = k_cache
    model.v_cache = v_cache

    # ==========================================
    # Phase 1: Prefill 性能测试
    # ==========================================
    print("\n>>> 开始测试 Prefill 阶段...")
    
    # 构造输入 (Flattened format for FlashAttn Varlen)
    q_prefill = torch.randn(total_tokens, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    k_prefill = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    v_prefill = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    
    cu_seqlens = torch.arange(0, (BATCH_SIZE + 1) * MAX_SEQ_LEN, step=MAX_SEQ_LEN, device=device, dtype=torch.int32)
    # 第idx个产生的kv cache，要写到kv cache的第几个token的位置
    slot_mapping_prefill = torch.arange(total_tokens, device=device, dtype=torch.int32)

    # block— tables 是虚拟页表，表示哪个seq的 kv cache在哪几个block里面存储

    # 设置 Context
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=MAX_SEQ_LEN,
        max_seqlen_k=MAX_SEQ_LEN,
        slot_mapping=slot_mapping_prefill
    )

    # 预热
    for _ in range(5):
        model(q_prefill, k_prefill, v_prefill)
    
    # 测速
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(20):
        model(q_prefill, k_prefill, v_prefill)
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / 20
    print(f"Prefill 耗时: {avg_time:.2f} ms")
    print(f"Prefill 吞吐: {total_tokens / (avg_time/1000):.2f} tokens/s")

    # ==========================================
    # Phase 2: Decode 性能测试
    # ==========================================
    print("\n>>> 开始测试 Decode 阶段...")

    # Decode 输入: Batch 维度
    q_decode = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    k_decode = torch.randn(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    v_decode = torch.randn(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float16)

    # 构造 Block Tables (简单的线性分配)
    blocks_per_seq = MAX_SEQ_LEN // BLOCK_SIZE
    block_tables = torch.zeros(BATCH_SIZE, blocks_per_seq + 1, device=device, dtype=torch.int32)
    for i in range(BATCH_SIZE):
        base = i * blocks_per_seq
        block_tables[i, :blocks_per_seq] = torch.arange(base, base + blocks_per_seq, dtype=torch.int32)

    context_lens = torch.full((BATCH_SIZE,), MAX_SEQ_LEN, device=device, dtype=torch.int32)
    
    # Slot mapping 指向当前生成的 token 写入位置 (假设是序列的最后一个位置)
    slot_mapping_decode = torch.zeros(BATCH_SIZE, device=device, dtype=torch.int32)
    for i in range(BATCH_SIZE):
        last_block = block_tables[i, blocks_per_seq - 1]
        slot_mapping_decode[i] = last_block * BLOCK_SIZE + (BLOCK_SIZE - 1)

    # 设置 Context
    set_context(
        is_prefill=False,
        context_lens=context_lens,
        block_tables=block_tables,
        slot_mapping=slot_mapping_decode
    )

    # 预热
    for _ in range(10):
        model(q_decode, k_decode, v_decode)

    # 测速
    start.record()
    for _ in range(100):
        model(q_decode, k_decode, v_decode)
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / 100
    print(f"Decode 耗时: {avg_time:.4f} ms")
    print(f"Decode 吞吐: {BATCH_SIZE / (avg_time/1000):.2f} tokens/s")

if __name__ == "__main__":
    main()