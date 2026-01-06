import os
import torch
import torch.distributed as dist  # 新增引用
from dataclasses import fields
from time import perf_counter
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.memory_manager import MemoryManager # 新增引用

def main():
    # 1. 基础配置
    model_path = os.path.expanduser("/model/Qwen2.5-3B-Instruct/")
    # model_path = os.path.expanduser("/model/DeepSeek-R1-Distill-Qwen-1.5B/")

    prompt_text = "who you are."

    # 增加 gpu_memory_utilization 配置，确保显存池大小明确
    kwargs = {
        "tensor_parallel_size": 1, 
        "enforce_eager": True,
        "gpu_memory_utilization": 0.9 
    }
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(model_path, **config_kwargs)

    # 2. 初始化核心组件 
    # [修改点 1] 显式初始化 MemoryManager
    memory_manager = MemoryManager(config, device="cuda")

    # [修改点 2] 传入 memory_manager，ModelRunner 不会自动分配 KV
    model_runner = ModelRunner(config, 0, [], memory_manager=memory_manager)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config.eos = tokenizer.eos_token_id

    # [修改点 3] 显式执行 ModelRunner 的准备工作 (与 multi_model.py 保持一致)
    print(">>> Allocating KV Cache and Warming up...")
    model_runner.warmup_model()
    model_runner.allocate_kv_cache()
    if not config.enforce_eager:
        model_runner.capture_cudagraph()

    # BlockManager 使用 config 中计算好的 block 数量（由 allocate_kv_cache更新）
    block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

    # 3. 准备数据
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]

    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(max_tokens=1024)
    seq = Sequence(prompt_tokens, sampling_params)
    
    # 物理显存分配
    block_manager.allocate(seq)

    # --- 阶段一：Prefill (预填充) ---
    print(f"\n[Prefill 阶段] Model: {config.model}")
    prefill_start = perf_counter()
    
    new_token_ids = model_runner.call("run", [seq], True)
    
    prefill_time = perf_counter() - prefill_start
    gen_token_id = new_token_ids[0]
    seq.append_token(gen_token_id)
    
    prefill_tokens = len(prompt_tokens)
    print(f"总字数: {prefill_tokens} tok")
    print(f"总耗时: {prefill_time:.4f} s")
    print(f"吞吐量: {prefill_tokens / prefill_time:.2f} tok/s")

    # --- 阶段二：Decode (解码) ---
    decode_start = perf_counter()
    decode_token_count = 0
    
    while not (gen_token_id == config.eos or seq.num_completion_tokens >= sampling_params.max_tokens):
        # 动态申请显存块
        block_manager.may_append(seq)
        
        new_token_ids = model_runner.call("run", [seq], False)
        gen_token_id = new_token_ids[0]
        seq.append_token(gen_token_id)
        decode_token_count += 1
        
    decode_time = perf_counter() - decode_start
    
    print(f"\n[Decode 阶段]")
    print(f"总字数: {decode_token_count} tok")
    print(f"总耗时: {decode_time:.4f} s")
    print(f"吞吐量: {decode_token_count / decode_time:.2f} tok/s")

    # 4. 最终输出
    print(f"\n最终回答: {tokenizer.decode(seq.completion_token_ids)}")
    
    # [修改点 4] 清理资源
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()