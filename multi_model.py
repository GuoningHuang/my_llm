import os
import torch
from dataclasses import fields
from time import perf_counter
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.memory_manager import MemoryManager

def create_config(model_path, **kwargs):
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    return Config(model_path, **config_kwargs)

def run_chat(runner, tokenizer, block_manager, prompt_text):
    """一个简单的聊天运行函数"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    sampling_params = SamplingParams(max_tokens=128)
    seq = Sequence(prompt_tokens, sampling_params)
    
    block_manager.allocate(seq)

    # Prefill
    print(f"[Model]: {runner.config.model}")
    prefill_start = perf_counter()
    new_token_ids = runner.call("run", [seq], True)
    prefill_time = perf_counter() - prefill_start
    
    gen_token_id = new_token_ids[0]
    seq.append_token(gen_token_id)
    print(f"prefill Time: {prefill_time:.4f}s")
    print(f"prefill throughput: {len(prompt_tokens) / prefill_time:.2f} tok/s")


    # Decode
    decode_start = perf_counter()
    while not (gen_token_id == runner.config.eos or seq.num_completion_tokens >= sampling_params.max_tokens):
        block_manager.may_append(seq)
        new_token_ids = runner.call("run", [seq], False)
        gen_token_id = new_token_ids[0]
        seq.append_token(gen_token_id)
    decode_time = perf_counter() - decode_start
    print(f"decode Time: {decode_time:.4f}s")
    print(f"decode throughput: {seq.num_completion_tokens / decode_time:.2f} tok/s")
    
    print(f"Output: {tokenizer.decode(seq.completion_token_ids)}")
    
    # 清理 BlockManager，释放逻辑块，以便下一个模型使用同一个物理空间
    block_manager.deallocate(seq)


def main():
    # 1. 准备模型路径
    path_1 = os.path.expanduser("/model/Qwen2.5-3B-Instruct/")
    path_2 = os.path.expanduser("/model/DeepSeek-R1-Distill-Qwen-1.5B/")

    # 2. 初始化全局 Memory Manager (只初始化一次)
    kwargs = {"tensor_parallel_size": 1, "enforce_eager": True, "gpu_memory_utilization": 0.9}
    base_config = create_config(path_1, **kwargs)
    
    print(">>> Initializing Unified Memory Manager...")
    memory_manager = MemoryManager(base_config, device="cuda")

    # 3. 加载模型 1 (权重加载，不分配KV)
    print(f"\n>>> Loading Model 1: {path_1}")
    config_1 = create_config(path_1, **kwargs)
    # 传入 memory_manager，ModelRunner 会跳过自动 KV 分配
    runner_1 = ModelRunner(config_1, 0, [], memory_manager=memory_manager)
    tokenizer_1 = AutoTokenizer.from_pretrained(path_1)
    config_1.eos = tokenizer_1.eos_token_id

    # 4. 加载模型 2 (权重加载，不分配KV)
    # 此时 Model 1 的权重已在显存中，Model 2 的权重紧随其后加载
    print(f"\n>>> Loading Model 2: {path_2}")
    config_2 = create_config(path_2, **kwargs)
    runner_2 = ModelRunner(config_2, 0, [], memory_manager=memory_manager)
    tokenizer_2 = AutoTokenizer.from_pretrained(path_2)
    config_2.eos = tokenizer_2.eos_token_id

    # 5. 【新增步骤】所有权重加载完毕，统一初始化 KV Cache
    print("\n>>> Allocating KV Cache for all models...")
    
    # 5.1 为 Runner 1 分配 (这会触发 memory_manager 锁定剩余显存为 KV 池)
    runner_1.warmup_model()
    runner_1.allocate_kv_cache()
    if not runner_1.enforce_eager:
        runner_1.capture_cudagraph()

    # 5.2 为 Runner 2 分配 (这会复用 Runner 1 创建的 KV 池)
    runner_2.warmup_model()
    runner_2.allocate_kv_cache()
    if not runner_2.enforce_eager:
        runner_2.capture_cudagraph()

    # 6. 初始化各自的 Block Manager
    bm_1 = BlockManager(config_1.num_kvcache_blocks, config_1.kvcache_block_size)
    bm_2 = BlockManager(config_2.num_kvcache_blocks, config_2.kvcache_block_size)

    # 7. 按需运行
    run_chat(runner_1, tokenizer_1, bm_1, "Who are you?")
    run_chat(runner_2, tokenizer_2, bm_2, "Who are you?")

if __name__ == "__main__":
    main()
