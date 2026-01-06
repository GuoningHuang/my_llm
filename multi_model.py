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
    print(f"\n[Prefill] Model: {runner.config.model}")
    prefill_start = perf_counter()
    new_token_ids = runner.call("run", [seq], True)
    prefill_time = perf_counter() - prefill_start
    
    gen_token_id = new_token_ids[0]
    seq.append_token(gen_token_id)
    print(f"Time: {prefill_time:.4f}s")

    # Decode
    print(f"[Decode]")
    decode_start = perf_counter()
    while not (gen_token_id == runner.config.eos or seq.num_completion_tokens >= sampling_params.max_tokens):
        block_manager.may_append(seq)
        new_token_ids = runner.call("run", [seq], False)
        gen_token_id = new_token_ids[0]
        seq.append_token(gen_token_id)
    
    print(f"Output: {tokenizer.decode(seq.completion_token_ids)}")
    
    # 清理 BlockManager，释放逻辑块，以便下一个模型使用同一个物理空间
    block_manager.deallocate(seq)


def main():
    # 1. 准备模型路径
    path_1 = os.path.expanduser("/model/Qwen2.5-3B-Instruct/")
    path_2 = os.path.expanduser("/model/DeepSeek-R1-Distill-Qwen-1.5B/") # 假设的第二个模型

    # 2. 初始化全局 Memory Manager (只初始化一次)
    # 使用 path_1 的配置作为 base，主要是为了 gpu_memory_utilization
    kwargs = {"tensor_parallel_size": 1, "enforce_eager": True, "gpu_memory_utilization": 0.9}
    base_config = create_config(path_1, **kwargs)
    
    print(">>> Initializing Unified Memory Manager...")
    memory_manager = MemoryManager(base_config, device="cuda")

    # 3. 加载模型 1 (传入 memory_manager)
    print(f"\n>>> Loading Model 1: {path_1}")
    config_1 = create_config(path_1, **kwargs)
    runner_1 = ModelRunner(config_1, 0, [], memory_manager=memory_manager)
    tokenizer_1 = AutoTokenizer.from_pretrained(path_1)
    config_1.eos = tokenizer_1.eos_token_id

    # 4. 加载模型 2 (传入同一个 memory_manager)
    print(f"\n>>> Loading Model 2: {path_2}")
    config_2 = create_config(path_2, **kwargs)
    runner_2 = ModelRunner(config_2, 0, [], memory_manager=memory_manager)
    tokenizer_2 = AutoTokenizer.from_pretrained(path_2)
    config_2.eos = tokenizer_2.eos_token_id


    # 5. 初始化各自的 Block Manager (逻辑管理)
    bm_1 = BlockManager(config_1.num_kvcache_blocks, config_1.kvcache_block_size)
    bm_2 = BlockManager(config_2.num_kvcache_blocks, config_2.kvcache_block_size)

    # 6. 按需运行
    # 运行模型 1
    run_chat(runner_1, tokenizer_1, bm_1, "Who are you?")
    
    # 运行模型 2 (复用同一块 KV 显存，因为 bm_1 已经 deallocate 了，bm_2 会从 Block 0 开始写)
    run_chat(runner_2, tokenizer_2, bm_2, "Who are you?")

if __name__ == "__main__":
    main()