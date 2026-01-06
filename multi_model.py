import os
import gc
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

def run_chat(model_name, runner, tokenizer, block_manager, prompt_text):
    """聊天运行函数"""
    print(f"\n[{model_name}] >>> Running Chat...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    sampling_params = SamplingParams(max_tokens=128)
    seq = Sequence(prompt_tokens, sampling_params)
    
    block_manager.allocate(seq)

    # Prefill
    start = perf_counter()
    new_token_ids = runner.call("run", [seq], True)
    prefill_time = perf_counter() - start
    
    gen_token_id = new_token_ids[0]
    seq.append_token(gen_token_id)
    print(f"[{model_name}] Prefill: {len(prompt_tokens)/prefill_time:.2f} tok/s")

    # Decode
    while not (gen_token_id == runner.config.eos or seq.num_completion_tokens >= sampling_params.max_tokens):
        block_manager.may_append(seq)
        new_token_ids = runner.call("run", [seq], False)
        gen_token_id = new_token_ids[0]
        seq.append_token(gen_token_id)
    
    print(f"[{model_name}] Output: {tokenizer.decode(seq.completion_token_ids)}")
    
    # 清理 BlockManager，释放逻辑块
    block_manager.deallocate(seq)


def main():
    # 路径配置
    path_1 = os.path.expanduser("/model/Qwen2.5-3B-Instruct/")
    path_2 = os.path.expanduser("/model/DeepSeek-R1-Distill-Qwen-1.5B/")
    path_3 = os.path.expanduser("/model/Qwen2.5-0.5B-Instruct/")

    # 1. 初始化全局 Memory Manager
    kwargs = {"tensor_parallel_size": 1, "enforce_eager": True, "gpu_memory_utilization": 0.9}
    base_config = create_config(path_1, **kwargs)
    
    print(">>> Initializing Unified Memory Manager...")
    memory_manager = MemoryManager(base_config, device="cuda")

    # =========================================================================
    # 阶段 1: 加载 Model 1 和 Model 2
    # =========================================================================
    
    # --- 加载 Model 1 ---
    print(f"\n>>> Loading Model 1: {path_1}")
    memory_manager.start_allocation("model_1") # 标记 M1 的显存块
    config_1 = create_config(path_1, **kwargs)
    runner_1 = ModelRunner(config_1, 0, [], memory_manager=memory_manager)
    tokenizer_1 = AutoTokenizer.from_pretrained(path_1)
    config_1.eos = tokenizer_1.eos_token_id
    memory_manager.stop_allocation()

    # --- 加载 Model 2 ---
    print(f"\n>>> Loading Model 2: {path_2}")
    memory_manager.start_allocation("model_2") # 标记 M2 的显存块
    config_2 = create_config(path_2, **kwargs)
    runner_2 = ModelRunner(config_2, 0, [], memory_manager=memory_manager)
    tokenizer_2 = AutoTokenizer.from_pretrained(path_2)
    config_2.eos = tokenizer_2.eos_token_id
    memory_manager.stop_allocation()

    # =========================================================================
    # 阶段 2: 运行 Model 1 (这是你要求的关键步骤)
    # =========================================================================
    print(f"\n{'='*20} Running Model 1 First {'='*20}")
    
    # 1. 分配 KV Cache (Global Pool 将在此刻被锁定!)
    # 注意：此时 KV Pool 的大小 = 总显存 - (M1权重 + M2权重)
    runner_1.warmup_model()
    runner_1.allocate_kv_cache()
    if not runner_1.enforce_eager:
        runner_1.capture_cudagraph()

    # 2. 运行推理
    bm_1 = BlockManager(config_1.num_kvcache_blocks, config_1.kvcache_block_size)
    run_chat("Model 1", runner_1, tokenizer_1, bm_1, "Who are you?")

    # =========================================================================
    # 阶段 3: 删除 Model 1 (释放权重，但保留全局 KV Pool)
    # =========================================================================
    print(f"\n{'='*20} Deleting Model 1 {'='*20}")
    
    # 1. 删除 Python 对象引用
    del runner_1
    del bm_1
    del tokenizer_1
    
    # 2. 强制垃圾回收和清理碎片
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. 归还显存到 Memory Manager 的空闲链表
    # 这会在内存池的开头产生一个巨大的“空洞” (Hole)，可以用来放 Model 3
    memory_manager.free_model("model_1")

    # =========================================================================
    # 阶段 4: 加载 Model 3 (填补空缺) & 运行剩余模型
    # =========================================================================
    print(f"\n>>> Loading Model 3: {path_3} (Filling the hole)")
    memory_manager.start_allocation("model_3")
    config_3 = create_config(path_3, **kwargs)
    # 因为 M3 (0.5B) < M1 (3B)，它完全能塞进 M1 留下的空洞里
    runner_3 = ModelRunner(config_3, 0, [], memory_manager=memory_manager)
    tokenizer_3 = AutoTokenizer.from_pretrained(path_3)
    config_3.eos = tokenizer_3.eos_token_id
    memory_manager.stop_allocation()

    print("\n>>> Allocating KV Cache for M2 & M3 (Reusing existing pool)...")
    
    # 为 Model 2 分配 (复用之前 M1 创建的那个池子)
    runner_2.warmup_model()
    runner_2.allocate_kv_cache()
    if not runner_2.enforce_eager: runner_2.capture_cudagraph()

    # 为 Model 3 分配 (复用同一个池子)
    runner_3.warmup_model()
    runner_3.allocate_kv_cache()
    if not runner_3.enforce_eager: runner_3.capture_cudagraph()

    # 运行
    bm_2 = BlockManager(config_2.num_kvcache_blocks, config_2.kvcache_block_size)
    bm_3 = BlockManager(config_3.num_kvcache_blocks, config_3.kvcache_block_size)

    run_chat("Model 2", runner_2, tokenizer_2, bm_2, "Who are you?")
    run_chat("Model 3", runner_3, tokenizer_3, bm_3, "Who are you?")

if __name__ == "__main__":
    main()