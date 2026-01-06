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

def main():
    # 1. 基础配置
    # model_path = os.path.expanduser("/model/DeepSeek-R1-Distill-Qwen-1.5B/")
    model_path = os.path.expanduser("/model/Qwen2.5-3B-Instruct/")

    # prompt_file = "prompt_short.txt"    
    # with open(prompt_file, "r", encoding="utf-8") as f:
    #     prompt_text = f.read().strip()
    prompt_text = "who you are."

    kwargs = {"tensor_parallel_size": 1, "enforce_eager": True}
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(model_path, **config_kwargs)

    # 2. 初始化核心组件 (按你要求的直接调用方式)
    model_runner = ModelRunner(config, 0, [])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config.eos = tokenizer.eos_token_id
    block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

    # 3. 准备数据
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"}
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
    prefill_start = perf_counter()
    
    # 直接调用 ModelRunner 处理 Prompt
    new_token_ids = model_runner.call("run", [seq], True)
    
    prefill_time = perf_counter() - prefill_start
    gen_token_id = new_token_ids[0]
    seq.append_token(gen_token_id)
    
    prefill_tokens = len(prompt_tokens)
    print(f"\n[Prefill 阶段]")
    print(f"总字数: {prefill_tokens} tok")
    print(f"总耗时: {prefill_time:.4f} s")
    print(f"吞吐量: {prefill_tokens / prefill_time:.2f} tok/s")

    # --- 阶段二：Decode (解码) ---
    decode_start = perf_counter()
    decode_token_count = 0
    
    while not (gen_token_id == config.eos or seq.num_completion_tokens >= sampling_params.max_tokens):
        # 动态申请显存块
        block_manager.may_append(seq)
        
        # 直接调用 ModelRunner 生成下一个字
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
    


if __name__ == "__main__":
    main()