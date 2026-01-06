import torch
from transformers import AutoModelForCausalLM
import os
import sys

# --- 配置路径 ---
project_root = "/home/gnhuang/code/my_llm"
os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, project_root)

# --- 导入自定义模块 ---
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config

# --- 辅助函数：统一对比逻辑 ---
def check_diff(name, tensor_my, tensor_hf, threshold=1e-3):
    """
    对比两个 Tensor 的差异
    """
    # 确保都在 CPU 上并转为 float32 防止半精度溢出导致的误差显示问题
    t_my = tensor_my.cpu().float()
    t_hf = tensor_hf.cpu().float()
    
    # 形状检查
    if t_my.shape != t_hf.shape:
        print(f"❌ [Shape Mismatch] {name}")
        print(f"   Mine: {t_my.shape}")
        print(f"   HF:   {t_hf.shape}")
        return False

    diff = (t_my - t_hf).abs().max().item()
    
    if diff > threshold:
        print(f"❌ [FAIL] {name} | Diff: {diff:.6f}")
        # 打印前几个值方便调试
        print(f"   Mine (first 5): {t_my.flatten()[:5].tolist()}")
        print(f"   HF   (first 5): {t_hf.flatten()[:5].tolist()}")
        return False
    else:
        # 如果差异极小，为了版面整洁可以选择不打印，或者打印通过
        print(f"✅ [PASS] {name} | Diff: {diff:.6e}")
        return True

def compare_weights():
    model_path = "/model/Qwen2.5-3B-Instruct/"
    
    print("="*50)
    print(f"Loading models from: {model_path}")
    print("="*50)

    # 1. 加载你的模型
    print(">>> Loading Custom Model...")
    config = Config(model_path)
    config.tensor_parallel_size = 1 
    config.enforce_eager = True
    runner = ModelRunner(config, 0, None) 
    my_model = runner.model

    # 2. 加载 HF 标准模型
    print(">>> Loading HF Model (Reference)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )

    # 获取层数
    num_layers = hf_model.config.num_hidden_layers
    print(f"\nTotal Layers to check: {num_layers}")

    # --- 0. 对比 Embeddings (入口) ---
    print("\n=== Checking Embeddings ===")
    check_diff("Embed Tokens", 
               my_model.model.embed_tokens.weight, 
               hf_model.model.embed_tokens.weight)

    # --- 1. 循环遍历所有层 ---
    for i in range(num_layers):
        print(f"\n--- Layer {i} ---")
        
        # 获取当前层的句柄
        my_layer = my_model.model.layers[i]
        hf_layer = hf_model.model.layers[i]

        # A. 对比 Input Layernorm
        check_diff(f"L{i}.input_layernorm", 
                   my_layer.input_layernorm.weight, 
                   hf_layer.input_layernorm.weight)

        # B. 对比 Self Attention (QKV)
        hf_q = hf_layer.self_attn.q_proj.weight
        hf_k = hf_layer.self_attn.k_proj.weight
        hf_v = hf_layer.self_attn.v_proj.weight
        hf_o = hf_layer.self_attn.o_proj.weight

        my_attn = my_layer.self_attn
        
        # 逻辑：判断是合并还是分开
        if hasattr(my_attn, "qkv_proj"):
            # Fused 情况
            my_qkv = my_attn.qkv_proj.weight.cpu()
            
            # 计算切分维度
            hidden_size = hf_model.config.hidden_size
            num_heads = hf_model.config.num_attention_heads
            num_kv_heads = hf_model.config.num_key_value_heads
            head_dim = hidden_size // num_heads
            
            q_dim = num_heads * head_dim
            k_dim = num_kv_heads * head_dim
            v_dim = num_kv_heads * head_dim # 通常 K 和 V 大小一致

            my_q = my_qkv[:q_dim]
            my_k = my_qkv[q_dim : q_dim + k_dim]
            my_v = my_qkv[q_dim + k_dim :]

            check_diff(f"L{i}.Attn Fused-Q", my_q, hf_q)
            check_diff(f"L{i}.Attn Fused-K", my_k, hf_k)
            check_diff(f"L{i}.Attn Fused-V", my_v, hf_v)
            
        elif hasattr(my_attn, "q_proj"):
            # Split 情况
            check_diff(f"L{i}.Attn Q", my_attn.q_proj.weight, hf_q)
            check_diff(f"L{i}.Attn K", my_attn.k_proj.weight, hf_k)
            check_diff(f"L{i}.Attn V", my_attn.v_proj.weight, hf_v)
        else:
            print(f"❌ Layer {i}: Cannot find q_proj or qkv_proj in custom model")

        # C. 对比 Output Proj (O_proj)
        check_diff(f"L{i}.Attn O_proj", my_attn.o_proj.weight, hf_o)

        # D. 对比 Post Attention Layernorm
        check_diff(f"L{i}.post_attn_layernorm", 
                   my_layer.post_attention_layernorm.weight, 
                   hf_layer.post_attention_layernorm.weight)

        # E. 对比 MLP (Gate, Up, Down)
        # 注意：有些模型实现会将 gate 和 up 合并为 gate_up_proj，这里假设你是分开的或者你可以模仿 QKV 的逻辑修改
        # 如果你的模型有 gate_up_proj，请告诉我，我再加一段 split 逻辑
        if hasattr(my_layer.mlp, "gate_proj"):
            check_diff(f"L{i}.MLP Gate", my_layer.mlp.gate_proj.weight, hf_layer.mlp.gate_proj.weight)
            check_diff(f"L{i}.MLP Up",   my_layer.mlp.up_proj.weight,   hf_layer.mlp.up_proj.weight)
        elif hasattr(my_layer.mlp, "gate_up_proj"):
             # 简单的 Fused MLP 处理逻辑 (假设是 concat dim=0)
             my_gate_up = my_layer.mlp.gate_up_proj.weight
             intermediate_size = hf_model.config.intermediate_size
             my_gate = my_gate_up[:intermediate_size]
             my_up = my_gate_up[intermediate_size:]
             check_diff(f"L{i}.MLP Fused-Gate", my_gate, hf_layer.mlp.gate_proj.weight)
             check_diff(f"L{i}.MLP Fused-Up",   my_up,   hf_layer.mlp.up_proj.weight)

        check_diff(f"L{i}.MLP Down", my_layer.mlp.down_proj.weight, hf_layer.mlp.down_proj.weight)

    # --- 2. 对比 Final Norm 和 LM Head ---
    print("\n=== Checking Final Layers ===")
    check_diff("Final Norm", my_model.model.norm.weight, hf_model.model.norm.weight)
    
    # 你的模型可能直接叫 lm_head 或者 model.lm_head，根据实际情况调整
    # 通常 ModelRunner 里的 model 包含了 output head
    if hasattr(my_model, "lm_head"):
        check_diff("LM Head", my_model.lm_head.weight, hf_model.lm_head.weight)
    elif hasattr(my_model.model, "lm_head"):
        # 有些结构 lm_head 在 model 内部
        check_diff("LM Head", my_model.model.lm_head.weight, hf_model.lm_head.weight)
    else:
        print("⚠️ Warning: Could not locate 'lm_head' in custom model to compare.")

    print("\nDone.")

if __name__ == "__main__":
    compare_weights()