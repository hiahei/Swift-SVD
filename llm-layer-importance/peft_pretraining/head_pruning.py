import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def zero_out_attention_head(model_path, output_dir, layer_index, head_index):
    """
    加载模型，将指定层的第k个attention head的参数全部设置为0，然后保存模型。
    """
    # 加载模型和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 获取语言模型主干
    if hasattr(model, "model"):
        lang_model = model.model
    elif hasattr(model, "transformer"):
        lang_model = model.transformer
    else:
        raise ValueError("无法识别模型的语言模型结构")

    # 获取 Transformer 层序列
    if hasattr(lang_model, "layers"):
        lang_layers = lang_model.layers
    elif hasattr(lang_model, "h"):
        lang_layers = lang_model.h
    else:
        raise ValueError("无法识别 Transformer 层的位置")

    # 检查层索引是否有效
    if layer_index >= len(lang_layers) or layer_index < 0:
        raise ValueError(f"层索引 {layer_index} 超出范围 (0 至 {len(lang_layers)-1})")

    target_layer = lang_layers[layer_index]
    
    # 获取attention层
    if hasattr(target_layer, "self_attn"):
        attention = target_layer.self_attn
    elif hasattr(target_layer, "attn"):
        attention = target_layer.attn
    else:
        raise ValueError(f"无法在层 {layer_index} 中找到attention模块")

    # 获取当前的head数量
    if hasattr(model.config, "num_attention_heads"):
        num_query_heads = model.config.num_attention_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_query_heads)
    elif hasattr(model.config, "n_head"):
        num_query_heads = model.config.n_head
        num_kv_heads = num_query_heads
    else:
        raise ValueError("无法从config中获取attention head数量")

    # 检查head索引是否有效
    if head_index >= num_query_heads or head_index < 0:
        raise ValueError(f"Head索引 {head_index} 超出范围 (0 至 {num_query_heads-1})")

    head_dim = model.config.hidden_size // num_query_heads
    print(f"模型信息: 层{layer_index}, query_heads: {num_query_heads}, kv_heads: {num_kv_heads}, head维度: {head_dim}")
    print(f"将第 {head_index} 个head的参数设置为0")

    # 将指定head的参数设置为0
    with torch.no_grad():
        # 处理query, key, value权重
        if hasattr(attention, "q_proj") and hasattr(attention, "k_proj") and hasattr(attention, "v_proj"):
            # 分离的q,k,v投影
            
            # 处理query投影
            q_weight = attention.q_proj.weight.data
            q_weight_reshaped = q_weight.view(num_query_heads, head_dim, q_weight.size(-1))
            q_weight_reshaped[head_index, :, :] = 0  # 将指定head的权重设为0
            attention.q_proj.weight.data = q_weight_reshaped.view(q_weight.size())
            
            # 处理key投影 - 在GQA中需要考虑head映射
            k_weight = attention.k_proj.weight.data
            # 计算对应的kv head索引（多个query head可能共享一个kv head）
            kv_head_index = head_index // (num_query_heads // num_kv_heads)
            if kv_head_index < num_kv_heads:  # 确保索引有效
                k_weight_reshaped = k_weight.view(num_kv_heads, head_dim, k_weight.size(-1))
                k_weight_reshaped[kv_head_index, :, :] = 0  # 将对应的kv head权重设为0
                attention.k_proj.weight.data = k_weight_reshaped.view(k_weight.size())
            
            # 处理value投影 - 同key投影
            v_weight = attention.v_proj.weight.data
            if kv_head_index < num_kv_heads:  # 确保索引有效
                v_weight_reshaped = v_weight.view(num_kv_heads, head_dim, v_weight.size(-1))
                v_weight_reshaped[kv_head_index, :, :] = 0  # 将对应的kv head权重设为0
                attention.v_proj.weight.data = v_weight_reshaped.view(v_weight.size())
            
            # 处理bias（如果存在）
            if hasattr(attention.q_proj, "bias") and attention.q_proj.bias is not None:
                q_bias = attention.q_proj.bias.data.view(num_query_heads, head_dim)
                q_bias[head_index, :] = 0
                attention.q_proj.bias.data = q_bias.view(-1)
                
            if hasattr(attention.k_proj, "bias") and attention.k_proj.bias is not None and kv_head_index < num_kv_heads:
                k_bias = attention.k_proj.bias.data.view(num_kv_heads, head_dim)
                k_bias[kv_head_index, :] = 0
                attention.k_proj.bias.data = k_bias.view(-1)
                
            if hasattr(attention.v_proj, "bias") and attention.v_proj.bias is not None and kv_head_index < num_kv_heads:
                v_bias = attention.v_proj.bias.data.view(num_kv_heads, head_dim)
                v_bias[kv_head_index, :] = 0
                attention.v_proj.bias.data = v_bias.view(-1)
                
        elif hasattr(attention, "c_attn"):
            # 合并的qkv投影 (如GPT样式) - 这种情况下通常不是GQA
            qkv_weight = attention.c_attn.weight.data
            # 重塑为 [3, num_query_heads, head_dim, hidden_size]
            qkv_weight_reshaped = qkv_weight.view(3, num_query_heads, head_dim, qkv_weight.size(-1))
            qkv_weight_reshaped[:, head_index, :, :] = 0  # 将指定head的q,k,v权重都设为0
            attention.c_attn.weight.data = qkv_weight_reshaped.view(qkv_weight.size())
            
            if hasattr(attention.c_attn, "bias") and attention.c_attn.bias is not None:
                qkv_bias = attention.c_attn.bias.data.view(3, num_query_heads, head_dim)
                qkv_bias[:, head_index, :] = 0
                attention.c_attn.bias.data = qkv_bias.view(-1)
        
        # 处理输出投影 - 输出投影通常按query heads组织
        if hasattr(attention, "o_proj"):
            o_weight = attention.o_proj.weight.data
            # 输出投影权重形状通常是 [hidden_size, num_query_heads * head_dim]
            # 重塑为 [hidden_size, num_query_heads, head_dim]
            o_weight_reshaped = o_weight.view(o_weight.size(0), num_query_heads, head_dim)
            o_weight_reshaped[:, head_index, :] = 0  # 将指定head对应的输出权重设为0
            attention.o_proj.weight.data = o_weight_reshaped.view(o_weight.size())
            
        elif hasattr(attention, "c_proj"):
            o_weight = attention.c_proj.weight.data
            # 重塑为 [hidden_size, num_query_heads, head_dim]
            o_weight_reshaped = o_weight.view(o_weight.size(0), num_query_heads, head_dim)
            o_weight_reshaped[:, head_index, :] = 0  # 将指定head对应的输出权重设为0
            attention.c_proj.weight.data = o_weight_reshaped.view(o_weight.size())

    print(f"✅ 已将层 {layer_index} 的第 {head_index} 个head参数设置为0")

    # 保存模型和 tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ 模型已保存至: {output_dir}")

def zero_out_heads_sequentially(model_path, output_base_dir, layer_index, max_heads_to_process=None):
    """
    按顺序逐个将指定层的attention heads参数设置为0，每处理一个head就保存一次模型。
    """
    # 首先加载模型获取head数量信息
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype="auto",
        device_map="auto"
    )
    
    if hasattr(model.config, "num_attention_heads"):
        total_heads = model.config.num_attention_heads
    elif hasattr(model.config, "n_head"):
        total_heads = model.config.n_head
    else:
        raise ValueError("无法从config中获取attention head数量")
    
    del model  # 释放内存
    
    if max_heads_to_process is None:
        max_heads_to_process = total_heads
    
    print(f"开始逐个将层 {layer_index} 的attention heads参数设置为0")
    print(f"总head数量: {total_heads}, 处理数量: {max_heads_to_process}")
    
    # 按顺序处理heads
    for i in range(max_heads_to_process):
        output_dir = os.path.join(output_base_dir, f"zeroed_head_{i}")
        
        print(f"\n--- 将第 {i} 个head参数设置为0 ---")
        zero_out_attention_head(
            model_path, 
            output_dir, 
            layer_index, 
            i
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将模型指定层中的attention head参数设置为0")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--save_path", type=str, required=True, help="保存目录")
    parser.add_argument("--layer_index", type=int, required=True, help="要处理的层索引")
    parser.add_argument("--max_heads_to_process", type=int, default=None, 
                       help="sequential模式下最多处理的head数量，默认为总数")
    parser.add_argument("--mode", type=str, choices=["sequential", "single"], default="single",
                       help="模式: sequential(逐个处理每个head) 或 single(处理指定单个head)")
    parser.add_argument("--head_index", type=int, default=None,
                       help="single模式下要处理的head索引")

    args = parser.parse_args()
    
    if args.mode == "sequential":
        zero_out_heads_sequentially(
            args.model_path, 
            args.save_path, 
            args.layer_index, 
            args.max_heads_to_process
        )
    elif args.mode == "single":
        if args.head_index is None:
            raise ValueError("single模式下必须指定--head_index参数")
        zero_out_attention_head(
            args.model_path,
            args.save_path,
            args.layer_index,
            args.head_index
        )