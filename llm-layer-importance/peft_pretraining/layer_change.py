import os
import argparse
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

def replace_layers_and_save(source_model_path, target_model_path, output_dir, layers_to_replace=None):
    """
    加载 Llama-Distilled 和 Llama 模型，将 Llama-Distilled 的指定层替换到 Llama 模型中，并保存。
    
    Args:
        source_model_path: Llama-Distilled 模型路径 (源模型，提供替换层)
        target_model_path: Llama 模型路径 (目标模型，被替换层)
        output_dir: 保存替换后模型的目录
        layers_to_replace: 要替换的层索引列表
    """
    print(f"📂 加载源模型 (Llama-Distilled): {source_model_path}")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        torch_dtype="auto",
        device_map="cpu"  # 先加载到CPU避免GPU内存不足
    )
    
    print(f"📂 加载目标模型 (Llama): {target_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype="auto",
        device_map="cpu"
    )
    
    # 加载tokenizer (使用目标模型的tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    # 获取模型主干
    def get_model_layers(model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        else:
            raise ValueError("无法识别模型的 Transformer 层结构")

    source_layers = get_model_layers(source_model)
    target_layers = get_model_layers(target_model)
    
    print(f"🔍 源模型层数: {len(source_layers)}")
    print(f"🔍 目标模型层数: {len(target_layers)}")
    
    # 检查层数是否兼容
    if len(source_layers) != len(target_layers):
        print(f"⚠️  警告: 源模型层数 ({len(source_layers)}) 与目标模型层数 ({len(target_layers)}) 不同")
        max_layers = min(len(source_layers), len(target_layers))
        print(f"🔧 将限制操作范围到前 {max_layers} 层")
    else:
        max_layers = len(target_layers)

    # 替换指定的层
    if layers_to_replace:
        # 过滤掉无效的层索引（如999等用于表示不替换的标记）
        valid_layers = [layer_idx for layer_idx in layers_to_replace if 0 <= layer_idx < max_layers]
        
        if valid_layers:
            print(f"🔄 开始替换层: {valid_layers}")
            for layer_idx in valid_layers:
                print(f"  📝 替换第 {layer_idx} 层...")
                
                # 深拷贝源模型的层到目标模型
                target_layers[layer_idx] = copy.deepcopy(source_layers[layer_idx])
                
                print(f"  ✅ 第 {layer_idx} 层替换完成")
        else:
            print("ℹ️  没有有效的层索引，将保存原始模型（不替换任何层）")
            
        # 报告跳过的无效索引
        invalid_layers = [layer_idx for layer_idx in layers_to_replace if layer_idx < 0 or layer_idx >= max_layers]
        if invalid_layers:
            print(f"  ⚠️  跳过无效层索引: {invalid_layers} (有效范围: 0 至 {max_layers-1})")
    else:
        print("⚠️  未指定要替换的层")

    # 保存替换后的模型
    print(f"💾 保存模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    target_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ 层替换完成！模型已保存至: {output_dir}")
    print(f"📄 替换的层: {layers_to_replace}")
    print(f"📄 最终模型层数: {len(target_layers)}")
    
    # 清理内存
    del source_model, target_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Llama-Distilled 模型的指定层替换到 Llama 模型中")
    parser.add_argument("--source_model_path", type=str, required=True, 
                       help="源模型路径 (Llama-Distilled)")
    parser.add_argument("--target_model_path", type=str, required=True, 
                       help="目标模型路径 (Llama)")
    parser.add_argument("--save_path", type=str, required=True, 
                       help="保存替换后模型的目录")
    parser.add_argument("--layer_index", type=str, required=True, 
                       help="要替换的层索引，例如 '0,5,23'")

    args = parser.parse_args()
    
    # 解析层索引
    layers_to_replace = [int(x.strip()) for x in args.layer_index.split(",") if x.strip().isdigit()]
    
    print("🚀 开始层替换操作...")
    print(f"📍 源模型 (Llama-Distilled): {args.source_model_path}")
    print(f"📍 目标模型 (Llama): {args.target_model_path}")
    print(f"📍 输出目录: {args.save_path}")
    print(f"📍 替换层索引: {layers_to_replace}")
    
    replace_layers_and_save(
        source_model_path=args.source_model_path,
        target_model_path=args.target_model_path,
        output_dir=args.save_path,
        layers_to_replace=layers_to_replace
    )