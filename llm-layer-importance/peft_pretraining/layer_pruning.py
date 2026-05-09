import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def remove_language_layers_and_save(model_path, output_dir, lang_layers_to_remove=None):
    """
    加载 Qwen3 模型，删除指定的语言 Transformer 层，并更新 config。
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
        raise ValueError("无法识别 Qwen3 的语言模型结构")

    # 获取 Transformer 层序列
    if hasattr(lang_model, "layers"):
        lang_layers = lang_model.layers
    elif hasattr(lang_model, "h"):
        lang_layers = lang_model.h
    else:
        raise ValueError("无法识别 Transformer 层的位置")

    # 删除指定的语言层
    if lang_layers_to_remove:
        for idx in sorted(lang_layers_to_remove, reverse=True):
            if 0 <= idx < len(lang_layers):
                del lang_layers[idx]
                print(f"✅ 已删除语言层 {idx}")
            else:
                print(f"⚠️ 层索引 {idx} 超出范围 (0 至 {len(lang_layers)-1})")

        model.config.num_hidden_layers = len(lang_layers)

    # 保存模型和 tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ 模型已保存至: {output_dir}")
    print(f"📄 当前语言层数: {model.config.num_hidden_layers}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Qwen3 模型中删除指定语言层")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen3 模型路径")
    parser.add_argument("--save_path", type=str, required=True, help="保存目录")
    parser.add_argument("--layer_index", type=str, required=True, help="要删除的层索引，例如 '0,5,23'")

    args = parser.parse_args()
    lang_layers_to_remove = [int(x.strip()) for x in args.layer_index.split(",") if x.strip().isdigit()]
    remove_language_layers_and_save(args.model_path, args.save_path, lang_layers_to_remove)