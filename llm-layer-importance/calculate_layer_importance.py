"""
简化版 Layer Importance 计算工具
支持本地模型和数据集，自动检测层数，为每一层计算importance分数

# 基本使用
python calculate_layer_importance.py \
    --model_path /path/to/your/model \
    --dataset_path /path/to/your/dataset \
    --output_path layer_importance.json \
    --max_samples 1000

# 使用角度距离
python calculate_layer_importance.py \
    --model_path /path/to/your/model \
    --dataset_path /path/to/your/dataset \
    --output_path layer_importance.json \
    --angular \
    --max_samples 1000

# 如果数据集是JSON文件
python calculate_layer_importance.py \
    --model_path /path/to/your/model \
    --dataset_path /path/to/dataset.json \
    --text_column text \
    --output_path layer_importance.json

"""

from typing import List, Optional, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
import argparse


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular: bool = False,
):
    """
    计算block influence，衡量输入和输出hidden state之间的变化程度
    
    Args:
        input_hidden_state: 输入hidden state，形状为 (B, S, D)
        output_hidden_state: 输出hidden state，形状为 (B, S, D)
        angular: 是否使用角度距离，默认False使用cosine相似度
        
    Returns:
        如果angular=False: 返回 1 - cosine_similarity（值越大表示变化越大）
        如果angular=True: 返回归一化的角度距离 arccos(sim) / π
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim


def auto_detect_layers_path(model):
    """
    自动检测模型的层路径
    
    Args:
        model: 加载的模型对象
        
    Returns:
        layers_path: 层路径字符串，例如 "model.layers" 或 "bert.encoder.layer"
    """
    # 常见的层路径模式
    common_paths = [
        "model.layers",           # LLaMA, Qwen, DeepSeek等
        "transformer.h",          # GPT-2等
        "bert.encoder.layer",     # BERT
        "roberta.encoder.layer",  # RoBERTa
        "gpt_neox.layers",        # GPT-NeoX
    ]
    
    for path in common_paths:
        modules = path.split(".")
        mod = model
        try:
            for m in modules:
                mod = getattr(mod, m)
            if hasattr(mod, '__len__') and len(mod) > 0:
                print(f"✅ 自动检测到层路径: {path}")
                return path
        except AttributeError:
            continue
    
    # 如果都没找到，尝试从config中获取
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'num_hidden_layers'):
            # 尝试常见的路径
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return "model.layers"
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                return "transformer.h"
            elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
                return "bert.encoder.layer"
    
    raise ValueError("无法自动检测层路径，请手动指定 layers_path 参数")


class LayerImportanceCalculator:
    """
    用于计算LLM各层重要性的工具类
    """
    
    def __init__(
        self, 
        model_path: str, 
        layers_path: Optional[str] = None,
        device: str = "cuda",
        local_files_only: bool = True,
    ):
        """
        初始化模型和tokenizer
        
        Args:
            model_path: 本地模型路径
            layers_path: 模型层的路径，使用点号分隔。如果为None，则自动检测
            device: 设备，默认"cuda"
            local_files_only: 是否只使用本地文件，默认True
        """
        self.device = device
        self.model_path = model_path
        
        print(f"📂 加载模型: {model_path}")
        print(f"   使用本地文件: {local_files_only}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
            device_map="auto" if device == "cuda" else None,
        )
        
        # 如果没有自动分配到设备，手动移动
        if device != "cuda" or not hasattr(self.model, 'hf_device_map'):
            self.model.to(self.device)
        
        # 自动检测或使用指定的层路径
        if layers_path is None:
            self.layers_path = auto_detect_layers_path(self.model)
        else:
            self.layers_path = layers_path
        
        # 获取层列表
        modules = self.layers_path.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod
        
        # 获取层数
        self.num_layers = len(self.layers)
        print(f"✅ 模型层数: {self.num_layers}")
        print(f"✅ 层路径: {self.layers_path}")
        
        # 初始化importance分数列表
        self.importances = [0.0 for _ in range(self.num_layers)]
        self.num_samples = 0  # 记录处理的样本数
    
    def reset_importances(self):
        """重置importance分数"""
        self.importances = [0.0 for _ in range(self.num_layers)]
        self.num_samples = 0
    
    def compute_bi(
        self, 
        hiddens: List[torch.Tensor], 
        angular: bool = False, 
        n: int = 1
    ):
        """
        计算block influence并累积到importances中
        
        Args:
            hiddens: 各层的hidden states列表
            angular: 是否使用角度距离
            n: 比较第i层输入和第i+n层输出，默认n=1（相邻层）
        """
        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i + n]
            
            if angular:
                # 使用角度距离时，只使用最后一个token的hidden state
                in_hidden = in_hidden[:, -1:]
                out_hidden = out_hidden[:, -1:]
            
            # 计算influence并累积
            influence = block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).mean().cpu().item()
            
            self.importances[i] += influence
    
    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int = 1024,
        stride: int = 256,
        max_gen_len: int = 0,
        angular: bool = False,
        n: int = 1,
        verbose: bool = True,
    ):
        """
        计算layer-wise importance的主函数
        
        Args:
            prompts: 输入文本列表
            max_seq_len: 最大序列长度，滑动窗口大小
            stride: 滑动窗口的步长
            max_gen_len: 生成的最大长度，0表示不生成（推荐用于importance计算）
            angular: 是否使用角度距离
            n: 比较第i层输入和第i+n层输出
            verbose: 是否打印进度信息
            
        Returns:
            calc_times: 处理的窗口数量
        """
        # Tokenization
        prompt_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=256,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)
        calc_times = 0
        
        # 使用滑动窗口处理长文本
        for start in range(0, max_prompt_len, stride):
            inputs = input_ids[0:1, start:start + max_seq_len]
            attn = attn_mask[0:1, start:start + max_seq_len]

            if max_gen_len == 0:
                # 不生成，只前向传播获取hidden states
                outputs = self.model(
                    input_ids=inputs.to(self.device),
                    attention_mask=attn.to(self.device),
                    output_hidden_states=True,
                )
            else:
                # 生成模式（通常不用于importance计算）
                outputs = self.model.generate(
                    input_ids=inputs.to(self.device),
                    attention_mask=attn.to(self.device),
                    max_new_tokens=max_gen_len,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            
            # 计算并累积importance
            self.compute_bi(outputs.hidden_states, angular=angular, n=n)
            calc_times += 1

        self.num_samples += calc_times
        return calc_times
    
    def get_importances(self, normalize: bool = False):
        """
        获取importance分数
        
        Args:
            normalize: 是否归一化到[0, 1]区间
            
        Returns:
            importance分数列表
        """
        if normalize and self.num_samples > 0:
            # 归一化：除以样本数，然后缩放到[0, 1]
            importances = [imp / self.num_samples for imp in self.importances]
            min_imp = min(importances)
            max_imp = max(importances)
            if max_imp > min_imp:
                importances = [(imp - min_imp) / (max_imp - min_imp) for imp in importances]
            return importances
        return self.importances.copy()
    
    def print_importances(self, top_k: Optional[int] = None):
        """
        打印importance分数
        
        Args:
            top_k: 只打印前k个最重要的层，None表示打印所有层
        """
        importances = self.get_importances()
        sorted_indices = sorted(
            range(len(importances)), 
            key=lambda i: importances[i], 
            reverse=True
        )
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        print("\n" + "="*60)
        print("Layer Importance Scores (按重要性排序):")
        print("="*60)
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"Rank {rank:2d} | Layer {idx:3d}: {importances[idx]:.6f}")
        print("="*60 + "\n")
    
    def save_importances(self, output_path: str, normalize: bool = False):
        """
        保存importance分数到文件
        
        Args:
            output_path: 输出文件路径（支持.json和.npy格式）
            normalize: 是否归一化
        """
        importances = self.get_importances(normalize=normalize)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 创建输出目录: {output_dir}")
        
        if output_path.endswith('.json'):
            # 保存为JSON格式
            result = {
                "model_path": self.model_path,
                "layers_path": self.layers_path,
                "num_layers": self.num_layers,
                "num_samples": self.num_samples,
                "importances": importances,
                "layer_importance_dict": {
                        f"layer_{i}": float(imp) for i, imp in enumerate(importances)
                    }
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Importance分数已保存到: {output_path}")
        
        elif output_path.endswith('.npy'):
            # 保存为numpy格式
            np.save(output_path, np.array(importances))
            print(f"✅ Importance分数已保存到: {output_path}")
        
        else:
            raise ValueError("输出文件格式必须是 .json 或 .npy")


def load_dataset_from_path(dataset_path: str, text_column: str = "text", split: str = "train", max_samples: Optional[int] = None):
    """
    从本地路径加载数据集
    
    Args:
        dataset_path: 数据集路径（可以是目录、JSON文件、或HuggingFace数据集名称）
        text_column: 文本列名，默认"text"
        split: 数据集分割，默认"train"
        max_samples: 最大样本数，None表示使用全部
        
    Returns:
        文本列表
    """
    print(f"📂 加载数据集: {dataset_path}")
    
    # 如果是目录或JSON文件
    if os.path.isdir(dataset_path) or dataset_path.endswith('.json'):
        if os.path.isdir(dataset_path):
            # 尝试作为HuggingFace数据集目录加载
            try:
                dataset = load_dataset(dataset_path, split=split, streaming=False)
            except:
                # 如果不是标准格式，尝试加载JSON文件
                json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
                if json_files:
                    dataset = load_dataset('json', data_files=os.path.join(dataset_path, json_files[0]))
                    if split in dataset:
                        dataset = dataset[split]
                    else:
                        dataset = dataset[list(dataset.keys())[0]]
                else:
                    raise ValueError(f"无法从目录 {dataset_path} 加载数据集")
        else:
            # 直接加载JSON文件
            dataset = load_dataset('json', data_files=dataset_path)
            if split in dataset:
                dataset = dataset[split]
            else:
                dataset = dataset[list(dataset.keys())[0]]
    else:
        # 作为HuggingFace数据集名称加载（会尝试从本地缓存加载）
        dataset = load_dataset(dataset_path, split=split, streaming=False)
    
    # 提取文本
    texts = []
    for item in tqdm(dataset, desc="提取文本"):
        if isinstance(item, dict):
            if text_column in item:
                texts.append(item[text_column])
            elif 'content' in item:
                texts.append(item['content'])
            elif 'input' in item:
                texts.append(item['input'])
            elif 'sentence' in item:
                texts.append(item['sentence'])
        elif isinstance(item, str):
            texts.append(item)
        
        if max_samples and len(texts) >= max_samples:
            break
    
    print(f"✅ 加载了 {len(texts)} 个样本")
    return texts


def main():
    parser = argparse.ArgumentParser(description="计算LLM各层的importance分数")
    parser.add_argument("--model_path", type=str, required=True, help="本地模型路径")
    parser.add_argument("--dataset_path", type=str, required=True, help="本地数据集路径或HuggingFace数据集名称")
    parser.add_argument("--output_path", type=str, default="layer_importance.json", help="输出文件路径")
    parser.add_argument("--layers_path", type=str, default=None, help="层路径，如果为None则自动检测")
    parser.add_argument("--text_column", type=str, default="text", help="数据集中的文本列名")
    parser.add_argument("--split", type=str, default="train", help="数据集分割")
    parser.add_argument("--max_samples", type=int, default=None, help="最大处理样本数")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--stride", type=int, default=256, help="滑动窗口步长")
    parser.add_argument("--angular", action="store_true", help="使用角度距离")
    parser.add_argument("--n", type=int, default=1, help="比较第i层输入和第i+n层输出")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--normalize", action="store_true", help="归一化importance分数")
    
    args = parser.parse_args()
    
    # 初始化计算器
    calculator = LayerImportanceCalculator(
        model_path=args.model_path,
        layers_path=args.layers_path,
        device=args.device,
        local_files_only=True,
    )
    
    # 加载数据集
    texts = load_dataset_from_path(
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        split=args.split,
        max_samples=args.max_samples,
    )
    
    # 批量处理数据集
    print(f"\n🔄 开始计算layer importance...")
    print(f"   总样本数: {len(texts)}")
    print(f"   使用角度距离: {args.angular}")
    print(f"   比较间隔: n={args.n}")
    
    batch_size = 1  # 每次处理一个样本
    for i, text in enumerate(tqdm(texts, desc="处理样本")):
        calculator.eval_importance(
            prompts=[text],
            max_seq_len=args.max_seq_len,
            stride=args.stride,
            angular=args.angular,
            n=args.n,
            verbose=False,
        )
    
    # 打印结果
    calculator.print_importances()
    
    # 保存结果
    calculator.save_importances(args.output_path, normalize=args.normalize)
    
    print(f"\n✅ 完成！处理了 {calculator.num_samples} 个窗口")
    print(f"   模型层数: {calculator.num_layers}")
    print(f"   结果已保存到: {args.output_path}")


if __name__ == "__main__":
    # 如果直接运行，可以使用示例代码
    import sys
    
    if len(sys.argv) > 1:
        # 命令行模式
        main()
    else:
        # 示例代码
        print("="*60)
        print("Layer Importance Calculator - 示例使用")
        print("="*60)
        print("\n使用方法:")
        print("python calculate_layer_importance.py \\")
        print("    --model_path /path/to/your/model \\")
        print("    --dataset_path /path/to/your/dataset \\")
        print("    --output_path layer_importance.json \\")
        print("    --max_samples 1000 \\")
        print("    --angular")
        print("\n或者使用HuggingFace数据集:")
        print("python calculate_layer_importance.py \\")
        print("    --model_path /path/to/your/model \\")
        print("    --dataset_path allenai/c4 \\")
        print("    --output_path layer_importance.json")
        print("="*60)