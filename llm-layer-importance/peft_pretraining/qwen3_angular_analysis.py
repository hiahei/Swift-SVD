# qwen3_angular_analysis.py

import os
import numpy as np
import torch
import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import default_data_collator, Trainer, TrainingArguments
from utils.short_hf import ShortHFModel
from itertools import islice
import random

# # ------------------- 设置环境变量 -------------------
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# # ------------------- 加载模型 -------------------
# short_model = ShortHFModel(
#     # model_name="/home/swf/mixln/Llama-2-13b-hf",
#     # model_name="/home/swf/mixln/Llama-2-7b-hf",
#     # model_name="/home/swf/mixln/deepseek-7b",
#     # model_name="/home/swf/mixln/Qwen2.5-7B",
# #     model_name="/home/swf/mixln/130m_cod_lr1e-3/model_20000",
#     # model_name="/home/swf/mixln/bert-large",
#     model_name="/home/swf/mixln/Qwen3-8B",
#     layers_path="model.layers",
#     # layers_path="bert.encoder.layer",   # bert 关键
#     n_prune_layers=1,
# )
# _ = short_model.model  # 显式触发模型加载

# # ------------------- 加载数据 -------------------

# val_data = datasets.load_dataset("/home/swf/mixln/c4", "en", split="validation", streaming=True,trust_remote_code=True) #$ validation train
# # val_data = list(islice(val_data, 3000))  # 仅取前1000条
# # 3. 从中随机采样100条
# # val_data = random.sample(val_data, 20)


os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# ------------------- 加载模型 -------------------
short_model = ShortHFModel(
    model_name="Qwen/Qwen3-8B",  # 从Hugging Face加载Qwen模型
    layers_path="model.layers",
    n_prune_layers=1,
)
_ = short_model.model  # 显式触发模型加载

# ------------------- 加载数据 -------------------
val_data = load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)





# ------------------- 收集每层重要性 -------------------
idx = 0# qwen3 36  # llama2-13b 40  7b 32 # deepseek 30  qwen2.5 28 BERT-large 24 

layerss = 36  # Qwen3-8B has 36 layers
test_times = 2
alldata = []

for nidx in range(1, layerss): # 28
    idx = 0
    short_model.importances = [0 for _ in range(layerss)]
    # for cur in val_data:
    for i, cur in enumerate(tqdm(val_data)):
        if i >= 1000:
            break

        prompts = cur['text']
        _ = short_model.eval_importance(
            prompts=prompts,
            max_seq_len=256,
            stride=256,
            max_gen_len=0,
            angular=True,
            n=nidx,
        )
        idx += 1
        if idx > test_times:
            break
    alldata.append(short_model.importances)

alldata = np.array(alldata) / 3

# ------------------- 归一化函数 -------------------
def normalize_rows_robustly(data, p_low=1, p_high=99):
    normalized_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        row = data[i, :]
        val_low = np.nanpercentile(row, p_low)
        val_high = np.nanpercentile(row, p_high)

        if val_low == val_high:
            normalized_data[i, :] = 0.0
            continue

        row_clipped = np.clip(row, val_low, val_high)
        min_clipped = np.min(row_clipped)
        max_clipped = np.max(row_clipped)

        if max_clipped == min_clipped:
            normalized_data[i, :] = 0.0
        else:
            normalized_data[i, :] = (row_clipped - min_clipped) / (max_clipped - min_clipped)
    return normalized_data

# np.save("/home/swf/mixln/utils/cod1.npy", alldata)

# alldata = np.load("/home/swf/mixln/utils/qwen3_8b.npy")
alldata_normed = normalize_rows_robustly(alldata, p_low=1, p_high=99)


# ------------------- 可视化函数 -------------------
def plot_angular_distance_heatmap(data_array, L_total, vmin=0.0, vmax=1.0):
    N, M = data_array.shape
    l_indices = np.arange(M)
    n_indices = np.arange(N)
    mask = (l_indices[np.newaxis, :] + n_indices[:, np.newaxis] + 1) >= L_total

    masked_data = np.where(mask, np.nan, data_array)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_coords = np.arange(M + 1)
    y_coords = np.arange(N + 1)

    mesh = ax.pcolormesh(x_coords, y_coords, masked_data,
                         cmap='viridis_r',
                         edgecolors='white',
                         linewidth=0.2,
                         vmin=vmin,
                         vmax=vmax)

    ax.set_aspect('equal')
    ax.set_xlabel(r"Layer Index $\ell$", fontsize=10)
    ax.set_ylabel(r"Subsequent $n^{th}$ Layer", fontsize=10)
    # ax.set_title(r"(a) BERT-large Angular Distance", fontsize=12)
    # ax.set_title(r"(b)  DeepSeek-7B Angular Distance", fontsize=12)
    # ax.set_title(r"(c) Qwen3-8B Angular Distance", fontsize=12)
    # ax.set_title(r"(d) LLaMa2-13B Angular Distance", fontsize=12)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom=False, left=False)

    x_tick_interval = 5 if M > 5 else max(1, M // 10)
    y_tick_interval = 4 if N > 4 else max(1, N // 7)
    ax.set_xticks(np.arange(0, M, x_tick_interval))
    ax.set_yticks(np.arange(0, N, y_tick_interval))

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_ticks(np.arange(round(vmin, 1), vmax + 0.01, 0.1))

    plt.tight_layout()
    # plt.savefig("qwen3-8b.pdf", dpi=300, bbox_inches='tight', format='pdf')
    # plt.savefig("DeepSeek-7B2.pdf", dpi=300, bbox_inches='tight', format='pdf')
    # plt.savefig("BERT-large.pdf", dpi=300, bbox_inches='tight', format='pdf')
    # plt.savefig("llama2-13B.pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig("cod1.pdf", dpi=300, bbox_inches='tight', format='pdf')


    plt.show()

# ------------------- 绘图主函数调用 -------------------
L_total_qwen = layerss # qwen3 36  # llama2-13b 40  7b 32 # deepseek 30  qwen2.5 28 BERT-large 24 
plot_angular_distance_heatmap(alldata_normed, L_total_qwen)

# Reset matplotlib defaults
mpl.rcParams.update(mpl.rcParamsDefault)

