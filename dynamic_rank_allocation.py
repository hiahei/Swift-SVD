import os
import torch
import pickle as pk
import numpy as np
import argparse
import json
from tqdm import tqdm
from transformers import AutoConfig
import matplotlib.pyplot as plt
import math

# --- functions ---
def calculate_ner(sigma_tensor):
    """Compute NER (normalized effective rank via spectral entropy) from singular values."""
    if sigma_tensor is None or len(sigma_tensor) == 0:
        return 0.0
    
    s = sigma_tensor.float()
    s = s[s > 1e-9]
    if len(s) == 0: return 0.0
    p = s / torch.sum(s)
    # Spectral entropy (add a tiny value to avoid log(0)).
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    erank = torch.exp(entropy)
    ner = erank / len(sigma_tensor)
    return ner.item()

def load_importance(file_path):
    """Load importance scores from file (.pk/.pickle or .json)."""
    if file_path.endswith('.pk') or file_path.endswith('.pickle'):
        with open(file_path, 'rb') as f:
            data = pk.load(f)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported importance file format. Use .pk or .json")
    
    # If data is a dict and contains an 'importances' key, extract that list.
    if isinstance(data, dict) and 'importances' in data:
        return data['importances']
    return data

def softmax(x):
    """Generic Softmax normalization."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def dynamic_allocation_original(svd_list, importance_scores, compression_ratio, config, alpha=2.5):
    n_layers = len(svd_list)
    h_size = config.hidden_size
    
    # --- 1. Only adapt MLP dimensions; module names are already unified. ---
    model_type = getattr(config, "model_type", "").lower()
    if "opt" in model_type:
        # OPT architecture: module names are already unified in svd_list.
        # Only MLP dimensions need adaptation.
        layer_names = ['query', 'key', 'value', 'output', 'fc1', 'fc2']
        i_size = getattr(config, "ffn_dim", h_size * 4) 
    else:
        # Llama architecture
        layer_names = ['query', 'key', 'value', 'output', 'gate', 'up', 'down']
        i_size = getattr(config, "intermediate_size", 4096)

    # --- 2. Compute budgets (keep logic consistent). ---
    u_rank_attn = int(h_size * compression_ratio / 2)
    u_rank_mlp = int(i_size * h_size * compression_ratio / (i_size + h_size))
    
    type_budgets = {name: (u_rank_attn if name in ['query', 'key', 'value', 'output'] else u_rank_mlp) * n_layers 
                    for name in layer_names}

    # --- 3. Initialize base ranks and compute scores. ---
    current_ranks = {}
    full_ranks = {}
    score_lookup = {}
    
    scores_ts = torch.tensor(importance_scores)
    norm_imp = (scores_ts - scores_ts.min()) / (scores_ts.max() - scores_ts.min() + 1e-6) + 1.0

    for i in range(n_layers):
        imp_factor = np.power(norm_imp[i].item(), alpha)
        for name in layer_names:
            if name not in svd_list[i]: continue 
            
            S = svd_list[i][name].S
            f_r = len(S)
            full_ranks[(i, name)] = f_r
            
            # Base rank assignment.
            # Full-rank differences across modules are already reflected.
            # The 0.25 factor absorbs the division by 2 in compression-ratio scaling.
            current_ranks[(i, name)] = int(f_r * 0.25 * compression_ratio)
        
            epsilon = torch.sqrt(torch.sum(S ** 2)).item()  # Here S is the singular-value vector of this component.

            # Corresponds to log(constant + epsilon) in the formula.
            # log_loss = np.log(np.e + epsilon) # Used in our method (base e).
            log_loss = np.log(1.0 + epsilon) # Used in svdllmv2 (constant 1.0).

            # Compute final score.
            fnorm_loss_weighted = np.power(log_loss, 1 - alpha)
            
            # Score = I^alpha * (log(e + epsilon))^(1-alpha)
            score_lookup[(i, name)] = imp_factor * fnorm_loss_weighted

    # --- 4. Allocate remaining budget across layers within each module type. ---
    for m_type in layer_names:
        if m_type not in type_budgets: continue
        target_total_budget = type_budgets[m_type]
        used_base = sum(current_ranks[(i, m_type)] for i in range(n_layers) if (i, m_type) in current_ranks)
        rem_budget = target_total_budget - used_base
        
        if rem_budget <= 0: continue

        type_score_keys = [k for k in score_lookup.keys() if k[1] == m_type]
        total_type_score = sum(score_lookup[k] for k in type_score_keys)
        
        if total_type_score == 0: continue

        assigned_rem = 0
        for i in range(n_layers):
            if (i, m_type) not in score_lookup: continue
            share = score_lookup[(i, m_type)] / (total_type_score + 1e-9)
            extra_r = int(rem_budget * share)
            actual_add = min(extra_r, full_ranks[(i, m_type)] - current_ranks[(i, m_type)])
            current_ranks[(i, m_type)] += actual_add
            assigned_rem += actual_add
            
        # Handle integer-allocation remainder.
        leftover = int(rem_budget - assigned_rem)
        if leftover > 0:
            sorted_indices = sorted([k[0] for k in type_score_keys], 
                                    key=lambda idx: score_lookup[(idx, m_type)], reverse=True)
            for idx in sorted_indices:
                if leftover <= 0: break
                if current_ranks[(idx, m_type)] < full_ranks[(idx, m_type)]:
                    current_ranks[(idx, m_type)] += 1
                    leftover -= 1

    # --- 5. Package results (names are already unified, so direct iteration works). ---
    rank_allocation = []
    for i in range(n_layers):
        layer_dict = {'layer': i}
        for name in layer_names:
            if (i, name) not in current_ranks: continue
            
            r = current_ranks[(i, name)]
            layer_dict[f'{name}_rank'] = r
            
            S_vals = svd_list[i][name].S
            loss_val = torch.sqrt(torch.sum(S_vals[r:] ** 2)).item() if r < len(S_vals) else 0.0
            layer_dict[f'{name}_loss'] = loss_val
            
        rank_allocation.append(layer_dict)

    return rank_allocation

# --- main program ---
def main():
    parser = argparse.ArgumentParser(description="Fine-grained NER-based Dynamic Rank Allocation")
    parser.add_argument("--svd_file", type=str, required=True, help="Path to .pk file containing SVD info")
    parser.add_argument("--importance_file", type=str, required=True, help="Path to layer importance scores")
    parser.add_argument("--local_model_path", type=str, required=True, help="Path to model config")
    parser.add_argument("--output_file", type=str, default="rank_config.pk", help="Output path for rank allocation")
    parser.add_argument("--compression_ratio", type=float, required=True, help="Target compression ratio")
    parser.add_argument("--delta", type=float, required=True, help="Weight between Imp (delta) and NER (1-delta)")
    parser.add_argument("--importance_weight", type=float, default=1.0, help="Alpha for importance softmax scaling")
    
    args = parser.parse_args()

    print(f"Loading SVD results from {args.svd_file}...")
    with open(args.svd_file, 'rb') as f:
        svd_list = pk.load(f)
    
    print(f"Loading config from {args.local_model_path}...")
    config = AutoConfig.from_pretrained(args.local_model_path)
    
    print(f"Loading importance scores from {args.importance_file}...")
    importance_scores = load_importance(args.importance_file)
    
    if len(importance_scores) != len(svd_list):
        print(f"Warning: Importance scores length ({len(importance_scores)}) != Layer count ({len(svd_list)})")

    # Execute dynamic allocation.
    print("Executing dynamic_allocation_original dynamic allocation...")
    rank_allocation = dynamic_allocation_original(
        svd_list, 
        importance_scores,  
        args.compression_ratio,
        config,
        alpha=args.importance_weight,
        # delta=args.delta
    )

    # === Save JSON output (always append text mode). ===
    # json_path = args.output_file.replace('.pk', '.json')
    
    # with open(json_path, 'a', encoding='utf-8') as f:
    #     # Write a visible separator for easier manual inspection.
    #     f.write(f"\n\n{'='*30} NEW RUN {'='*30}\n")
    #     f.write(f"Metadata: Delta={args.delta}, Alpha={args.importance_weight}, Ratio={args.compression_ratio}\n")
        
    #     # Write current allocation results.
    #     json.dump(rank_allocation, f, indent=4)
        
    # print(f"✓ Results appended to: {json_path}")

    # === Save canonical Pickle output (overwrite mode). ===
    # Used by update_svd_weights.py, so it must always be clean and up to date.
    with open(args.output_file, 'wb') as f:
        pk.dump(rank_allocation, f)
        
    print(f"✓ Binary config (overwritten) saved to: {args.output_file}")
    print("✓ Done!")

if __name__ == "__main__":
    main()