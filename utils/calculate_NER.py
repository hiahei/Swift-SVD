import pickle as pk
import torch
import os
import sys
from transformers import AutoConfig

def calculate_ner_from_sigma(sigma_tensor, max_possible_rank):
    """
    Compute NER (normalized effective rank via spectral entropy) from singular values.
    """
    s = sigma_tensor.float()
    s = s[s > 1e-9]
    if len(s) <= 1: return 0.0
    
    p = s / torch.sum(s)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    erank = torch.exp(entropy)
    
    # Dynamic correction: ensure the denominator is not smaller than the effective rank
    actual_max_rank = max(max_possible_rank, len(s))
    
    ner = erank / actual_max_rank
    return ner.item()

def get_mandatory_rank_map(config):
    """
    Determine the theoretical maximum rank for each matrix from the model config.
    """
    h_size = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = h_size // n_heads
    
    # --- Key compatibility handling ---
    if hasattr(config, "intermediate_size"):
        inter_size = config.intermediate_size
    elif hasattr(config, "ffn_dim"):
        inter_size = config.ffn_dim
    else:
        # Default fallback
        inter_size = h_size * 4 
    
    kv_max_rank = n_kv_heads * head_dim

    # Define all possible module names and their max ranks
    rank_map = {
        # Attention
        'query': h_size,      'q_proj': h_size,
        'key': kv_max_rank,    'k_proj': kv_max_rank,
        'value': kv_max_rank,  'v_proj': kv_max_rank,
        'output': h_size,     'o_proj': h_size,
        
        # MLP (compatible with Llama gate/up/down and OPT fc1/fc2)
        'gate': min(h_size, inter_size),      'gate_proj': min(h_size, inter_size),
        'up': min(h_size, inter_size),        'up_proj': min(h_size, inter_size),
        'down': min(h_size, inter_size),      'down_proj': min(h_size, inter_size),
        'fc1': min(h_size, inter_size),       'fc2': min(h_size, inter_size), # OPT naming
    }
    
    print(f"[*] Architecture summary: Hidden={h_size}, Intermediate={inter_size}, KV_Rank={kv_max_rank}")
    return rank_map

def run_ner_analysis(model_path, svd_list_path):
    # 1. Load model config
    print(f"[*] Loading model config: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Build rank map
    rank_map = get_mandatory_rank_map(config)

    # 3. Load SVD data
    if not os.path.exists(svd_list_path):
        raise FileNotFoundError(f"SVD data file not found: {svd_list_path}")
        
    print(f"[*] Loading SVD data: {svd_list_path}")
    with open(svd_list_path, 'rb') as f:
        final_svd_list = pk.load(f)

    # 4. Compute NER for each layer
    results = {}
    # Dynamic init: use module names from the SVD file
    sample_layer = final_svd_list[0]
    for m_name in sample_layer.keys():
        results[m_name] = []

    for layer_idx, modules in enumerate(final_svd_list):
        for m_name, svd_obj in modules.items():
            sigma = getattr(svd_obj, 'S', svd_obj)
            
            # Auto-map OPT fc1/fc2 to common display columns
            # Mapping: fc1 -> Up/Gate, fc2 -> Down
            if m_name not in rank_map:
                print(f"[Warning] Module '{m_name}' not in rank_map; using hidden_size as default.")
                max_r = config.hidden_size
            else:
                max_r = rank_map[m_name]
                
            score = calculate_ner_from_sigma(sigma, max_r)
            results[m_name].append(score)

    return results

def format_output(ner_data):
    """
    Format and print results, compatible with OPT and Llama architectures.
    """
    # Define display columns and fallback keys
    columns = {
        "Query": ['q_proj', 'query'],
        "Key": ['k_proj', 'key'],
        "Value": ['v_proj', 'value'],
        "Output": ['o_proj', 'output'],
        "Up": ['up_proj', 'up', 'fc1'],    # OPT fc1 acts as Up
        "Gate": ['gate_proj', 'gate'],     # OPT usually has no Gate
        "Down": ['down_proj', 'down', 'fc2'] # OPT fc2 acts as Down
    }

    # Print header
    header = f"{'Layer':<6}"
    for col in columns.keys():
        header += f" | {col:<8}"
    print(header)
    print("-" * (len(header) + 5))

    # Get layer count
    first_list = next(iter(ner_data.values()))
    num_layers = len(first_list)

    for i in range(num_layers):
        row = f"L{i:<5}"
        for col, keys in columns.items():
            val = "N/A"
            for k in keys:
                if k in ner_data:
                    val = f"{ner_data[k][i]:.4f}"
                    break
            row += f" | {val:<8}"
        print(row)

# --- main entry ---
if __name__ == "__main__":
    MODEL_PATH = 'models/opt-6.7b'  # Replace with your model path
    SVD_PK_FILE = 'svd_list/C4/opt-6.7b_C4_svd_list_256_2048_s31.pk'

    try:
        final_ner_results = run_ner_analysis(MODEL_PATH, SVD_PK_FILE)
        format_output(final_ner_results)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Run failed: {e}")
        traceback.print_exc()