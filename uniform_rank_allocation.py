"""
Uniform Rank Allocation
Assign the same compression rank to all layers (uniform compression).
"""
import torch
import pickle as pk
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoConfig


parser = argparse.ArgumentParser()
parser.add_argument('--local_model_path', type=str, required=True,
                    help='Path to local model directory')
parser.add_argument('--svd_file', type=str, required=True,
                    help='Path to SVD results file (.pk)')
parser.add_argument('--compression_ratio', type=float, required=True,
                    help='Compression ratio for Key projections (e.g., 0.5 for 50% compression)')
parser.add_argument('--output_file', type=str, default='rank_allocation_uniform.pk',
                    help='Output file for rank allocation')
args = parser.parse_args()


def compute_frobenius_loss_at_rank(svd_module, rank):
    """
    Compute Frobenius loss at a specified rank.
    
    Args:
        svd_module: IncrementalSVD object
        rank: Compression rank
    
    Returns:
        loss: Frobenius loss
    """
    S = svd_module.S  # shape: (D,)
    D = S.shape[0]
    
    if rank >= D:
        return 0.0
    
    # Loss(r) = sqrt(sum(S[r:]^2))
    # i.e., sqrt of the sum of squared discarded singular values
    loss = torch.sqrt(torch.sum(S[rank:] ** 2)).item()
    return loss


def uniform_allocation(svd_list, compression_ratio, config):
    """
    Uniformly allocate rank (all layers use the same rank).
    
    Args:
        svd_list: List of per-layer SVD results
        compression_ratio: Global compression ratio for Key (0-1)
        config: Model config object
    
    Returns:
        rank_allocation: Allocated rank per layer (key and value)
    
        Allocate uniform ranks for the 7 projection layers based on the ratio and formula.
    """

    n_layers = len(svd_list)
    
    # 1. Get dimensions from config (supports Llama, OPT, Phi-3, etc.)
    hidden_size = config.hidden_size
    # Auto-handle different intermediate_size naming
    intermediate_size = getattr(config, "intermediate_size", 
                        getattr(config, "ffn_dim", 
                        getattr(config, "d_ff", None)))
    
    if intermediate_size is None:
        # Fallback: infer from SVD matrix
        intermediate_size = svd_list[0]['gate'].V.shape[0]

    # 2. Compute truncated ranks
    # k, v, o, q logic
    rank_attn = max(1, int(hidden_size * compression_ratio / 2))
    # gate, up, down logic
    rank_mlp = max(1, int(intermediate_size * hidden_size * compression_ratio / (intermediate_size + hidden_size)))
    
    print(f"\n{'='*60}")
    print(f"Allocation Summary (Uniform Allocation)")
    print(f"{'='*60}")
    print(f"Hidden size: {hidden_size} -> Rank: {rank_attn}")
    print(f"Intermediate size:  {intermediate_size} -> Rank: {rank_mlp}")
    print(f"{'='*60}\n")
    
    # 3. Define mappings based on model architecture
    # Check model type (via config.model_type or architecture)
    model_type = getattr(config, "model_type", "").lower()
    
    # Check if model is OPT (OPT uses fc1/fc2)
    is_opt = "opt" in model_type
    
    if is_opt:
        # OPT model: q_proj, k_proj, v_proj, out_proj + fc1, fc2
        targets_config = {
            'query': rank_attn, 'key': rank_attn, 'value': rank_attn, 'output': rank_attn,
            'fc1': rank_mlp, 'fc2': rank_mlp
        }
        print(f"Detected OPT model architecture; using fc1/fc2 config\n")
    else:
        # LLaMA/Mistral/Phi models: q/k/v/o + gate, up, down
        targets_config = {
            'query': rank_attn, 'key': rank_attn, 'value': rank_attn, 'output': rank_attn,
            'gate': rank_mlp, 'up': rank_mlp, 'down': rank_mlp
        }
        print(f"Detected LLaMA/Mistral/Phi architecture; using gate/up/down config\n")

    rank_allocation = []
    
    # 4. Process each layer
    for i in tqdm(range(n_layers), desc="Computing Frobenius Loss"):
        layer_data = {'layer': i}
        
        for name, r in targets_config.items():
            # Create the key required by apply_full_svd_redemption: '{name}_rank'
            layer_data[f'{name}_rank'] = r
            
            # Compute and store the estimated Frobenius loss
            # Note: svd_list[i] must contain the 7 corresponding keys
            loss = compute_frobenius_loss_at_rank(svd_list[i][name], r)
            layer_data[f'{name}_loss'] = loss
            
        rank_allocation.append(layer_data)
        
    return rank_allocation


if __name__ == "__main__":
    
    # Load SVD results
    print(f"Loading SVD results from: {args.svd_file}")
    svd_list = pk.load(open(args.svd_file, 'rb'))

    # 1. Load config only (no model weights)
    # args.local_model_path should point to a folder containing config.json
    config = AutoConfig.from_pretrained(args.local_model_path)
    
    # Run uniform rank allocation
    rank_allocation = uniform_allocation(
        svd_list, 
        args.compression_ratio, 
        config
    )
    
    # Save results
    print(f"Saving rank allocation to: {args.output_file}")
    pk.dump(rank_allocation, open(args.output_file, 'wb'))
    
    print("\n✓ Done!")
