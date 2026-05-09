import argparse
import torch
import os
import pickle as pk
import json
from tqdm import tqdm
from ModelLoader import ModelLoader

def apply_full_svd_redemption(model, svd_list, ratio=None, rank_allocation=None, save_loss_path=None):
    """
    Replace full-layer weights with SVD-equivalent weights and record Frobenius-norm loss.
    """
    model_type = model.config.model_type
    config = model.config
    hidden_size = config.hidden_size
    intermediate_size = getattr(config, "intermediate_size", getattr(config, "ffn_dim", None))

    # Get layer list
    layers = model.model.decoder.layers if model_type == "opt" else model.model.layers
    n_layers = len(layers)
    
    loss_records = []

    print(f"\n[*] Starting SVD weight injection...")
    if rank_allocation is not None:
        print(f"[*] Mode: Adaptive Rank Allocation (from file)")
    else:
        print(f"[*] Mode: Uniform Ratio = {ratio}")

    for i in tqdm(range(n_layers), desc="Compressing layers"):
        layer = layers[i]
        layer_losses = {"layer_idx": i, "modules": {}}

        # 1. Determine per-module ranks for this layer
        if rank_allocation is not None:
            # Read attention ranks (shared across models)
            # Dynamic allocation uses _rank suffix
            r_q = rank_allocation[i]['query_rank']
            r_k = rank_allocation[i]['key_rank']
            r_v = rank_allocation[i]['value_rank']
            r_o = rank_allocation[i]['output_rank']
            
            # Read MLP ranks based on model type
            if model_type == "opt":
                r_fc1 = rank_allocation[i]['fc1_rank']
                r_fc2 = rank_allocation[i]['fc2_rank']
            else:
                r_gate = rank_allocation[i]['gate_rank']
                r_up = rank_allocation[i]['up_rank']
                r_down = rank_allocation[i]['down_rank']
        else:
            # Uniform compression
            r_attn = int(hidden_size * ratio / 2)
            r_mlp = int(intermediate_size * hidden_size * ratio / (intermediate_size + hidden_size))
            r_q = r_k = r_v = r_o = r_attn
            
            if model_type == "opt":
                r_fc1 = r_fc2 = r_mlp
            else:
                r_gate = r_up = r_down = r_mlp

        # 2. Define target module mapping
        if model_type in ["llama", "mistral", "qwen3", "phi3"]:
            targets = [
                (layer.self_attn.q_proj, 'query', r_q),
                (layer.self_attn.k_proj, 'key', r_k),
                (layer.self_attn.v_proj, 'value', r_v),
                (layer.self_attn.o_proj, 'output', r_o),
                (layer.mlp.gate_proj, 'gate', r_gate),
                (layer.mlp.up_proj, 'up', r_up),
                (layer.mlp.down_proj, 'down', r_down),
            ]
        elif model_type == "opt":
            targets = [
                (layer.self_attn.q_proj, 'query', r_q),
                (layer.self_attn.k_proj, 'key', r_k),
                (layer.self_attn.v_proj, 'value', r_v),
                (layer.self_attn.out_proj, 'output', r_o),
                (layer.fc1, 'fc1', r_fc1), 
                (layer.fc2, 'fc2', r_fc2),
            ]

        # 3. Iterate and replace
        for module, name, r in targets:
            # Clone original weights for loss computation (use float32 to avoid overflow)
            W_orig = module.weight.data.clone().float()
            
            # Load SVD results
            V = svd_list[i][name].V.to(device=W_orig.device, dtype=W_orig.dtype)
            Vr = V[:, :r] # Truncated basis matrix [D, r]
            
            # Compute equivalent weight: W' = Vr @ (Vr^T @ W)
            # Compute the inner part first for efficiency
            # print(f"Vr shape: {Vr.shape}")
            # print(f"Vr.T shape: {Vr.T.shape}")
            # print(f"W_orig shape: {W_orig.shape}")
            # Ensure (Vr.T @ W_orig) has matching inner dimensions
            W_compressed = Vr @ (Vr.T @ W_orig)
            
            # --- Compute Frobenius-norm loss ---
            original_norm = torch.norm(W_orig, p='fro')
            diff_norm = torch.norm(W_orig - W_compressed, p='fro')
            # relative_loss = (diff_norm / (original_norm + 1e-9)).item() # Avoid division by zero
            
            layer_losses["modules"][name] = {
                "rank": r,
                "original_norm": original_norm.item(),
                "diff_norm": diff_norm.item(),
            }

            # Write back weights (restore original dtype)
            module.weight.data = W_compressed.to(dtype=module.weight.dtype)

        loss_records.append(layer_losses)

    return model


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--svd_file', type=str, required=True)
    parser.add_argument('--rank_allocation', type=str)
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for compressed model')
    parser.add_argument('--ratio', type=float, help='Compression ratio')
    args = parser.parse_args()

    # 1. Ensure output directory exists
    # If you already passed a full path, this is fine as-is
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Build loss file path
    # Since output_dir includes ratio info, storing loss there is reasonable
    # loss_out_path = "svd_loss/svd_loss_results.json"
    model = ModelLoader.load_model(args.model_path)
    
    # Load SVD file
    print(f"[*] Loading SVD file: {args.svd_file}")
    with open(args.svd_file, 'rb') as f:
        svd_list = pk.load(f)

    # Load rank_allocation file (if provided)
    rank_alloc_data = args.rank_allocation
    if args.rank_allocation is not None:
        print(f"[*] Loading Rank Allocation file: {args.rank_allocation}")
        with open(args.rank_allocation, 'rb') as f:
            rank_alloc_data = pk.load(f)

    # 3. Run compression
    # Note: pass loss_out_path so apply_full_svd_redemption can write there
    model = apply_full_svd_redemption(
        model, 
        svd_list, 
        ratio=args.ratio, 
        rank_allocation=rank_alloc_data, 
        # save_loss_path=loss_out_path
    )
    
    # 4. Save model and tokenizer
    print(f"[*] Saving weights to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    tokenizer = ModelLoader.load_tokenizer(args.model_path)
    tokenizer.save_pretrained(args.output_dir)
    
    # print(f"\n✓ Done. Loss file saved to: {loss_out_path}")