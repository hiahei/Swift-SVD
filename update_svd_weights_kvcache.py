import argparse
import torch
import os
import pickle as pk
import json
from tqdm import tqdm
from ModelLoader import ModelLoader
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import transformers.models.llama.modeling_llama as llama_modeling

# ------------------------------------------------------------------------------
# Implement weight replacement after SVD-equivalent compression

# ==================== 1. Utilities: dynamic RoPE and KV replication ====================

def get_rope_function(orig_attn_module):
    """Dynamically get RoPE function to match transformers versions."""
    if hasattr(orig_attn_module, "_apply_rotary_pos_emb"):
        return orig_attn_module._apply_rotary_pos_emb
    if hasattr(llama_modeling, "apply_rotary_pos_emb"):
        return llama_modeling.apply_rotary_pos_emb
    
    # Fallback: define a standard rotation implementation
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    return apply_rotary_pos_emb

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Standard GQA repeat logic."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_to_tensor(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    if cos.dim() == x.dim() - 1:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def _normalize_rank_entry(rank_entry):
    """Accept either `query_rank` style or plain `query` style keys."""
    normalized = {}
    for key, value in rank_entry.items():
        if key.endswith("_rank"):
            normalized[key[:-5]] = value
        else:
            normalized[key] = value
    return normalized


def _get_rank_entry(rank_allocation, layer_idx):
    if isinstance(rank_allocation, list):
        return rank_allocation[layer_idx]
    if isinstance(rank_allocation, dict):
        if layer_idx in rank_allocation:
            return rank_allocation[layer_idx]
        if str(layer_idx) in rank_allocation:
            return rank_allocation[str(layer_idx)]
    raise TypeError("rank_allocation must be a list or dict indexed by layer")


def _ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# ==================== 2. SVD LLaMA Attention ====================

class SVD_LlamaAttention(nn.Module):
    def __init__(self, config, ranks, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_out_dim = self.num_heads * self.head_dim
        self.kv_out_dim = self.num_key_value_heads * self.head_dim
        self.o_in_dim = self.num_heads * self.head_dim
        
        # Internal projections: x -> u_proj (rank) -> v_proj (output)
        # Note: u_proj corresponds to VT [r, in_dim], v_proj corresponds to U [out_dim, r]
        self.q_u_proj = nn.Linear(self.hidden_size, ranks['query'], bias=False)
        self.q_v_proj = nn.Linear(ranks['query'], self.q_out_dim, bias=False)
        
        self.k_u_proj = nn.Linear(self.hidden_size, ranks['key'], bias=False)
        self.k_v_proj = nn.Linear(ranks['key'], self.kv_out_dim, bias=False)
        
        self.v_u_proj = nn.Linear(self.hidden_size, ranks['value'], bias=False)
        self.v_v_proj = nn.Linear(ranks['value'], self.kv_out_dim, bias=False)
        
        self.o_u_proj = nn.Linear(self.o_in_dim, ranks['output'], bias=False)
        self.o_v_proj = nn.Linear(ranks['output'], self.hidden_size, bias=False)
        
        self.rotary_emb = None
        self._apply_rotary_pos_emb = None

    # This version stores RoPE-applied full K cache and low-rank V cache.
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=True,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        
        # print(f"DEBUG: Layer {self.layer_idx} forward called, use_cache={use_cache}")
        bsz, q_len, _ = hidden_states.size()

        # Resolve positions for the current query tokens
        if position_ids is None:
            if cache_position is not None:
                query_position_ids = (
                    cache_position.unsqueeze(0)
                    if cache_position.dim() == 1
                    else cache_position
                )
            else:
                query_position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
        else:
            query_position_ids = position_ids
        
        # 1. Query is always reconstructed immediately.
        query_states = self.q_v_proj(self.q_u_proj(hidden_states))
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Mirror svd_llama_kvcache.py's idea: keep the low-rank K/V
        # intermediates available for cache storage, while reconstructing the
        # current-step full K/V activations immediately.
        key_latent = self.k_u_proj(hidden_states)
        value_latent = self.v_u_proj(hidden_states)
        key_states = self.k_v_proj(key_latent)
        value_states = self.v_v_proj(value_latent)

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                kv_seq_len += past_key_values.get_seq_length()
            else:
                kv_seq_len += past_key_values[0].shape[1]

        # 3. Apply RoPE only to the current-step reconstructed Q/K, matching the
        # original svd_llama_kvcache.py idea before cached low-rank states are restored.
        if self.rotary_emb is not None:
            if position_embeddings is not None:
                cos_q, sin_q = position_embeddings
            else:
                cos_q, sin_q = self.rotary_emb(query_states, query_position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos_q, sin_q)

        # 4. Cache adaptation:
        #    - Cache stores low-rank K/V intermediates, not reconstructed full K/V.
        if use_cache and past_key_values is not None:
            if hasattr(past_key_values, "update"):
                cache_kwargs = {}
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                key_latent, value_latent = past_key_values.update(
                    key_latent, value_latent, self.layer_idx, cache_kwargs
                )
            elif isinstance(past_key_values, tuple):
                key_latent = torch.cat([past_key_values[0], key_latent], dim=1)
                value_latent = torch.cat([past_key_values[1], value_latent], dim=1)
                past_key_values = (key_latent, value_latent)
        elif use_cache:
            past_key_values = (key_latent, value_latent)

        # --- Print cache stats ---
        if use_cache:
            key_shape = key_latent.shape
            value_shape = value_latent.shape
            total_mem_mb = (
                key_latent.nelement() * key_latent.element_size()
                + value_latent.nelement() * value_latent.element_size()
            ) / (1024**2)

            # if self.layer_idx == 0:
            #     print(f"\n[Layer {self.layer_idx}] Hybrid KV Cache Info:")
            #     print(f" - K Shape (RoPE/full): {list(key_shape)}")
            #     print(f" - V Shape (latent): {list(value_shape)} (Rank: {value_shape[-1]})")
            #     print(f" - Memory Usage: {total_mem_mb:.4f} MB")

        # 5. Restore full K/V from the cached low-rank intermediates before the
        # actual attention calculation, matching svd_llama_kvcache.py's flow.
        if use_cache:
            total_len = key_latent.shape[1]
        else:
            total_len = key_states.shape[-2]

        if use_cache and value_latent.shape[1] != total_len:
            raise ValueError(
                f"K/V cache length mismatch in layer {self.layer_idx}: "
                f"K={total_len}, V={value_latent.shape[1]}"
            )

        if use_cache:
            key_states = self.k_v_proj(key_latent)
            value_states = self.v_v_proj(value_latent)
            key_states = key_states.view(bsz, total_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, total_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 6. Compute attention in full space after restoring K/V from cache.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :total_len]
            if causal_mask.dtype == torch.bool:
                # SDPA-style masks use True for visible positions. Convert them to
                # additive masking before the eager softmax below.
                attn_weights = attn_weights.masked_fill(
                    ~causal_mask,
                    torch.finfo(attn_weights.dtype).min,
                )
            else:
                attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 8. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_v_proj(self.o_u_proj(attn_output))

        # LlamaAttention returns a 2-tuple: (attn_output, attn_weights)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None

# ==================== 3. SVD LLaMA MLP ====================

class SVD_LlamaMLP(nn.Module):
    def __init__(self, config, ranks):
        super().__init__()
        self.gate_u_proj = nn.Linear(config.hidden_size, ranks['gate'], bias=False)
        self.gate_v_proj = nn.Linear(ranks['gate'], config.intermediate_size, bias=False)
        self.up_u_proj = nn.Linear(config.hidden_size, ranks['up'], bias=False)
        self.up_v_proj = nn.Linear(ranks['up'], config.intermediate_size, bias=False)
        self.down_u_proj = nn.Linear(config.intermediate_size, ranks['down'], bias=False)
        self.down_v_proj = nn.Linear(ranks['down'], config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.gate_v_proj(self.gate_u_proj(x))
        up = self.up_v_proj(self.up_u_proj(x))
        return self.down_v_proj(self.down_u_proj(self.act_fn(gate) * up))

# ==================== 4. Core application function ====================

def apply_svd_compression(model, svd_list, ratio=None, rank_allocation=None, save_loss_path=None):
    config = model.config
    layers = model.model.layers
    loss_records = []

    for i in tqdm(range(len(layers)), desc="Compressing Layers"):
        layer = layers[i]
        
        # 1. Determine ranks
        if rank_allocation is not None:
            ranks = _normalize_rank_entry(_get_rank_entry(rank_allocation, i))
        else:
            r_attn = int(config.hidden_size * ratio / 2)
            r_mlp = int(config.intermediate_size * config.hidden_size * ratio / (config.intermediate_size + config.hidden_size))
            ranks = {k: (r_mlp if k in ['gate', 'up', 'down'] else r_attn) for k in ['query', 'key', 'value', 'output', 'gate', 'up', 'down']}

        # 2. Create new modules
        new_attn = SVD_LlamaAttention(config, ranks, layer_idx=i)
        new_mlp = SVD_LlamaMLP(config, ranks)
        
        # 3. Decompose and fill weights (only when svd_list is provided)
        if svd_list is not None:
            module_map = {
                'query': (layer.self_attn.q_proj, new_attn.q_u_proj, new_attn.q_v_proj),
                'key':   (layer.self_attn.k_proj, new_attn.k_u_proj, new_attn.k_v_proj),
                'value': (layer.self_attn.v_proj, new_attn.v_u_proj, new_attn.v_v_proj),
                'output':(layer.self_attn.o_proj, new_attn.o_u_proj, new_attn.o_v_proj),
                'gate':  (layer.mlp.gate_proj,   new_mlp.gate_u_proj, new_mlp.gate_v_proj),
                'up':    (layer.mlp.up_proj,     new_mlp.up_u_proj,   new_mlp.up_v_proj),
                'down':  (layer.mlp.down_proj,   new_mlp.down_u_proj, new_mlp.down_v_proj),
            }

            layer_loss = {"layer_idx": i, "modules": {}}
            for name, (orig_m, u_m, v_m) in module_map.items():
                W = orig_m.weight.data.float()
                r = ranks[name]
                
                # Load precomputed V from svd_list
                V = svd_list[i][name].V.to(W.device, dtype=torch.float32)
                Vr = V[:, :r]

                # The SVD basis is learned from forward-hook outputs, so Vr spans
                # the output space of the linear layer and must project W on rows.
                if Vr.shape[0] != W.shape[0]:
                    raise ValueError(
                        f"Expected output-space basis for {name}, got V shape {Vr.shape} and W shape {tuple(W.shape)}"
                    )
                U_reduced = Vr
                VT = Vr.T @ W
                
                # Write weights (Linear expects [out, in])
                u_m.weight.data.copy_(VT.to(orig_m.weight.dtype))
                v_m.weight.data.copy_(U_reduced.to(orig_m.weight.dtype))
                
                # Record error
                error = torch.norm(W - (U_reduced @ VT), p='fro') / torch.norm(W, p='fro')
                layer_loss["modules"][name] = {"rank": r, "rel_error": error.item()}
            
            loss_records.append(layer_loss)

        # 4. Attach attributes
        # new_attn.rotary_emb = layer.self_attn.rotary_emb
        # 1. Try to locate rotary_emb
        if hasattr(layer.self_attn, 'rotary_emb'):
            # Older or specific versions
            new_attn.rotary_emb = layer.self_attn.rotary_emb
        elif hasattr(model.model, 'rotary_emb'):
            # Newer versions store rotary_emb on LlamaModel
            new_attn.rotary_emb = model.model.rotary_emb
        else:
            # Fallback: try to grab it from another layer
            print("Warning: Could not find rotary_emb in standard locations. Checking layers[0]...")
            new_attn.rotary_emb = model.model.layers[0].self_attn.rotary_emb

        new_attn._apply_rotary_pos_emb = get_rope_function(layer.self_attn)
        
        # 5. Replace modules
        layer.self_attn = new_attn
        layer.mlp = new_mlp
        torch.cuda.empty_cache()

    return model

# ----------------------------------------------------------------------------------------------------------


def apply_full_svd(model, svd_list, ratio=None, rank_allocation=None, save_loss_path=None):
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
            print(f"Vr shape: {Vr.shape}")
            print(f"Vr.T shape: {Vr.T.shape}")
            print(f"W_orig shape: {W_orig.shape}")
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

    # 4. Save loss file
    if save_loss_path:
        _ensure_parent_dir(save_loss_path)
        with open(save_loss_path, 'w') as f:
            json.dump(loss_records, f, indent=4)
        print(f"[*] Full-layer loss records saved to: {save_loss_path}")

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
    loss_out_path = os.path.join(args.output_dir, "svd_loss_results.json")
    model = ModelLoader.load_model(args.model_path)
    
    # Load SVD file
    print(f"[*] Loading SVD file: {args.svd_file}")
    with open(args.svd_file, 'rb') as f:
        svd_list = pk.load(f)

    # Load rank_allocation file (if provided)
    rank_alloc_data = None
    if args.rank_allocation is not None:
        print(f"[*] Loading Rank Allocation file: {args.rank_allocation}")
        with open(args.rank_allocation, 'rb') as f:
            rank_alloc_data = pk.load(f)

    # 3. Run compression
    compressed_model = apply_svd_compression(
        model, 
        svd_list, 
        ratio=args.ratio, 
        rank_allocation=rank_alloc_data, 
        save_loss_path=loss_out_path
    )

    # 4. Save model and tokenizer
    print(f"[*] Saving results to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    compressed_model = compressed_model.to(torch.bfloat16) # or torch.bfloat16

    model_name = os.path.basename(args.model_path).replace('/', '_')
    state_dict_path = os.path.join(args.output_dir, f'{model_name}_weights_r{args.ratio}.pt')
    torch.save(compressed_model.state_dict(), state_dict_path)

    tokenizer = ModelLoader.load_tokenizer(args.model_path)
    tokenizer.save_pretrained(args.output_dir)
