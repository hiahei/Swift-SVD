import re
import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
import random
from utils.data_utils import get_test_data
import os
import sys
import transformers

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import os
import torch
import argparse
import pickle as pk
from datasets import load_dataset
from transformers import AutoTokenizer

from ModelLoader import ModelLoader

from update_svd_weights_forward import SVD_LlamaAttention, SVD_LlamaMLP
from update_svd_weights_forward import apply_svd_compression

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _tensor_nbytes(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return 0
    return tensor.nelement() * tensor.element_size()


def _legacy_kv_cache_nbytes(past_key_values):
    total = 0
    for layer_cache in past_key_values:
        if not isinstance(layer_cache, (list, tuple)) or len(layer_cache) < 2:
            continue
        total += _tensor_nbytes(layer_cache[0])
        total += _tensor_nbytes(layer_cache[1])
    return total


def _kv_cache_nbytes(past_key_values):
    if past_key_values is None:
        return 0

    # Newer transformers caches store one cache layer per decoder block.
    if hasattr(past_key_values, "layers"):
        total = 0
        for layer in past_key_values.layers:
            total += _tensor_nbytes(getattr(layer, "keys", None))
            total += _tensor_nbytes(getattr(layer, "values", None))
        return total

    # Older cache layouts may expose the tensors directly.
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return _legacy_kv_cache_nbytes(zip(past_key_values.key_cache, past_key_values.value_cache))

    if hasattr(past_key_values, "to_legacy_cache"):
        return _legacy_kv_cache_nbytes(past_key_values.to_legacy_cache())

    if isinstance(past_key_values, (list, tuple)):
        return _legacy_kv_cache_nbytes(past_key_values)

    return 0


def _print_attn_backend(model, prefix):
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", "unknown")
    first_attn = None
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        first_attn = model.model.layers[0].self_attn

    attn_cls = type(first_attn).__name__ if first_attn is not None else "unknown"
    print(f"{prefix} attention config: {attn_impl}")
    print(f"{prefix} first layer attention class: {attn_cls}")
    if "SVD" in attn_cls:
        print(f"{prefix} note: custom SVD attention is active and already uses eager-style matmul/softmax attention.")


def _describe_tensor(name, tensor):
    if tensor is None:
        return f"{name}: None"

    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = finite_mask.sum().item()
    total_count = detached.numel()

    if finite_count > 0:
        finite_vals = detached[finite_mask].float()
        return (
            f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
            f"finite={finite_count}/{total_count}, min={finite_vals.min().item():.6f}, "
            f"max={finite_vals.max().item():.6f}, mean={finite_vals.mean().item():.6f}"
        )

    return (
        f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"finite=0/{total_count}, all values are non-finite"
    )


# Evaluate perplexity (PPL) on specified datasets
@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], data_root=None, model_seq_len=2048, batch_size=32, device="cuda", seed=42):
    # model.to(device) # Do not use on multi-GPU setups
    model.eval()
    # Get the device of the model entry (usually cuda:0)
    first_device = next(model.parameters()).device

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ppls = {}
    for dataset in datasets:
        bad_logits_reported = False
        bad_loss_reported = False
        # Special handling for Alpaca dataset
        if 'alpaca' in dataset.lower():
            import json
            data_root = os.path.join(data_root, "Alpaca") # data_root is the base directory
            alpaca_path = os.path.join(data_root, "alpaca_data.json")
            with open(alpaca_path, "r", encoding="utf-8") as f:
                alpaca_data = json.load(f)
            rng = random.Random(seed)
            rng.shuffle(alpaca_data)
            alpaca_data = alpaca_data[:1000]
            
            total_nll = 0.0
            total_tokens = 0
            
            print(f"Evaluating Alpaca dataset ({len(alpaca_data)} samples)...")
            for sample_idx, d in enumerate(tqdm(alpaca_data), start=1):
                instruction = d.get('instruction', '')
                input_text = d.get('input', '')
                output_text = d.get('output', '')
                
                # ---- Build prompt + reference answer (for PPL) ----
                user_content = input_text if not instruction else f"{instruction}\n\n{input_text}"
                messages = [{"role": "user", "content": user_content}]
                
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        chat_template="{% for message in messages %}{{ '### Instruction:' if message['role'] == 'user' else '### Response:' }}\n{{ message['content'] }}\n{% endfor %}"
                    )
                except:
                    # Fallback to a simple format if apply_chat_template fails
                    if input_text:
                        text = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
                    else:
                        text = f"### Instruction:\n{instruction}\n### Response:\n"
                
                text += output_text
                
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=model_seq_len).to(first_device)
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
                
                # ---- PPL ----
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    lm_logits = outputs.logits
                    shift_logits = None
                    shift_labels = None
                    loss = None
                    if not torch.isfinite(lm_logits).all():
                        if not bad_logits_reported:
                            print(f"[PPL Debug] Non-finite logits detected on dataset={dataset}, sample_idx={sample_idx}")
                            print(_describe_tensor("lm_logits", lm_logits))
                            bad_logits_reported = True
                    else:
                        shift_logits = lm_logits[:, :-1, :].contiguous().float()
                        shift_labels = input_ids[:, 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        if not torch.isfinite(loss).all():
                            if not bad_loss_reported:
                                print(f"[PPL Debug] Non-finite token loss detected on dataset={dataset}, sample_idx={sample_idx}")
                                print(_describe_tensor("shift_logits", shift_logits))
                                print(_describe_tensor("loss", loss))
                                bad_loss_reported = True
                        else:
                            total_nll += loss.sum().item()
                            total_tokens += loss.numel()
                
                # Clear CUDA memory
                del encodings, input_ids, attention_mask, outputs, lm_logits
                if shift_logits is not None:
                    del shift_logits
                if shift_labels is not None:
                    del shift_labels
                if loss is not None:
                    del loss
                torch.cuda.empty_cache()
            
            if total_tokens == 0:
                print(f"Warning: total_tokens is 0 for dataset {dataset}; check debug output above.")
                ppl = float('inf')
            else:
                avg_loss = total_nll / total_tokens
                print(f"avg_loss for dataset {dataset}: {avg_loss}")
                ppl = np.exp(avg_loss)
            ppls[dataset] = ppl
            
        else:
            # Use the original logic for other datasets
            test_loader = get_test_data(
                dataset,
                data_root,
                tokenizer,
                seq_len=model_seq_len,
                batch_size=batch_size,
                seed=seed,
            )
            # Accumulate instead of storing a list to reduce memory usage
            total_loss = 0.0
            total_count = 0
            for batch_idx, batch in enumerate(tqdm(test_loader), start=1):
                # batch = batch.to(device)
                batch = batch.to(first_device)
                
                output = model(batch, use_cache=False)
                lm_logits = output.logits
                shift_logits = None
                shift_labels = None
                loss = None

                if not torch.isfinite(lm_logits).all():
                    if not bad_logits_reported:
                        print(f"[PPL Debug] Non-finite logits detected on dataset={dataset}, batch_idx={batch_idx}")
                        print(_describe_tensor("batch", batch))
                        print(_describe_tensor("lm_logits", lm_logits))
                        bad_logits_reported = True
                else:
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = batch[:, 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    if not torch.isfinite(loss).all():
                        if not bad_loss_reported:
                            print(f"[PPL Debug] Non-finite token loss detected on dataset={dataset}, batch_idx={batch_idx}")
                            print(_describe_tensor("shift_logits", shift_logits))
                            print(_describe_tensor("loss", loss))
                            bad_loss_reported = True
                    else:
                        total_loss += loss.sum().item()
                        total_count += loss.numel()
                # Clear CUDA memory after each batch
                del batch, output, lm_logits
                if shift_logits is not None:
                    del shift_logits
                if shift_labels is not None:
                    del shift_labels
                if loss is not None:
                    del loss
                torch.cuda.empty_cache()
            if total_count == 0:
                print(f"Warning: total_count is 0 for dataset {dataset}; check for non-finite logits or overlong inputs.")
                ppl = float('inf')
            else:
                avg_loss = total_loss / total_count
                print(f"avg_loss for dataset {dataset}: {avg_loss}")
                ppl = np.exp(avg_loss)

            ppls[dataset] = ppl
        
    print(f"\nEvaluation complete! PPL per dataset: {ppls}")
    return ppls

# Evaluate generation efficiency on the specified dataset
@torch.no_grad()
def eff_eval(model, tokenizer, data_root, dataset='C4', input_seq_len=64, generated_len=1024, batch_size=1, device="cuda"):

    model.eval()
    cuda_device = next(model.parameters()).device
    total_time = 0.0
    token_num = 0
    end_memory = 0
    start_memory = weight_memory = 0
    num_batches_to_fetch = 2
    dataset = dataset[0] if isinstance(dataset, list) else dataset
    test_loader = get_test_data(
        dataset,
        data_root,
        tokenizer,
        seq_len=input_seq_len,
        batch_size=batch_size,
    )
    weight_memory = torch.cuda.memory_allocated(cuda_device)

    if generated_len <= 0:
        print("generated_len must be positive to evaluate decode throughput.")
        return

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated(cuda_device)

        try:
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
            start_time = time.time()

            generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
                top_k=50,
                min_new_tokens=generated_len,
                max_new_tokens=generated_len,
                top_p=0.95,
                temperature=1,
            )
            torch.cuda.synchronize(cuda_device)
            end_time = time.time()
            end_memory = max(torch.cuda.max_memory_allocated(cuda_device), end_memory)

            batch_time = end_time - start_time
            if torch.isfinite(generation_output).all():
                total_time += batch_time
                print(f"time: {batch_time}")

            del generation_output, batch

        except RuntimeError as e:
            print(f"Error during generation: {e}")
            torch.cuda.empty_cache()  # Avoid OOM

    weight_memory_gb = weight_memory / (1024 ** 3)

    print(f"Weight Memory: {weight_memory_gb:.2f} GB")
    
    if total_time > 0:
        print(f"Throughput: {token_num / total_time:.2f} tokens/sec")
    else:
        print("Throughput could not be calculated due to errors.")

# Evaluate accuracy
def evaluate_commonsense(compressed_model_loaded, tokenizer, eval_dataset):
    import lm_eval
    from lm_eval import tasks
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.registry import ALL_TASKS
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    
    compressed_model = HFLM(pretrained=compressed_model_loaded, tokenizer=tokenizer)

    eval_dataset_name = eval_dataset
    
    results = evaluator.simple_evaluate(
        model=compressed_model,
        tasks=[eval_dataset_name],
        # trust_remote_code=True
    )
    
    return results['results']


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer"""
    print(f"Loading model from: {model_path}")
    model = ModelLoader.load_model(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        # device_map="cuda",
    )
    model.eval()
    _print_attn_backend(model, "[Original Model]")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    return model, tokenizer

def load_model_and_tokenizer_compressed(model_path):
    print(f"[*] Loading compressed model from: {model_path}")
    
    # 1. Load the saved dict with torch.load
    # map_location='cpu' helps avoid OOM, then move to CUDA as needed
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    
    # 2. Extract objects from the dict
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    
    # 3. Set eval mode (important after SVD compression; disable dropout, etc.)
    model.eval()
    
    # 4. Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        # model = model.half()
        model = model.to(torch.bfloat16)
        
    print("[*] Model and Tokenizer loaded successfully.")
    _print_attn_backend(model, "[Compressed Checkpoint]")
    return model, tokenizer


def load_model_and_tokenizer_compressed(ratio, weights_pt_path, base_model_path, rank_allocation_file=None):
    """
        Load model and tokenizer for compressed model with SVD applied.
    """
    # Ensure ratio is a float
    ratio = float(ratio)
    print(f"[*] Using specified compression ratio: {ratio}")

    
    print(f"[*] Initializing base structure: {base_model_path}")
    model = ModelLoader.load_model(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="eager",
    )
    _print_attn_backend(model, "[Compressed Base Model]")

    # 2. Apply SVD structure transformation
    if rank_allocation_file:
        print(f"[*] Rebuilding SVD structure (rank_allocation={rank_allocation_file})...")
        with open(rank_allocation_file, "rb") as f:
            rank_allocation = pk.load(f)
        model = apply_svd_compression(
            model=model,
            svd_list=None,
            ratio=None,
            rank_allocation=rank_allocation,
        )
    else:
        print(f"[*] Rebuilding SVD structure (ratio={ratio})...")
        model = apply_svd_compression(model=model, svd_list=None, ratio=ratio)

    # 3. Load weights
    print(f"[*] Loading weights file: {weights_pt_path}")
    state_dict = torch.load(weights_pt_path, map_location='cuda', weights_only=True)
    
    # 4. Load weights into the model
    model.load_state_dict(state_dict, strict=True)
    
    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = model.to(torch.bfloat16).cuda()
    model.eval()
    
    print(f"[*] Compressed model loaded successfully (Ratio: {ratio})")
    _print_attn_backend(model, "[Compressed Final Model]")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate model using evaluater.py functions")
    parser.add_argument('--ratio', type=float,
                        help='Compression ratio')    
    parser.add_argument('--original_model', type=str, required=True,
                        help='Original model path')
    parser.add_argument('--full_model', type=str, required=True,
                        help='Full compressed model load path')
    parser.add_argument('--compressed_model', type=str,
                        help='Compressed model path')
    parser.add_argument('--rank_allocation_file', type=str, default=None,
                        help='Path to rank_allocation .pk file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--data_name', type=str, default='wikitext2',
                        help='Dataset name: wikitext2, c4, etc.')
    parser.add_argument('--eval_type', type=str, default='ppl',
                        choices=['ppl', 'eff', 'acc', 'all'],
                        help='Evaluation type: ppl (perplexity), eff (efficiency), acc (accuracy), or all')
    parser.add_argument('--input_seq_len', type=int, default=256,
                        help='Input sequence length for efficiency evaluation')
    parser.add_argument('--model_seq_len', type=int, default=2048,
                        help='Model sequence length for PPL evaluation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--gen_seq_len', type=int, default=1024,
                        help='Generation sequence length for efficiency evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for evaluation')

    args = parser.parse_args()
    
    print("="*80)
    print("Model Evaluation using evaluater.py")
    print("="*80)
    
    # Load model and tokenizer
    print("\n[Step 1] Loading model and tokenizer...")

    # Choose compressed model or original model based on compressed_model arg
    if args.compressed_model:
        print(f"DEBUG: args.ratio type is {type(args.ratio)}, value is {args.ratio}")
        model, tokenizer = load_model_and_tokenizer_compressed(
            args.ratio,
            args.compressed_model,
            args.original_model,
            rank_allocation_file=args.rank_allocation_file,
        )
    else:
        # Load end-to-end model
        model, tokenizer = load_model_and_tokenizer(args.full_model)
    
    # Ensure no residual hooks interfere with inference
    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
    
    # Run evaluations
    print(f"\n[Step 2] Running evaluations...")
    print("="*80)
    
    # ---------------------------------------------------   
    # Evaluate perplexity (PPL)
    if args.eval_type in ['ppl', 'all']:
        print(f"\n[PPL Evaluation]")
        print(f"  Model sequence length: {args.model_seq_len}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Dataset: {args.data_name}")
        print("-"*80)
        ppl_eval(
            model=model,
            tokenizer=tokenizer,
            data_root=args.data_root,
            datasets=[args.data_name.lower()],
            model_seq_len=args.model_seq_len,
            batch_size=args.batch_size,
            device="cuda",
            seed=args.seed,
        )
    
    # ---------------------------------------------------   
    # Evaluate efficiency
    if args.eval_type in ['eff', 'all']:
        print(f"\n[Efficiency Evaluation]")
        print(f"  Generation sequence length: {args.gen_seq_len}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Dataset: {args.data_name}")
        print("-"*80)
        
        eff_eval(
            model=model,
            tokenizer=tokenizer,
            data_root=args.data_root,
            dataset=[args.data_name.lower()],  # Now passing as a list
            input_seq_len=args.input_seq_len,
            generated_len=args.gen_seq_len,
            batch_size=args.batch_size,
            device="cuda",
        )

    # ---------------------------------------------------   
    # Evaluate accuracy
    if args.eval_type in ['acc', 'all']:
        valid_datasets = {
            "arc_easy", "arc_challenge", "openbookqa", "winogrande", "piqa", "hellaswag", "mathqa"
        }

        for eval_dataset_name in valid_datasets:
            import time
            result = evaluate_commonsense(model, tokenizer, eval_dataset_name)
            print(f"{eval_dataset_name}: {result}")
            # Clear CUDA memory and add delay between evaluations to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            time.sleep(5)

    print("\n" + "="*80)
    print("✓ Evaluation completed!")
    print("="*80)

if __name__ == "__main__":
    main()
