# 1. File header: set TF32 precision
import torch
import concurrent.futures

torch.set_float32_matmul_precision('high')

import random
import numpy as np
import os
from utils.data_utils import build_dataset
from tqdm import tqdm
os.system('clear')
from transformers import AutoTokenizer

from transformers.models.llama.configuration_llama import LlamaConfig
from ModelLoader import ModelLoader

from torch.utils.data import DataLoader
import torch
from torch.nn import DataParallel
import argparse
import json
import pickle as pk
from utils.svd import IncrementalSVD
from utils.acc_dataset import get_acc_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True,
                    help='Root directory of the dataset')
parser.add_argument('--model_name', type=str, required=True,
                    help='model name')
parser.add_argument('--data_name', type=str, required=True,
                    help='dataset name')
parser.add_argument('--max_samples', type=int, default=2048,
                    help='maximum number of samples to process')
parser.add_argument('--seqlen', type=int, default=1024,
                    help='Sequence length for continuous datasets')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for sampling')
args = parser.parse_args()


def is_continuous_dataset(dataset_name):
    """
    Check if dataset is continuous text type (WikiText2, C4)
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        bool: True if continuous dataset, False otherwise
    """
    return dataset_name in ['WikiText2', 'C4']

def svd_array_update(model_type, svd_list, n_layers, mode='flush'):
    """
    Update SVD objects in svd_list based on mode ('flush' or 'finalize').
    'flush' clears buffers to free memory; 'finalize' runs the final SVD.
    """
    assert mode in ['flush', 'finalize']
    
    pbar = tqdm(range(n_layers), desc='flushing layers', total=n_layers, ncols=150)
    for i in pbar:
        # Process key, value, query, output, up, gate, down for each layer
        # OPT models use fc1/fc2 instead of up/gate/down
        if model_type == "opt":
            for svd_type in ['key', 'value', 'query', 'output', 'fc1', 'fc2']:
                if svd_type in svd_list[i]:
                    pbar.set_description(f'{mode} layer {i}, {svd_type}')
                    
                    if mode == 'flush':
                        svd_list[i][svd_type].flush()
                    else:
                        svd_list[i][svd_type].finalize()
        else:
            for svd_type in ['key', 'value', 'query', 'output', 'up', 'gate', 'down']:
                if svd_type in svd_list[i]:
                    pbar.set_description(f'{mode} layer {i}, {svd_type}')
                    
                    if mode == 'flush':
                        svd_list[i][svd_type].flush()
                    else:
                        svd_list[i][svd_type].finalize()

    return svd_list


def get_hook(layer_idx, name, svd_list):
    """
    Create a hook to capture layer outputs and feed IncrementalSVD.
    """
    def hook(module, input, output):
        data = output.detach()
        svd_list[layer_idx][name].feed(data)
    return hook



if __name__=="__main__":

    model_name = args.model_name
    data_name = args.data_name
    
    # Batch processing config - H100 80GB optimization (balance speed and memory)
    BATCH_SIZE = 8  # Layers per batch (8 layers x 7 matrices = 56 SVD objects)
    FLUSH_THRESHOLD = 20000  # Balance GPU utilization and memory
    
    print(f"\n{'='*80}")
    print(f"H100 config: batch layers={BATCH_SIZE}, flush threshold={FLUSH_THRESHOLD}")
    print(f"{'='*80}\n")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True
    )

    model = ModelLoader.load_model(
        model_name, 
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    # model = DataParallel(model)  # Use DataParallel
    model.eval()
    
    # Get layer settings
    configure = model.config # model is loaded
    print(configure)
    # Identify model type
    model_type = configure.model_type

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Process data

    # Load and prepare dataset based on type
    is_continuous = is_continuous_dataset(data_name)
    is_acc_dataset = data_name in ["openbookqa", "arc_challenge", "arc_easy", "hellaswag", "winogrande", "piqa", "mathqa"]
    print(f"Dataset type: {data_name} | Mode: {'ACC' if is_acc_dataset else ('Continuous' if is_continuous else 'Conversation')}")
    
    if is_continuous:
        # For continuous datasets (WikiText2, C4): load locally, merge all text, then sample
        from datasets import load_dataset
        
        if data_name == "WikiText2":
            local_data_dir = os.path.join(args.data_root, "wikitext-2-raw-v1")
            train_path = os.path.join(local_data_dir, "wikitext-train.arrow")
            data_files = {"train": train_path}
            traindata = load_dataset("arrow", data_files=data_files)["train"]
            separator = "\n\n"  # WikiText2 uses double newline
            # Merge all text with appropriate separator (following reference code)
            print(f"Merging {len(traindata)} text samples with separator '{repr(separator)}'...")
            tot_text = separator.join(traindata["text"])
            print(f"Total text length: {len(tot_text)} chars")
            tokenized_samples = []
            random.seed(args.seed)
            for _ in range(args.max_samples):
                i = random.randint(0, len(tot_text) - args.seqlen - 1)
                j = i + args.seqlen * 10
                # Tokenize the sampled text chunk
                trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
                # Take first seqlen tokens - store directly as dict with input_ids
                inp = trainenc.input_ids[:, :args.seqlen]
                attention_mask = torch.ones_like(inp)
                tokenized_samples.append({"input_ids": inp, "attention_mask": attention_mask})
            
            dataset_to_process = tokenized_samples   

        elif data_name == "C4":
            # C4 has multiple shard files
            train_dir = args.data_root
            import glob
            train_files = sorted(glob.glob(os.path.join(train_dir, "c4-train-*.arrow")))
            print(f"Found {len(train_files)} C4 training files: {[os.path.basename(f) for f in train_files]}")
            data_files = {"train": train_files}
            traindata = load_dataset("arrow", data_files=data_files)["train"]
            separator = "\n\n"  # C4 uses double newline
            # Merge all text with appropriate separator (following reference code)
            print(f"Merging {len(traindata)} text samples with separator '{repr(separator)}'...")
            tot_text = separator.join(traindata["text"])
            print(f"Total text length: {len(tot_text)} chars")
            tokenized_samples = []
            for _ in range(args.max_samples):
                i = random.randint(0, len(tot_text) - args.seqlen * 10 - 1)
                j = i + args.seqlen * 10
                # Tokenize the sampled text chunk
                trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
                # Take first seqlen tokens - store directly as dict with input_ids
                inp = trainenc.input_ids[:, :args.seqlen]
                attention_mask = torch.ones_like(inp)
                tokenized_samples.append({"input_ids": inp, "attention_mask": attention_mask})
            
            dataset_to_process = tokenized_samples            

    # ACC dataset handling
    elif data_name in ["openbookqa", "arc_challenge", "arc_easy", "hellaswag", "winogrande", "piqa", "mathqa"]:
        # For QA datasets: use local arrow files
        # dataset_to_process = get_acc_dataset(data_name, args.data_root, tokenizer, args.max_samples, args.seqlen, args.seed)   
        # Optionally skip seqlen truncation
        print(f"Start looping over 7 datasets...")
        target_datasets = ["openbookqa", "arc_challenge", "arc_easy", "hellaswag", "winogrande", "piqa", "mathqa"]
        samples_per_dataset = 256
        dataset_to_process = [] # Final combined list
        # Call get_acc_dataset and extend the combined list
        for d_name in target_datasets:
            data_root = os.path.join(args.data_root, d_name)
            acc_samples = get_acc_dataset(d_name, data_root, tokenizer, samples_per_dataset, args.seqlen, args.seed)
            dataset_to_process.extend(acc_samples)

    else:
        # For conversation datasets (Alpaca, etc.): use build_dataset
        my_dataset = build_dataset(
            name=data_name,
            root=args.data_root,
            split='train',
        )
        dataset_to_process = [my_dataset[i] for i in range(min(args.max_samples, len(my_dataset)))]
    
    # Process each sample
    for idx, sample in enumerate(tqdm(dataset_to_process, desc="Processing samples")):
        if is_continuous:
            # Continuous dataset: sample already contains tokenized input_ids
            model_inputs = {
                "input_ids": sample["input_ids"].to(model.device),
                "attention_mask": sample["attention_mask"].to(model.device)
            }
        
        elif is_acc_dataset:
            # ACC dataset: sample already contains tokenized input_ids
            model_inputs = {
                "input_ids": sample["input_ids"].to(model.device),
                "attention_mask": sample["attention_mask"].to(model.device)
            }

        else:
            # Conversation dataset: sample is a dict with 'input' and 'output'
            input_text = sample['input']
            output_text = sample['output']
            
            # Use chat template for conversation data
            messages = [{"role": "user", "content": input_text}]
            
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    max_length=args.seqlen,
                    truncation=True,
                    fix_mistral_regex=True,
                    chat_template="{% for message in messages %}{{ '### Instruction:' if message['role'] == 'user' else '### Response:' }}\n{{ message['content'] }}\n{% endfor %}"
                )
            except (TypeError, AttributeError, Exception) as e:
                # Fallback: some models do not support chat_template or args
                print(f"Warning: apply_chat_template failed for {model_type}: {e}")
                print(f"   Using raw text format instead")
                text = f"### Instruction:\n{input_text}\n### Response:\n"
            
            # Append output text
            text += output_text
            # Tokenize for conversation data
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Handle head_dim calculations and dimension settings by model type
    if model_type in ["llama"]: # 1
        # Attention layers
        head_dim = configure.hidden_size // configure.num_attention_heads
        # Key/value head count (usually equal; query head count may be larger)
        # Compatible with Llama/Llama2/Llama3
        num_key_value_heads = getattr(configure, "num_key_value_heads", configure.num_attention_heads)
        kv_merge_dim = head_dim * num_key_value_heads
        qo_merge_dim = head_dim * configure.num_attention_heads
        # MLP layers
        gate_up_dim = configure.intermediate_size
        down_dim = configure.hidden_size

    elif model_type in ["mistral"]:
        # Attention layers
        head_dim = configure.hidden_size // configure.num_attention_heads
        num_key_value_heads = configure.num_key_value_heads
        kv_merge_dim = head_dim * num_key_value_heads # k total dim; aligned with v, q, o
        qo_merge_dim = head_dim * configure.num_attention_heads
        # MLP layers
        gate_up_dim = configure.intermediate_size
        down_dim = configure.hidden_size
    
    elif model_type in ["opt"]:
        # Attention layers
        # OPT head_dim is hidden_size // num_attention_heads
        head_dim = configure.hidden_size // configure.num_attention_heads
        # OPT uses standard MHA; KV and QO dims match
        kv_merge_dim = configure.hidden_size 
        qo_merge_dim = configure.hidden_size
        # MLP layers (note field name differences)
        gate_up_dim = configure.ffn_dim  # OPT uses ffn_dim
        down_dim = configure.hidden_size
    
    elif model_type in ["qwen3"]:
        # Attention layers: Qwen3 head_dim may differ; compute q/k/v/o dims separately
        head_dim = getattr(configure, "head_dim", configure.hidden_size // configure.num_attention_heads)
        num_key_value_heads = getattr(configure, "num_key_value_heads", configure.num_attention_heads)
        kv_merge_dim = head_dim * num_key_value_heads
        qo_merge_dim = head_dim * configure.num_attention_heads
        # MLP layers
        gate_up_dim = configure.intermediate_size
        down_dim = configure.hidden_size
    
    elif model_type in ["phi3"]:
        # Attention layers
        head_dim = configure.hidden_size // configure.num_attention_heads # 3072 // 32 = 96
        num_key_value_heads = configure.num_key_value_heads # 32
        kv_merge_dim = head_dim * num_key_value_heads # 96 * 32 = 3072
        # MLP layers
        gate_up_dim = configure.intermediate_size # 8192
        down_dim = configure.hidden_size # 3072
    
    else:
        raise NotImplementedError(f"Model type {model_type} not supported yet.")

    # Query/Output dims default to qo_merge_dim
    # Qwen3 o_proj outputs hidden_size; handle separately
    # qwen3-4b differs
    query_dim = qo_merge_dim
    output_dim = qo_merge_dim
    if model_type == "qwen3":
        output_dim = configure.hidden_size
    
    # Number of layers
    n_layers = configure.num_hidden_layers
    
    # Use base model name only (strip path)
    model_safe_name = os.path.basename(model_name.rstrip('/')).replace('/', '-')
    
    # Create save directory
    save_dir = f'./svd_list/{data_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Final SVD list for all layers
    final_svd_list = [{} for _ in range(n_layers)]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Batch processing over layers
    print(f"\nStarting batch processing of {n_layers} layers, {BATCH_SIZE} per batch...")
    
    for batch_start in range(0, n_layers, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_layers)
        batch_layers = list(range(batch_start, batch_end))
        print(f"\n{'='*80}")
        print(f"Batch: layers {batch_start} to {batch_end-1} ({len(batch_layers)} layers)")
        print(f"{'='*80}")
        
        # Initialize svd_list for this batch
        batch_svd_list = {}
        for i in batch_layers:
            if model_type == "opt":
                batch_svd_list[i] = {
                    'key': IncrementalSVD(dim=kv_merge_dim, device='cuda', name=f'key_{i}'),
                    'value': IncrementalSVD(dim=kv_merge_dim, device='cuda', name=f'value_{i}'),
                    'query': IncrementalSVD(dim=query_dim, device='cuda', name=f'query_{i}'),
                    'output': IncrementalSVD(dim=output_dim, device='cuda', name=f'output_{i}'),
                    'fc1': IncrementalSVD(dim=gate_up_dim, device='cuda', name=f'fc1_{i}'),
                    'fc2': IncrementalSVD(dim=down_dim, device='cuda', name=f'fc2_{i}'),
                }
            else:
                batch_svd_list[i] = {
                    'key': IncrementalSVD(dim=kv_merge_dim, device='cuda', name=f'key_{i}'),
                    'value': IncrementalSVD(dim=kv_merge_dim, device='cuda', name=f'value_{i}'),
                    'query': IncrementalSVD(dim=query_dim, device='cuda', name=f'query_{i}'),
                    'output': IncrementalSVD(dim=output_dim, device='cuda', name=f'output_{i}'),
                    'up': IncrementalSVD(dim=gate_up_dim, device='cuda', name=f'up_{i}'),
                    'gate': IncrementalSVD(dim=gate_up_dim, device='cuda', name=f'gate_{i}'),
                    'down': IncrementalSVD(dim=down_dim, device='cuda', name=f'down_{i}'),
                }
        
        # Register hooks for this batch
        hooks = []
        for i in batch_layers:
            if model_type in ["llama", "mistral", "qwen3", "phi3"]:
                layer = model.model.layers[i]
                hooks.append(layer.self_attn.q_proj.register_forward_hook(get_hook(i, 'query', batch_svd_list)))
                hooks.append(layer.self_attn.k_proj.register_forward_hook(get_hook(i, 'key', batch_svd_list)))
                hooks.append(layer.self_attn.v_proj.register_forward_hook(get_hook(i, 'value', batch_svd_list)))
                hooks.append(layer.self_attn.o_proj.register_forward_hook(get_hook(i, 'output', batch_svd_list)))
                hooks.append(layer.mlp.gate_proj.register_forward_hook(get_hook(i, 'gate', batch_svd_list)))
                hooks.append(layer.mlp.up_proj.register_forward_hook(get_hook(i, 'up', batch_svd_list)))
                hooks.append(layer.mlp.down_proj.register_forward_hook(get_hook(i, 'down', batch_svd_list)))
            elif model_type == "opt":
                layer = model.model.decoder.layers[i]
                hooks.append(layer.self_attn.q_proj.register_forward_hook(get_hook(i, 'query', batch_svd_list)))
                hooks.append(layer.self_attn.k_proj.register_forward_hook(get_hook(i, 'key', batch_svd_list)))
                hooks.append(layer.self_attn.v_proj.register_forward_hook(get_hook(i, 'value', batch_svd_list)))
                hooks.append(layer.self_attn.out_proj.register_forward_hook(get_hook(i, 'output', batch_svd_list)))
                hooks.append(layer.fc1.register_forward_hook(get_hook(i, 'fc1', batch_svd_list)))
                hooks.append(layer.fc2.register_forward_hook(get_hook(i, 'fc2', batch_svd_list)))
        
        # Iterate dataset and collect data for this batch
        print(f"Collecting batch data...")
        for idx, sample in enumerate(tqdm(dataset_to_process, desc=f"Batch {batch_start}-{batch_end-1}")):
            if is_continuous:
                model_inputs = {
                    "input_ids": sample["input_ids"].to(model.device),
                    "attention_mask": sample["attention_mask"].to(model.device)
                }
            elif is_acc_dataset:
                model_inputs = {
                    "input_ids": sample["input_ids"].to(model.device),
                    "attention_mask": sample["attention_mask"].to(model.device)
                }
            else:
                input_text = sample['input']
                output_text = sample['output']
                messages = [{"role": "user", "content": input_text}]
                
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        max_length=args.seqlen,
                        truncation=True,
                        fix_mistral_regex=True,
                        chat_template="{% for message in messages %}{{ '### Instruction:' if message['role'] == 'user' else '### Response:' }}\n{{ message['content'] }}\n{% endfor %}"
                    )
                except (TypeError, AttributeError, Exception) as e:
                    print(f"Warning: apply_chat_template failed for {model_type}: {e}")
                    text = f"### Instruction:\n{input_text}\n### Response:\n"
                
                text += output_text
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # Forward computation
            with torch.no_grad():
                try:
                    outputs = model(**model_inputs, use_cache=False)
                
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
        
        # Remove hooks for this batch
        for h in hooks:
            h.remove()
        
        # Finalize this batch
        print(f"\nFinalizing batch {batch_start}-{batch_end-1}...")
        for i in batch_layers:
            for svd_type in batch_svd_list[i].keys():
                batch_svd_list[i][svd_type].finalize()
        
        # Save batch results into final list
        for i in batch_layers:
            final_svd_list[i] = batch_svd_list[i]
        
        # Save intermediate results (optional, for recovery)
        intermediate_path = f'{save_dir}/{model_safe_name}_{data_name}_batch_{batch_start}_{batch_end-1}.pk'
        with open(intermediate_path, 'wb') as f:
            pk.dump(batch_svd_list, f)
        print(f"✓ Batch results saved to: {intermediate_path}")
        
        # Clean up memory for this batch
        del batch_svd_list
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Show GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        
        print(f"✓ Batch {batch_start}-{batch_end-1} done, memory cleaned\n")
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Save final full results
    final_path = f'{save_dir}/{model_safe_name}_{data_name}_svd_list_{args.max_samples}_{args.seqlen}_s{args.seed}.pk'
    with open(final_path, 'wb') as f:
        pk.dump(final_svd_list, f)
    print(f"\n{'='*80}")
    print(f"✓ All layers completed. Final results saved to: {final_path}")
    print(f"{'='*80}")
