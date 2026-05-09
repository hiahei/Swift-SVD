import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


def get_acc_dataset(data_name, data_root, tokenizer, max_samples=None, seqlen=2048, seed=42):
    """
    Process QA datasets record by record and return tokenized samples.
    """
    data_name = data_name.lower()
    dataset_path = os.path.join(data_root, f"{data_name}-train.arrow")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    # 1. Load raw Arrow data
    traindata = load_dataset("arrow", data_files={"train": dataset_path})["train"]
    
    # 2. If max_samples is set, truncate (or randomly sample)
    if max_samples is not None and max_samples < len(traindata):
        # Use shuffle(seed=seed) to make randomness deterministic
        # Then select the first max_samples entries
        traindata = traindata.shuffle(seed=seed).select(range(max_samples))
        print(f"Randomly sampled {max_samples} samples with seed {seed}")
    
    tokenized_samples = []

    print(f"Processing {len(traindata)} samples from {data_name}...")

    # 3. Concatenate fields per record and tokenize
    for item in traindata:
        if data_name in ["openbookqa", "arc_challenge", "arc_easy"]:
            q = item.get('question_stem') or item.get('question', '')
            choices = " ".join(item['choices']['text'])
            text = f"Question: {q} Choices: {choices} Answer: {item.get('answerKey', '')}"
            
        elif data_name == "hellaswag":
            endings = " ".join(item['endings'])
            text = f"{item['ctx']} Endings: {endings}"
            
        elif data_name == "piqa":
            text = f"Goal: {item['goal']} Solution 1: {item['sol1']} Solution 2: {item['sol2']}"
            
        elif data_name == "winogrande":
            # Winogrande uses '_' for blanks; concatenate options directly
            text = f"Context: {item['sentence']} Options: {item['option1']}, {item['option2']}"
            
        elif data_name == "mathqa":
            text = f"Problem: {item['Problem']} Rationale: {item['Rationale']} Options: {item['options']}"
        else:
            # Auto-concatenate all string fields
            text = " ".join([str(v) for v in item.values() if isinstance(v, str)])
            
        # 4. Tokenize and handle length
        # padding='max_length' ensures consistent length
        # truncation=True ensures not exceeding seqlen
        enc = tokenizer(
            text, 
            truncation=False, 
            # max_length=seqlen, 
            padding=False, 
            return_tensors="pt"
        )
        
        tokenized_samples.append({
            "input_ids": enc.input_ids, 
            "attention_mask": enc.attention_mask
        })

    return tokenized_samples