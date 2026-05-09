import inspect
import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


DATASET_REGISTRY = {}

def register_dataset(cls=None, *, name=None):
    def do_register(cls_):
        key = name or cls_.__name__
        DATASET_REGISTRY[key] = cls_
        return cls_

    if cls is None:
        return do_register  # decorator used with arguments
    else:
        return do_register(cls)  # decorator used without arguments

def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered.")
    else:
        print(f'Using {name} dataset.')
    
    dataset_cls = DATASET_REGISTRY[name]
    sig = inspect.signature(dataset_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return dataset_cls(**filtered_kwargs)

def get_calib_train_data(dataset_name, tokenizer, nsamples, seqlen=2048, seed=42, batch_size=1, local_dataset_path=None):
    import random
    random.seed(seed)
    # Support local dataset paths
    if dataset_name in ["wikitext2", "WikiText2"]:
        local_data_dir = os.path.join(local_dataset_path, "wikitext-2-raw-v1")
        train_path = os.path.join(local_data_dir, "wikitext-train.arrow")
        data_files = {"train": train_path}
        traindata = load_dataset("arrow", data_files=data_files)["train"]
        separator = "\n\n"  # WikiText2 uses double newline
        # Merge all text with appropriate separator
        print(f"Merging {len(traindata)} text samples with separator '{repr(separator)}'...")
        tot_text = separator.join(traindata["text"])
        print(f"Total text length: {len(tot_text)} chars")
    elif dataset_name in ["C4", "c4"]:
        import glob
        train_files = sorted(glob.glob(os.path.join(local_dataset_path, "c4-train-*.arrow")))
        print(f"Found {len(train_files)} C4 training files: {[os.path.basename(f) for f in train_files]}")
        data_files = {"train": train_files}
        traindata = load_dataset("arrow", data_files=data_files)["train"]
        separator = "\n\n"  # C4 uses space
        print(f"Merging {len(traindata)} text samples with separator '{repr(separator)}'...")
        tot_text = separator.join(traindata["text"])
        print(f"Total text length: {len(tot_text)} chars")        
    
    # Randomly sample and tokenize (following reference code logic)
    # Sample seqlen*10 chars, tokenize, then take first seqlen tokens
    print(f"Sampling {nsamples} chunks of {seqlen} tokens...")
    tokenized_samples = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen * 10 - 1)
        j = i + seqlen * 10
        # Tokenize the sampled text chunk
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        # Take first seqlen tokens - store directly as dict with input_ids
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        tokenized_samples.append({"input_ids": inp, "attention_mask": attention_mask})
    
    train_dataset = tokenized_samples
    return train_dataset


def get_test_data(name, data_root, tokenizer, seq_len=2048, batch_size=1, seed=42):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(separator, samples, tokenizer, seq_len, field_name, seed=None):
        test_ids = tokenizer(separator.join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        indices = list(range(nsamples))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(indices)

        for i in indices:
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    def process_data_aligned(samples, tokenizer, seq_len, field_name):
        test_ids_list = []
        test_mask_list = []
        test_label_list = []
        # Ensure pad_token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Processing {len(samples)} samples individually for test...")
        for i in range(len(samples)):
            text = samples[i][field_name]
            if len(text.strip()) == 0:
                continue
            # Encode per sample: truncate to seq_len and pad to seq_len
            # Ensure tokenizer has pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                # Some models need pad_token_id set manually
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding='max_length'
            )
            ids = enc.input_ids[0]
            mask = enc.attention_mask[0]
            # Build labels: set padding positions to -100
            label = ids.clone()
            label[mask == 0] = -100
            test_ids_list.append(ids)
            test_mask_list.append(mask)
            test_label_list.append(label)
        # Wrap into a dict
        data_dict = {
            "input_ids": torch.stack(test_ids_list),
            "attention_mask": torch.stack(test_mask_list),
            "labels": torch.stack(test_label_list)
        }
        return IndexDataset(data_dict)
    ####
    
    name = name.lower()
    if 'wikitext2' in name:
        # Original logic: concatenate into one large text, then split
        data_root = os.path.join(data_root, "WikiText2")
        local_data_dir = os.path.join(data_root, "wikitext-2-raw-v1")
        test_path = os.path.join(local_data_dir, "wikitext-test.arrow")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"WikiText2 test file not found at {test_path}")
        
        data_files = {"test": test_path}
        raw_datasets = load_dataset("arrow", data_files=data_files)
        testdata = raw_datasets["test"]
        separator = "\n\n"
        test_dataset = process_data(separator, testdata, tokenizer, seq_len, 'text', seed=seed)
    
    elif 'c4' in name:
        data_root = os.path.join(data_root, "C4")
        val_path = os.path.join(data_root, "c4-validation.arrow")
        
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"C4 validation file not found at {val_path}")
        
        data_files = {"test": val_path}
        raw_datasets = load_dataset("arrow", data_files=data_files)
        testdata = raw_datasets["test"]
        if len(testdata) > 2000:
            testdata = testdata.shuffle(seed=seed).select(range(2000))
        separator = "\n\n"
        test_dataset = process_data(separator, testdata, tokenizer, seq_len, 'text', seed=seed)
    elif 'alpaca' in name:
        import json
        data_root = os.path.join(data_root, "Alpaca")
        alpaca_path = os.path.join(data_root, "alpaca_data.json")
        with open(alpaca_path, "r", encoding="utf-8") as f:
            alpaca_data = json.load(f)
        rng = random.Random(seed)
        rng.shuffle(alpaca_data)
        alpaca_data = alpaca_data[:1000]
        
        test_ids_batch = []
        test_mask_batch = []
        test_label_batch = []

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        for item in alpaca_data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # 1. Define two parts: prompt-only and full dialogue
            if input_text:
                prompt_no_resp = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
            else:
                prompt_no_resp = f"### Instruction:\n{instruction}\n### Response:\n"
            
            full_prompt = prompt_no_resp + output_text
            
            # 2. Get prompt length (no BOS to avoid double counting)
            # We only need prompt token count
            prompt_enc = tokenizer(prompt_no_resp, return_tensors="pt", add_special_tokens=False)
            prompt_len = prompt_enc.input_ids.shape[1]
            
            # 3. Encode full dialogue (with padding and truncation)
            enc = tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=seq_len, 
                padding='max_length'
            )
            
            ids = enc.input_ids[0]
            mask = enc.attention_mask[0]
            label = ids.clone()

            # 4. Key change: loss masking
            # a. Handle padding: set padding positions to -100
            label[mask == 0] = -100
            
            # b. Handle prompt: set prompt positions to -100
            # Note: if the model uses BOS token, prompt_len may need +1 (add_special_tokens=True usually handles it)
            # Be conservative: mask the first prompt_len tokens
            label[:prompt_len] = -100 
            
            test_ids_batch.append(ids)
            test_mask_batch.append(mask)
            test_label_batch.append(label)

        test_data_dict = {
            "input_ids": torch.stack(test_ids_batch),
            "attention_mask": torch.stack(test_mask_batch),
            "labels": torch.stack(test_label_batch)
        }
        test_dataset = IndexDataset(test_data_dict)
        
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataset
