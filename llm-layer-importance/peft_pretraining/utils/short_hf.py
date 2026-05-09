from typing import List, Optional

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

from transformers import GPT2Tokenizer, GPT2Model

from .metrics import *

import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
class ShortHFModel():

    def __init__(self, model_name: str, layers_path: str, n_prune_layers: Optional[int] = None, mode='hf'):
        """
        HuggingFace Model Wrapper

        Args:
            model_name (str): HuggingFace model name
            layers_path (str): String in dot notation demonstrating how to access layers of the model. Ex: "model.layers"
            (Optional) n_prune_layers (int): Number of layers to prune. Defaults to None.
        """
        if mode == 'hf':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True)
        
        if mode == 'diy':
            self.tokenizer = AutoTokenizer.from_pretrained("/lpai/volumes/ad-vla-vol-ga/lipengxiang/code/ShortGPT/T5-base", model_max_length=256)
            self.tokenizer.pad_token_id = 0
            self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            # model_config = AutoConfig.from_pretrained("/defaultShare/SA-1B/lpx_workspace/LLM_Research/GaLore/configs/llama_350m.json")
            # self.model = LlamaForCausalLM(model_config)
        
        # model_config = AutoConfig.from_pretrained("/defaultShare/SA-1B/lpx_workspace/LLM_Research/GaLore/configs/llama_250m.json")
        # self.model = LlamaForCausalLM(model_config)
        
        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # self.model = GPT2Model.from_pretrained(model_name)
        if 'bert' in model_name.lower():
            print("Detected BERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                local_files_only=True
            ).to("cuda")

            self.layers = self.model.bert.encoder.layer  # ✅ 直接指定 BERT 层


        if mode == 'glm':
            from transformers import AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, cache_dir="/defaultShare/SA-1B/hugging_face_backup", torch_dtype=torch.float16)

        self.model.to("cuda")

        modules = layers_path.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod

        self.n_prune_layers = n_prune_layers
        self.importances = [0 for _ in self.layers]  # layer-wise importance scores

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = [],
        angular: Optional[bool] = False
    ):
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor], angular: bool, n):

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            if angular:
                # use only last token for angular distance as described in section 3.2
                # https://arxiv.org/pdf/2403.17887.pdf
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            print(i)
            
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).mean().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        angular: Optional[bool] = False,
        n=1,
    ):
        """
        Computes layer-wise importances over input texts.

        NOTE: ShortGPT paper performs no generation during importance computation, which suggests a `max_gen_len`= 0.

        Args:
            prompts (List[str]): List of prompts.
            max_seq_len (int): Maximum sequence length for model input, the sliding window size.
            (Optional) stride (int): Number of tokens to skip/shift between each window inference.
            (Optional) max_gen_len (int): Maximum length of the generated text sequence.
            (Optional) temperature (float): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            (Optional) top_p (float): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            (Optional) angular (bool): Whether to ues angular distance. Defaults to False.

        Returns:
            None
        """
        prompt_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=256,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)
        calc_times = 0
        # authors use a sliding window of size 1024 with a shift of 256
        for start in range(0, max_prompt_len, stride):
            # seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            # seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids  # ensure 2d
            inputs = input_ids[0:1, start:start+max_seq_len]
            attn = attn_mask[0:1, start:start+max_seq_len]

            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len, 
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            
            self.compute_bi(outputs.hidden_states, angular=angular, n=n)
            calc_times += 1

        return calc_times