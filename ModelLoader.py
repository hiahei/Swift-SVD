"""
Generic model loader - auto-detects and loads multiple LLM architectures.
Reusable across multiple scripts.
"""
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Phi3ForCausalLM, 
    OPTForCausalLM,
    # GemmaForCausalLM,
    MistralForCausalLM,
    Qwen3ForCausalLM,
    LlamaForCausalLM,
)
# from model.qwen3.modeling_qwen3 import Qwen3ForCausalLM
# from model.llama2.modeling_llama import LlamaForCausalLM



class ModelLoader:
    """
    Generic model loader class.
    Automatically detects the model type and loads the matching model and tokenizer.
    """
    
    SUPPORTED_MODELS = {
        'qwen': ['qwen', 'qwen2', 'qwen3'],
        'llama': ['llama', 'llama2', 'llama3'],
        'mistral': ['mistral'],
        'gemma': ['gemma', 'gemma2'],
        'phi': ['phi', 'phi2', 'phi3'],
        'opt': ['opt'],
        'vicuna': ['vicuna'],
    }
    
    # Whether to use custom model classes (default: native transformers)
    USE_CUSTOM_MODELS = False  # False uses native classes; True uses custom classes
    
    @staticmethod
    def detect_model_type(model_name_or_path):
        """
        Auto-detect model type.
        
        Args:
            model_name_or_path: Model name or path
        
        Returns:
            model_type: 'qwen3', 'llama', 'mistral', 'gemma', etc.
        """
        try:
            # Try reading the model type from config
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_type = config.model_type.lower()
            print(f"✓ Detected model type from config: {model_type}")
            return model_type
        except Exception as e:
            # If config lookup fails, infer from path/name
            model_name_lower = model_name_or_path.lower()
            
            for family, types in ModelLoader.SUPPORTED_MODELS.items():
                if any(t in model_name_lower for t in types):
                    # Return specific type if identifiable; otherwise return family name
                    for t in types:
                        if t in model_name_lower:
                            print(f"✓ Inferred model type from path: {t}")
                            return t
                    print(f"✓ Inferred model family from path: {family}")
                    return family
            
            raise ValueError(
                f"Cannot detect model type from: {model_name_or_path}\n"
                f"Supported models: {ModelLoader.SUPPORTED_MODELS}\n"
                f"Error: {e}"
            )
    
    @staticmethod
    def load_model(model_name_or_path, device_map="cpu", torch_dtype="auto", **kwargs):
        """
        Load a model based on auto-detected type.
        
        Args:
            model_name_or_path: Model name or path
            device_map: Device map (default: "cpu")
            torch_dtype: Data type (default: "auto")
            **kwargs: Extra args passed to from_pretrained
        
        Returns:
            model: Loaded model
        """
        model_type = ModelLoader.detect_model_type(model_name_or_path)
        return ModelLoader.load_model_by_type(
            model_name_or_path,
            model_type,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs
        )
    
    @staticmethod
    def load_model_by_type(model_name_or_path, model_type, device_map="cpu", torch_dtype="auto", **kwargs):
        """
        Load a model by explicit model type.
        
        Args:
            model_name_or_path: Model name or path
            model_type: Model type
            device_map: Device map
            torch_dtype: Data type
            **kwargs: Other parameters
        
        Returns:
            model: Loaded model
        """
        print(f"\nLoading {model_type} model: {model_name_or_path}")
        
        # Qwen family
        if model_type in ModelLoader.SUPPORTED_MODELS['qwen']:
            model = Qwen3ForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                **kwargs
            )
        
        # Llama family
        elif model_type in ModelLoader.SUPPORTED_MODELS['llama']:
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )
        
        # Vicuna family (Llama architecture)
        elif model_type in ModelLoader.SUPPORTED_MODELS['vicuna']:
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )
        
        # Mistral family (Llama architecture)
        elif model_type in ModelLoader.SUPPORTED_MODELS['mistral']:
            model = MistralForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )
        
        # # Gemma family
        # elif model_type in ModelLoader.SUPPORTED_MODELS['gemma']:
        #     model = GemmaForCausalLM.from_pretrained(
        #         model_name_or_path,
        #         torch_dtype=torch_dtype,
        #         device_map=device_map,
        #         **kwargs
        #     )
        
        # Phi family
        elif model_type in ModelLoader.SUPPORTED_MODELS['phi']:
            model = Phi3ForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )        
        # OPT family
        elif model_type in ModelLoader.SUPPORTED_MODELS['opt']:
            model = OPTForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}\n"
                f"Supported models: {ModelLoader.SUPPORTED_MODELS}")
        
        print(f"✓ Model loaded successfully")
        return model
    
    @staticmethod
    def load_tokenizer(model_name_or_path, **kwargs):
        """
        Load tokenizer.
        
        Args:
            model_name_or_path: Model name or path
            **kwargs: Extra args passed to AutoTokenizer
        
        Returns:
            tokenizer: Loaded tokenizer
        """
        print(f"Loading tokenizer from: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
            **kwargs
        )
        print(f"✓ Tokenizer loaded successfully")
        return tokenizer
    
    @staticmethod
    def load_model_and_tokenizer(model_name_or_path, device_map="cpu", torch_dtype="auto", **kwargs):
        """
        Load both model and tokenizer.
        
        Args:
            model_name_or_path: Model name or path
            device_map: Device map
            torch_dtype: Data type
            **kwargs: Other parameters
        
        Returns:
            tuple: (model, tokenizer)
        """
        model = ModelLoader.load_model(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs
        )
        tokenizer = ModelLoader.load_tokenizer(model_name_or_path)
        return model, tokenizer

# Helper functions for direct use
def load_model(model_name_or_path, **kwargs):
    """Load model."""
    return ModelLoader.load_model(model_name_or_path, **kwargs)

def load_tokenizer(model_name_or_path, **kwargs):
    """Load tokenizer."""
    return ModelLoader.load_tokenizer(model_name_or_path, **kwargs)

def load_model_and_tokenizer(model_name_or_path, **kwargs):
    """Load model and tokenizer."""
    return ModelLoader.load_model_and_tokenizer(model_name_or_path, **kwargs)