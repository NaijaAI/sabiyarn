
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple
from torch import nn
import torch
import torch.nn.functional as F
import math

repo_name = "BeardedMonster/SabiYarn-125M"


class GPTJXConfig(PretrainedConfig):
    """Configuration class for SabiYarn model."""
    
    model_type = "SabiYarn"
    
    def __init__(
        self,
        block_size: int = 32768 * 2,
        vocab_size: int = 52050,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12,
        n_heads: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        max_batch_size: int = 1,
        use_kv_cache: bool = True,
        bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        kv_cache_dtype: str = "float32",  # "float32" or "float16" for memory savings
        **kwargs
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_kv_cache = use_kv_cache
        self.max_batch_size = max_batch_size
        self.kv_cache_dtype = kv_cache_dtype  # Memory optimization: use float16 for cache
        
        super().__init__(**kwargs)

