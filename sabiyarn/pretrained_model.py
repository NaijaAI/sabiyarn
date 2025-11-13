"""
SabiYarn Model Implementation - Optimized Version
Memory-efficient with performance optimizations for generation.
Matches original implementation exactly but with memory optimizations.
"""

from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from pretrained_config import GPTJXConfig
from typing import List, Optional, Tuple
from torch import nn
import torch
import torch.nn.functional as F
import math

repo_name = "BeardedMonster/SabiYarn-125M"


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    CRITICAL: Must match original implementation exactly for HuggingFace generate() compatibility.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, x, attn_mask=None):
        """
        Forward pass through attention.
        CRITICAL: Must match original signature exactly - no start_pos, no past_key_values.
        HuggingFace's generate() calls this directly and expects this signature.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            if attn_mask is not None:
                # efficient attention using Flash Attention CUDA kernels
                attn_mask = attn_mask.to(torch.bool)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attn_mask, 
                    dropout_p=self.dropout if self.training else 0
                )
            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout if self.training else 0, 
                    is_causal=True
                )
        else:
            # manual implementation of attention
            # CRITICAL: Use the shared bias mask from the model class, not create new ones
            # This matches the original implementation which uses self.bias[:,:,:T,:T]
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Apply causal mask - will be provided by model class via attn_mask
            if attn_mask is not None:
                # Custom mask provided (model class provides optimized mask)
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            else:
                # Fallback: create small mask on-the-fly if no mask provided
                # This should not happen in normal usage, but handles edge cases
                # CRITICAL: Use float dtype to match original's behavior with == 0 check
                causal_mask = torch.tril(torch.ones(T, T, device=att.device))  # float32 by default
                att = att.masked_fill(causal_mask == 0, float('-inf'))  # Match original's == 0 check
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BlockJ(nn.Module):
    """Transformer block with pre-norm architecture and additional layer norm."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # CRITICAL: Original has bug - uses config.n_embd (768) as second argument instead of config.bias
        # Since config.n_embd is truthy (768), this always creates a bias parameter
        # This means j layer norm ALWAYS has bias, even when config.bias=False
        # We must match this exactly to get identical behavior
        self.j = LayerNorm(config.n_embd, config.n_embd)  # Original's bug: always uses bias
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass through transformer block.
        CRITICAL: Must match original signature exactly - no start_pos.
        """
        h = x
        x = self.ln_1(x)
        x = h + self.attn(x, attn_mask=attn_mask) + self.j(x)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTJXForCausalLM(PreTrainedModel):
    """SabiYarn model for causal language modeling."""
    
    config_class = GPTJXConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["BlockJ"]
    _supports_flash_attn_2 = True
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        
        self.config = config
        
        # CRITICAL: Match original implementation's mask handling
        # Original uses: self.register_buffer("bias", torch.tril(...))
        # We optimize by using a smaller base mask and expanding dynamically
        # But we must ensure the behavior is identical
        
        # Check if flash attention is available
        self.flash_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash_available:
            # Memory optimization: Use smaller base mask (1024) instead of full block_size
            # For sequences <= 1024, use base mask directly
            # For longer sequences, create mask on-the-fly (will be cached)
            # CRITICAL: Use float dtype (default) to match original exactly
            # Original uses: torch.tril(torch.ones(...)) which creates float32
            base_mask_size = min(1024, config.block_size)
            base_mask = torch.tril(
                torch.ones(base_mask_size, base_mask_size)  # float32 by default, matches original
            ).view(1, 1, base_mask_size, base_mask_size)
            self.register_buffer("bias", base_mask)
            self._base_mask_size = base_mask_size
            
            # Cache for larger masks (created on-demand)
            self._mask_cache = {}
        else:
            # Flash attention handles causality, no mask buffer needed
            # But we still register a dummy buffer for compatibility
            self.register_buffer("bias", torch.tril(torch.ones(1, 1)))
            self._base_mask_size = 0
            self._mask_cache = {}
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockJ(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings
    
    def _get_mask_for_sequence(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get causal mask for sequence of given length.
        Optimized version: uses base mask if possible, creates larger masks on-demand.
        CRITICAL: Must produce same results as original's self.bias[:,:,:T,:T]
        """
        if self.flash_available:
            # Flash attention handles causality, return None
            return None
        
        # If sequence fits in base mask, use it directly
        if seq_len <= self._base_mask_size:
            mask = self.bias.to(device)
            # Slice to actual sequence length (matches original's self.bias[:,:,:T,:T])
            return mask[:, :, :seq_len, :seq_len]
        
        # For longer sequences, create mask on-the-fly and cache it
        # Check cache first
        cache_key = (seq_len, device)
        if cache_key not in self._mask_cache:
            # Create mask of exact size needed (not power-of-2, to match original exactly)
            # CRITICAL: Use float dtype (default) to match original exactly
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # float32 by default
            mask = mask.view(1, 1, seq_len, seq_len)
            self._mask_cache[cache_key] = mask
        
        return self._mask_cache[cache_key]
    
    def forward(self, idx, targets=None, attn_mask=None, output_hidden_states: Optional[bool] = None, **kwargs):
        """
        Forward pass through the model.
        CRITICAL: Must match original signature exactly for HuggingFace generate() compatibility.
        """
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # CRITICAL: Provide optimized mask to attention layers
        # If no custom mask provided, create causal mask matching original behavior
        if attn_mask is None:
            # Get optimized causal mask (matches original's self.bias[:,:,:T,:T])
            # CRITICAL: Don't expand - PyTorch will broadcast (1, 1, T, T) to (B, nh, T, T) automatically
            # This matches the original's behavior exactly
            optimized_mask = self._get_mask_for_sequence(t, device)
            if optimized_mask is not None:
                attn_mask = optimized_mask  # Shape: (1, 1, T, T), will broadcast to (B, nh, T, T)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask)
        
        x = self.transformer.ln_f(x)
        
        # CRITICAL: Match original's logits computation exactly
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # CRITICAL: This matches the original exactly - using list [-1] to preserve time dim
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=x if output_hidden_states else None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """
        Prepare inputs for generation. Called by HuggingFace's generate().
        CRITICAL: Must match original exactly.
        """
        # Default model inputs
        model_inputs = {"idx": input_ids}
        
        # Add attention mask if provided
        if attention_mask is not None:
            model_inputs["attn_mask"] = attention_mask
        
        return model_inputs
    
    def crop_block_size(self, block_size):
        """model surgery to decrease the block size if necessary"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        
        # Update bias mask if needed (only if not using flash attention)
        if not self.flash_available:
            # Update base mask size if block_size is smaller
            if block_size < self._base_mask_size:
                self._base_mask_size = min(block_size, self._base_mask_size)
                base_mask = torch.tril(
                    torch.ones(self._base_mask_size, self._base_mask_size)  # float32 by default
                ).view(1, 1, self._base_mask_size, self._base_mask_size)
                # CRITICAL: Use register_buffer to match original, not nn.Parameter
                self.register_buffer("bias", base_mask)
            
            # Clear mask cache (will be recreated with new size when needed)
            self._mask_cache.clear()


# Register model with AutoConfig and AutoModel
AutoConfig.register("SabiYarn", GPTJXConfig)
AutoModelForCausalLM.register(GPTJXConfig, GPTJXForCausalLM)
