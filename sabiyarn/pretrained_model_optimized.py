"""
SabiYarn Model Implementation - Optimized Version
Memory-efficient with performance optimizations for generation.
Matches original implementation exactly for consistent outputs.
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
    
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional KV caching.
    Memory-optimized: KV cache uses lazy allocation and optional half precision.
    """
    
    def __init__(self, config: GPTJXConfig):
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
        self.head_dim = config.n_embd // config.n_heads
        self.block_size = config.block_size
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        self.use_kv_cache = config.use_kv_cache
        # Parse kv_cache_dtype string to torch dtype
        if isinstance(config.kv_cache_dtype, str):
            self.kv_cache_dtype = getattr(torch, config.kv_cache_dtype, torch.float32)
        else:
            self.kv_cache_dtype = config.kv_cache_dtype
        
        # Memory optimization: Lazy KV cache allocation
        # Only allocate when actually needed during inference, not at initialization
        # This prevents memory bloat during model loading
        self._cache_k = None
        self._cache_v = None
        self._cache_initialized = False
        self.max_batch_size = config.max_batch_size
        
        # Cache for dynamically created masks (power-of-2 sizes)
        # Key: (mask_size, device), Value: mask tensor
        self._mask_cache: dict = {}
    
    @staticmethod
    def _get_mask_size(seq_len: int, max_size: int = 32768) -> int:
        """
        Round up sequence length to nearest power-of-2 mask size.
        This minimizes memory usage while ensuring mask is large enough.
        
        Args:
            seq_len: Actual sequence length
            max_size: Maximum mask size (block_size)
        
        Returns:
            Mask size (one of: 4096, 8192, 16384, 32768)
        """
        # Power-of-2 sizes for efficient memory usage
        sizes = [4096, 8192, 16384, 32768]
        # Filter to sizes <= max_size
        sizes = [s for s in sizes if s <= max_size]
        
        for size in sizes:
            if seq_len <= size:
                return size
        
        # Fallback to max_size if sequence is larger
        return max_size
    
    def _get_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Get or create a causal mask of appropriate size.
        Uses power-of-2 sizing to minimize memory usage.
        Masks are cached to avoid recreation.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            dtype: Data type for mask (default: bool)
        
        Returns:
            Causal mask of shape (1, 1, mask_size, mask_size)
        """
        # Determine appropriate mask size (power-of-2 rounded up)
        mask_size = self._get_mask_size(seq_len, self.block_size)
        
        # Check cache
        cache_key = (mask_size, device, dtype)
        if cache_key not in self._mask_cache:
            # Create mask on-the-fly
            mask = torch.tril(torch.ones(mask_size, mask_size, device=device, dtype=dtype))
            mask = mask.view(1, 1, mask_size, mask_size)
            self._mask_cache[cache_key] = mask
        
        return self._mask_cache[cache_key]
    
    def _init_kv_cache(self, device: torch.device, dtype: torch.dtype):
        """
        Lazy initialization of KV cache.
        Only allocates memory when actually needed during inference.
        """
        if not self._cache_initialized and self.use_kv_cache:
            # Use specified dtype (float16 for memory savings) for cache
            # If kv_cache_dtype is float32, use the input dtype instead
            if self.kv_cache_dtype == torch.float32:
                cache_dtype = dtype
            else:
                cache_dtype = self.kv_cache_dtype
            
            self._cache_k = torch.zeros(
                self.max_batch_size,
                self.n_heads,
                self.block_size,
                self.head_dim,
                device=device,
                dtype=cache_dtype
            )
            self._cache_v = torch.zeros(
                self.max_batch_size,
                self.n_heads,
                self.block_size,
                self.head_dim,
                device=device,
                dtype=cache_dtype
            )
            self._cache_initialized = True
    
    def clear_cache(self):
        """Clear the KV cache. Useful for resetting state between sequences."""
        if self._cache_initialized:
            self._cache_k.zero_()
            self._cache_v.zero_()
    
    def free_cache(self):
        """Free KV cache memory. Useful for memory cleanup."""
        if self._cache_initialized:
            del self._cache_k
            del self._cache_v
            self._cache_k = None
            self._cache_v = None
            self._cache_initialized = False
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int = 0, 
        *,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            x: Input tensor of shape (B, T, C)
            start_pos: Starting position for KV cache (for incremental decoding)
            attn_mask: Optional custom attention mask of shape (B, 1, T, T) or (1, 1, T, T).
                      If None, a causal mask will be created dynamically. 
                      Useful for multitask learning with custom masking patterns.
        
        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Handle KV cache for incremental decoding
        if self.use_kv_cache and not self.training:
            # Lazy initialization: allocate cache on first use
            if not self._cache_initialized:
                self._init_kv_cache(x.device, x.dtype)
            
            # Convert k, v to cache dtype if needed (for float16 cache)
            k_for_cache = k.to(self._cache_k.dtype)
            v_for_cache = v.to(self._cache_v.dtype)
            
            # Update cache with new keys and values
            self._cache_k[:B, :, start_pos:start_pos + T] = k_for_cache
            self._cache_v[:B, :, start_pos:start_pos + T] = v_for_cache
            
            # Use cached keys and values up to current position
            # Convert back to input dtype for computation
            k = self._cache_k[:B, :, :start_pos + T].to(x.dtype)
            v = self._cache_v[:B, :, :start_pos + T].to(x.dtype)
            
            # Adjust query to only query from current position
            q = q[:, :, -T:, :]  # Only query the new tokens
        
        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            if attn_mask is not None:
                # Custom mask provided (e.g., for multitask learning)
                attn_mask = attn_mask.to(torch.bool)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attn_mask, 
                    dropout_p=self.dropout if self.training else 0
                )
            else:
                # No custom mask: use built-in causal attention
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout if self.training else 0, 
                    is_causal=True
                )
        else:
            # Manual implementation of attention (fallback)
            # Use the full sequence length for attention computation
            seq_len = k.size(-2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            # Apply causal mask (only if no custom mask provided)
            if attn_mask is None:
                # Create dynamic mask with power-of-2 sizing
                if start_pos == 0:
                    # Full sequence: use cached mask of appropriate size
                    causal_mask = self._get_causal_mask(seq_len, att.device, dtype=torch.bool)
                    # Slice to actual sequence length
                    att = att.masked_fill(
                        causal_mask[:, :, :T, :seq_len] == 0, 
                        float('-inf')
                    )
                else:
                    # Incremental decoding: create small mask on-the-fly
                    causal_mask = torch.tril(torch.ones(T, seq_len, device=att.device, dtype=torch.bool))
                    att = att.masked_fill(~causal_mask, float('-inf'))
            else:
                # Custom attention mask provided (for multitask learning, etc.)
                # Apply custom mask directly
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, config: GPTJXConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BlockJ(nn.Module):
    """Transformer block with pre-norm architecture and additional layer norm."""
    
    def __init__(self, config: GPTJXConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.j = LayerNorm(config.n_embd, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int = 0, 
        *,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor
            start_pos: Starting position for KV cache
            attn_mask: Optional custom attention mask (keyword-only argument)
        """
        h = x
        x = self.ln_1(x)
        x = h + self.attn(x, start_pos, attn_mask=attn_mask) + self.j(x)
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
    
    def __init__(self, config: GPTJXConfig):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        
        self.config = config
        
        # Memory optimization: No mask buffers created at initialization
        # Masks are created dynamically on-demand with power-of-2 sizing
        # This saves ~2GB+ that would be allocated as a buffer
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                BlockJ(config) 
                for _ in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between input embeddings and output projection
        # This reduces parameters and can improve performance
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize all weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 initialization scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding: bool = False) -> int:
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
        """Fixed: return transformer.wte instead of self.wte"""
        return self.transformer.wte
    
    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """Fixed: set transformer.wte instead of self.wte"""
        self.transformer.wte = new_embeddings
    
    def clear_kv_cache(self):
        """Clear KV cache in all attention layers. Useful for resetting state."""
        for block in self.transformer.h:
            block.attn.clear_cache()
    
    def free_kv_cache(self):
        """
        Free KV cache memory in all attention layers.
        Useful for memory cleanup when switching between training and inference.
        """
        for block in self.transformer.h:
            block.attn.free_cache()
    
    def forward(
        self,
        idx: torch.Tensor,
        start_pos: int = 0,
        targets: Optional[torch.Tensor] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_logits_only: bool = False,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.
        Matches original implementation exactly for consistent outputs.
        
        Args:
            idx: Input token indices of shape (B, T)
            start_pos: Starting position for KV cache (for incremental decoding)
            targets: Target token indices for loss computation of shape (B, T)
            attn_mask: Optional custom attention mask (keyword-only argument).
                      If None, causal masks will be created dynamically.
                      Useful for multitask learning with custom masking patterns.
            output_hidden_states: Whether to return hidden states
            return_logits_only: If True, return only logits tensor instead of CausalLMOutputWithPast
        
        Returns:
            CausalLMOutputWithPast or logits tensor
        """
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        
        # CRITICAL: Position embeddings always start from 0 for the current input
        # This matches the original implementation exactly
        # The KV cache handles remembering past positions, not position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        
        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Note: attn_mask is passed through to blocks
        # If None, each attention layer will create its own causal mask dynamically
        # If provided, it will be used as-is (for custom masking patterns)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x, start_pos, attn_mask=attn_mask)
        
        x = self.transformer.ln_f(x)
        
        # Compute logits and loss
        if targets is not None:
            # Training: calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100
            )
        else:
            # Inference: compute logits for all positions (matches original)
            # During generation, we'll take the last position's logits
            logits = self.lm_head(x)
            loss = None
        
        if return_logits_only:
            return logits
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=x if output_hidden_states else None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # Default model inputs - matches original implementation
        model_inputs = {"idx": input_ids}
        # Add attention mask if provided
        if attention_mask is not None:
            model_inputs["attn_mask"] = attention_mask
        return model_inputs
    
    def crop_block_size(self, block_size: int):
        """Crop the model's block size to a smaller value."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        
        # Update block_size in all attention layers
        for block in self.transformer.h:
            block.attn.block_size = block_size
            # Clear mask cache (masks will be recreated with new size when needed)
            block.attn._mask_cache.clear()
            # Free and reinitialize cache if it was already allocated
            if block.attn._cache_initialized:
                block.attn.free_cache()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.8,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        eos_token_id: Optional[int] = 1,
        *,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Custom generation method that matches original implementation exactly.
        Uses internal KV cache for efficient incremental decoding.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Sampling temperature (higher = more random)
            num_beams: Number of beams for beam search (1 = no beam search)
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
            top_k: Top-k sampling: keep only top k logits
            top_p: Nucleus sampling: keep tokens with cumulative probability <= top_p
            length_penalty: Length penalty for beam search (>1.0 encourages longer sequences)
            early_stopping: Whether to stop beam search when all beams find EOS
            eos_token_id: End-of-sequence token ID (None to disable)
            attn_mask: Optional attention mask for input sequence (keyword-only argument).
                      If None, causal masks will be created dynamically.
        
        Returns:
            Generated token indices of shape (batch_size, seq_len + generated_tokens)
        """
        self.eval()  # Ensure model is in eval mode
        
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        
        # Initialize generation state
        generated_sequences = input_ids.clone()
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        
        # For beam search, we need to track multiple candidate sequences
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids=input_ids,
                attn_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                eos_token_id=eos_token_id,
            )
        
        # Standard autoregressive generation (greedy or sampling)
        current_pos = seq_len
        
        # Clear KV cache at the start of generation
        self.clear_kv_cache()
        
        for step in range(max_new_tokens):
            # Crop sequence if it exceeds block_size
            if current_pos >= self.config.block_size:
                # Keep only the last block_size tokens
                generated_sequences = generated_sequences[:, -self.config.block_size:]
                current_pos = self.config.block_size
                # Clear cache and recompute from scratch
                self.clear_kv_cache()
            
            # Get the current input (last token or sequence)
            if step == 0:
                # First step: use full input sequence
                current_input = generated_sequences
                start_pos = 0
            else:
                # Subsequent steps: only use the last token (incremental decoding)
                current_input = generated_sequences[:, -1:]
                start_pos = current_pos - 1
            
            # Forward pass - matches original implementation exactly
            logits = self(
                current_input,
                start_pos=start_pos,
                attn_mask=attn_mask,
                return_logits_only=True
            )
            
            # Get logits for the last position
            # Shape is (B, T, vocab_size), take last position
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply repetition penalty (works for batched generation)
            if repetition_penalty != 1.0:
                # Get recently generated tokens from each sequence in the batch
                for batch_idx in range(bsz):
                    if not finished[batch_idx]:  # Only apply to unfinished sequences
                        # Get last 50 tokens from this sequence
                        recent_tokens = generated_sequences[batch_idx, -50:].tolist()
                        recent_token_set = set(recent_tokens)
                        
                        # Apply penalty to each recently seen token
                        for token_id in recent_token_set:
                            if next_token_logits[batch_idx, token_id] > 0:
                                next_token_logits[batch_idx, token_id] /= repetition_penalty
                            else:
                                next_token_logits[batch_idx, token_id] *= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Handle finished sequences (force EOS token)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.tensor(eos_token_id, device=device).expand_as(next_token),
                    next_token
                )
                finished |= (next_token.squeeze(1) == eos_token_id)
            
            # Append to sequence
            generated_sequences = torch.cat([generated_sequences, next_token], dim=1)
            current_pos += 1
            
            # Early stopping if all sequences are finished
            if eos_token_id is not None and finished.all():
                break
        
        return generated_sequences
    
    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        num_beams: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        length_penalty: float,
        early_stopping: bool,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """
        Beam search generation implementation.
        
        This is a simplified beam search. For production use, consider using
        HuggingFace's built-in beam search which is more optimized.
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        
        # Initialize beams: (batch_size * num_beams, seq_len)
        # For simplicity, we'll do beam search per batch item
        if bsz > 1:
            # For batched beam search, we'd need to handle this more carefully
            # This is a simplified version that processes one batch at a time
            results = []
            for i in range(bsz):
                single_input = input_ids[i:i+1]
                single_result = self._generate_beam_search(
                    single_input, attn_mask, max_new_tokens, num_beams,
                    temperature, top_k, top_p, repetition_penalty,
                    length_penalty, early_stopping, eos_token_id
                )
                results.append(single_result)
            return torch.cat(results, dim=0)
        
        # Single batch item beam search
        # Initialize beam hypotheses
        beams = [(input_ids.clone(), 0.0)]  # (sequence, score)
        
        for step in range(max_new_tokens):
            candidates = []
            
            for seq, score in beams:
                # Check if sequence is finished
                if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                    candidates.append((seq, score))
                    continue
                
                # Forward pass - always pass full sequence, no KV cache for beam search
                # This matches the original implementation
                logits = self(seq, start_pos=0, return_logits_only=True)
                # Logits shape is (B, T, vocab_size), take last position
                next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    generated_tokens = seq[0].tolist()
                    recent_tokens = set(generated_tokens[-50:])
                    for token_id in recent_tokens:
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= repetition_penalty
                
                # Get top-k candidates
                top_k_candidates = min(num_beams * 2, next_token_logits.size(-1))
                if top_k is not None:
                    top_k_candidates = min(top_k_candidates, top_k)
                
                top_logits, top_indices = torch.topk(next_token_logits, top_k_candidates)
                top_probs = F.softmax(top_logits, dim=-1)
                
                # Create candidate sequences
                for i in range(top_k_candidates):
                    token_id = top_indices[0, i].item()
                    token_prob = top_probs[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    # Apply length penalty
                    new_score = score + math.log(token_prob) / (length_penalty ** (step + 1))
                    candidates.append((new_seq, new_score))
            
            # Select top num_beams candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:num_beams]
            
            # Early stopping check
            if early_stopping and eos_token_id is not None:
                if all(beam[0][0, -1].item() == eos_token_id for beam in beams):
                    break
        
        # Return the best sequence
        return beams[0][0]


# Register model with HuggingFace
AutoConfig.register("SabiYarn", GPTJXConfig)
AutoModelForCausalLM.register(GPTJXConfig, GPTJXForCausalLM)
