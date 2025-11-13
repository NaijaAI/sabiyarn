from model import (SabiYarn, ModelArgs, AttentionType, DiffAttnArgs, MLAConfig, 
                   TransformerBlock, RMSNorm, MultiTokenPredictor, 
                   _detect_distributed_config)
import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class Args(ModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    # MoE parameters - only used with MLA
    moe: Optional[bool] = False
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    route_scale: float = 1.0
    n_shared_experts: int = 1
    moe_inter_dim: int = 2048
    logic_network: Optional[bool] = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_j: bool = True
    tie_weights: bool = True
    attention_type: str = AttentionType.SELF_ATTENTION
    diff_attn_args: Optional[DiffAttnArgs] = None
    mla_config: Optional[MLAConfig] = None
    
    # Distributed training configuration
    auto_detect_distributed: bool = True
    distributed: bool = False
    data_parallel: bool = False  # For MHA/DiffAttention
    tensor_parallel: bool = False  # For MLA
    world_size: int = 1
    rank: int = 0
    
    # Multi-token prediction parameters (only for MLA)
    multi_token_prediction: bool = False
    num_prediction_tokens: int = 4  # Number of future tokens to predict
    mtp_loss_weight: float = 1.0  # Weighting factor for MTP loss
    mtp_share_embeddings: bool = True  # Share embeddings with main model
    
    # Weight initialization
    init_std: float = 0.006  # DeepSeek paper default
    
    # Layer sharing (MobileLLM-style immediate block-wise repeat)
    layer_sharing: bool = False  # Enable layer sharing
    n_unique_layers: Optional[int] = None  # Number of unique layers (if None, same as n_layers)
    
    def __post_init__(self):
        """Auto-configure distributed settings and validate layer sharing configuration."""
        # Validate layer sharing configuration first
        if self.layer_sharing:
            if self.n_unique_layers is None:
                raise ValueError("n_unique_layers must be specified when layer_sharing is True")
            if self.n_unique_layers <= 0:
                raise ValueError(f"n_unique_layers must be positive, got {self.n_unique_layers}")
            if self.n_unique_layers > self.n_layers:
                raise ValueError(f"n_unique_layers ({self.n_unique_layers}) cannot be greater than n_layers ({self.n_layers})")
            if self.n_layers % self.n_unique_layers != 0:
                raise ValueError(f"n_layers ({self.n_layers}) must be divisible by n_unique_layers ({self.n_unique_layers}) for immediate block-wise repeat")
        
        if self.auto_detect_distributed:
            detected_dist, detected_dp, detected_tp, detected_ws, detected_rank = _detect_distributed_config()
            
            # Only override if not explicitly set
            if not self.distributed:
                self.distributed = detected_dist
                self.world_size = detected_ws
                self.rank = detected_rank
                
                # Auto-choose parallelism strategy based on attention type
                if self.distributed:
                    if self.attention_type == AttentionType.MLA:
                        self.tensor_parallel = True
                        self.data_parallel = False
                        print(f"🔧 Auto-configured: MLA with tensor parallelism ({self.world_size} GPUs)")
                    else:
                        self.tensor_parallel = False
                        self.data_parallel = True
                        print(f"🔧 Auto-configured: {self.attention_type} with data parallelism ({self.world_size} GPUs)")
                else:
                    print("🔧 Auto-configured: Single GPU/CPU mode")




class ThinkingModule(SabiYarn):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.
            diff_attn_Args( DiffAttnArgs): configuration parameters for Differential Attention.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        # super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        # Create layers with optional layer sharing (MobileLLM-style immediate block-wise repeat)
        if params.layer_sharing:
            # Create only unique layers
            self.unique_layers = torch.nn.ModuleList()
            # Safe to assert since validation in __post_init__ ensures n_unique_layers is not None when layer_sharing is True
            assert params.n_unique_layers is not None, "n_unique_layers should not be None when layer_sharing is True"
            self.n_unique_layers = params.n_unique_layers
            self.repeat_factor = params.n_layers // self.n_unique_layers
            
            for layer_id in range(self.n_unique_layers):
                self.unique_layers.append(TransformerBlock(layer_id, params, use_j_linear=params.use_j))
            
            # Create execution mapping for immediate block-wise repeat
            self.layer_execution_order = []
            for _ in range(self.repeat_factor):
                for unique_id in range(self.n_unique_layers):
                    self.layer_execution_order.append(unique_id)
        else:
            # Traditional: each layer is unique
            self.layers = torch.nn.ModuleList()
            for layer_id in range(params.n_layers):
                self.layers.append(TransformerBlock(layer_id, params, use_j_linear=params.use_j))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Initialize weights following DeepSeek paper recommendations
        self.apply(self._init_weights)

        # Precompute frequencies for non-MLA attention types
        if params.attention_type == AttentionType.SELF_ATTENTION:
            from .MHA import precompute_freqs_cis
            self.freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                self.params.max_seq_len * 2,
            )
        elif params.attention_type == AttentionType.DIFFERENTIAL_ATTENTION:
            if params.diff_attn_args is None:
                raise ValueError("diff_attn_args must be provided for DIFFERENTIAL_ATTENTION")
            from .differential_attention import precompute_freqs_cis
            self.freqs_cis = precompute_freqs_cis(
                # Differential attention uses half the head_dim for key and query vectors
                self.params.diff_attn_args.embed_dim  # type: ignore
                // self.params.diff_attn_args.n_heads  # type: ignore
                // 2,
                self.params.max_seq_len * 2,
            )
        elif params.attention_type == AttentionType.MLA:
            # MLA precomputes its own frequencies internally
            from .MLA import precompute_freqs_cis
            if params.mla_config is None:
                raise ValueError("mla_config must be provided for MLA")
            self.freqs_cis = precompute_freqs_cis(params.mla_config)
        else:
            self.freqs_cis = None

    def get_model_size(self):
        raise AttributeError("This method is not available.")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_multi_token=False):
        raise AttributeError("This method is not available.")

    def forward(self, tokens: torch.Tensor, start_pos: int, mask=None, return_multi_token=None):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.
            mask (Optional[torch.Tensor]): Attention mask.
            return_multi_token (Optional[bool]): Whether to return multi-token predictions.
                                               If None, uses self.use_multi_token

        Returns:
            Tuple or Triple: 
                - If MTP disabled: (hidden_states, logits)
                - If MTP enabled: (hidden_states, logits, multi_token_logits)
        """
        _bsz, seqlen = tokens.shape
        
        # Get input embeddings (needed for MTP)
        input_embeddings = self.tok_embeddings(tokens)
        h = input_embeddings
        
        # Prepare frequencies and mask based on attention type
        if self.params.attention_type == AttentionType.MLA:
            # MLA uses its own frequency computation
            freqs_cis = self.freqs_cis
        else:
            # For non-MLA attention, slice pre-computed frequencies
            if self.freqs_cis is not None:
                self.freqs_cis = self.freqs_cis.to(h.device)
                freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
            else:
                freqs_cis = None

            # Create causal mask if none provided
            if mask is None and seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
                ).type_as(h)

        # Forward through layers
        if self.params.layer_sharing:
            # Layer sharing: use immediate block-wise repeat
            for unique_layer_id in self.layer_execution_order:
                layer = self.unique_layers[unique_layer_id]
                h = layer(h, start_pos, freqs_cis, mask)
        else:
            # Traditional: each layer is unique
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
        
        # Main transformer output (needed for MTP)
        transformer_output = h
        hidden_states = self.norm(h)
        
        # Standard next-token prediction
        logits = self.lm_head(hidden_states).float()
        
        # Multi-token prediction
        multi_token_logits = None
        if return_multi_token or (return_multi_token is None and self.use_multi_token):
            if self.multi_token_predictor is not None:
                multi_token_logits = self.multi_token_predictor(
                    input_embeddings=input_embeddings,
                    transformer_output=transformer_output,
                    start_pos=start_pos,
                    freqs_cis=freqs_cis,
                    lm_head=self.lm_head if self.params.mtp_share_embeddings else None,
                    mask=mask
                )

        # Handle distributed training (existing logic)
        if (self.params.attention_type == AttentionType.MLA and 
            self.params.tensor_parallel and 
            self.params.world_size > 1):
            import torch.distributed as dist
            all_logits = [torch.empty_like(logits) for _ in range(self.params.world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
            
            # Also handle multi-token logits
            if multi_token_logits is not None:
                # Gather multi-token logits across devices
                all_multi_logits = [torch.empty_like(multi_token_logits) for _ in range(self.params.world_size)]
                dist.all_gather(all_multi_logits, multi_token_logits)
                multi_token_logits = torch.cat(all_multi_logits, dim=-1)
        
        if multi_token_logits is not None:
            return hidden_states, logits, multi_token_logits
        else:
            return hidden_states, logits


