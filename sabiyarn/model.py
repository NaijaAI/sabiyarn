import math
import os
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from .memory_reasoning import LogicNetwork
from .differential_attention import DiffAttention, DiffAttnArgs
from .MLA import MLA, MLAConfig, ColumnParallelLinear, RowParallelLinear, linear
from .MHA import SelfAttention, SelfAttnArgs, precompute_freqs_cis


class AttentionType(str, Enum):
    SELF_ATTENTION = "self_attention"
    DIFFERENTIAL_ATTENTION = "differential_attention"
    MLA = "MLA"


def _validate_attention_config(args: 'ModelArgs') -> None:
    """
    Validate attention configuration for consistency.
    
    Args:
        args (ModelArgs): Model configuration parameters.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    if args.attention_type == AttentionType.DIFFERENTIAL_ATTENTION:
        if args.diff_attn_args is None:
            raise ValueError("diff_attn_args must be provided for DIFFERENTIAL_ATTENTION")
        # Validate compatibility
        if args.diff_attn_args.embed_dim != args.dim:
            raise ValueError(f"diff_attn_args.embed_dim ({args.diff_attn_args.embed_dim}) must match args.dim ({args.dim})")
            
    elif args.attention_type == AttentionType.MLA:
        if args.mla_config is None:
            raise ValueError("mla_config must be provided for MLA")
        # Validate compatibility  
        if args.mla_config.hidden_size != args.dim:
            raise ValueError(f"mla_config.hidden_size ({args.mla_config.hidden_size}) must match args.dim ({args.dim})")
        if args.mla_config.num_heads != args.n_heads:
            raise ValueError(f"mla_config.num_heads ({args.mla_config.num_heads}) must match args.n_heads ({args.n_heads})")
    
    # MoE validation
    if args.moe:
        if args.attention_type != AttentionType.MLA:
            raise ValueError("MoE can only be used with MLA attention type")
        
        # Validate MoE parameters
        if args.n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts must be positive, got {args.n_routed_experts}")
        if args.n_activated_experts <= 0:
            raise ValueError(f"n_activated_experts must be positive, got {args.n_activated_experts}")
        if args.n_activated_experts > args.n_routed_experts:
            raise ValueError(f"n_activated_experts ({args.n_activated_experts}) cannot be greater than n_routed_experts ({args.n_routed_experts})")
        if args.moe_inter_dim <= 0:
            raise ValueError(f"moe_inter_dim must be positive, got {args.moe_inter_dim}")
        if args.n_shared_experts < 0:
            raise ValueError(f"n_shared_experts must be non-negative, got {args.n_shared_experts}")
        
        # Distributed training validation
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                world_size = dist.get_world_size()
                if args.n_routed_experts % world_size != 0:
                    raise ValueError(f"n_routed_experts ({args.n_routed_experts}) must be divisible by world_size ({world_size}) for distributed training")
        except ImportError:
            pass  # torch.distributed not available
    
    # Multi-token prediction validation
    if args.multi_token_prediction:
        if args.attention_type != AttentionType.MLA:
            raise ValueError("Multi-token prediction can only be used with MLA attention type")
        if args.num_prediction_tokens <= 0:
            raise ValueError(f"num_prediction_tokens must be positive, got {args.num_prediction_tokens}")
        if args.mtp_loss_weight < 0:
            raise ValueError(f"mtp_loss_weight must be non-negative, got {args.mtp_loss_weight}")


def _detect_distributed_config():
    """
    Auto-detect distributed training configuration from environment.
    
    Returns:
        Tuple[bool, bool, bool, int, int]: (distributed, data_parallel, tensor_parallel, world_size, rank)
    """
    try:
        import torch.distributed as dist
        
        # Check if distributed is initialized
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            distributed = world_size > 1
            return distributed, False, False, world_size, rank
        
        # Check environment variables for distributed setup
        if 'WORLD_SIZE' in os.environ or 'LOCAL_WORLD_SIZE' in os.environ:
            world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('LOCAL_WORLD_SIZE', '1')))
            rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
            distributed = world_size > 1
            return distributed, False, False, world_size, rank
            
    except ImportError:
        pass
    
    # Check for CUDA availability as fallback
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            # Multiple GPUs available but no distributed setup
            print(f"💡 Found {gpu_count} GPUs but no distributed training detected.")
            print("   Consider using torchrun for multi-GPU training.")
        return False, False, False, 1, 0
    
    # Single GPU or CPU
    return False, False, False, 1, 0


def _create_attention(layer_id: int, args: 'ModelArgs') -> nn.Module:
    """
    Factory function to create attention modules based on configuration.
    
    Args:
        layer_id (int): The layer identifier for depth-dependent configurations.
        args (ModelArgs): Model configuration parameters.
    
    Returns:
        nn.Module: The appropriate attention module.
    """
    # Validate configuration first
    _validate_attention_config(args)
    
    if args.attention_type == AttentionType.DIFFERENTIAL_ATTENTION:
        # Create a copy to avoid modifying the original config
        # We know diff_attn_args is not None due to validation
        diff_args = dataclasses.replace(args.diff_attn_args, depth=layer_id)  # type: ignore
        return DiffAttention(diff_args)
    
    elif args.attention_type == AttentionType.MLA:
        # We know mla_config is not None due to validation, but add runtime safety
        if args.mla_config is None:
            raise ValueError("mla_config must be provided for MLA attention type")
        
        return MLA(args.mla_config)
    
    else:  # Default to SELF_ATTENTION
        standard_args = SelfAttnArgs(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len
        )
        return SelfAttention(standard_args)


@dataclass
class ModelArgs:
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
    score_function: str = "sigmoid"
    bias_update_speed: float = 0.001
    moe_aux_loss_weight: float = 0.01  # Weight for sequence-wise auxiliary loss

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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiTokenPredictor(nn.Module):
    """
    Multi-Token Prediction module following DeepSeek's architecture.
    Uses a single transformer block as per the original design.
    """
    def __init__(self, args: 'ModelArgs'):
        super().__init__()
        self.dim = args.dim
        self.num_prediction_tokens = args.num_prediction_tokens
        
        # RMS norms for input embedding and transformer output
        self.input_embedding_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.transformer_output_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Linear projection after concatenation
        # Input: concatenated [normalized_embedding, normalized_transformer_output]
        # Output: projected features for MTP transformer
        projection_input_dim = 2 * args.dim  # Concatenated features
        self.projection = nn.Linear(projection_input_dim, args.dim, bias=False)
        
        # Single MTP transformer block - use TransformerBlock for consistent initialization
        self.mtp_transformer_block = TransformerBlock(0, args)
        
        # Final norm
        self.output_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Multi-token output heads (if not sharing embeddings)
        if not args.mtp_share_embeddings:
            self.output_heads = nn.ModuleList([
                nn.Linear(args.dim, args.vocab_size, bias=False)
                for _ in range(self.num_prediction_tokens)
            ])
        else:
            self.output_heads = None
    
    def forward(self, 
                input_embeddings: torch.Tensor, 
                transformer_output: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                lm_head: Optional[nn.Linear] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_embeddings: (batch_size, seq_len, dim) - Original input embeddings
            transformer_output: (batch_size, seq_len, dim) - Output from main transformer
            start_pos: Starting position for attention caching
            freqs_cis: Frequency embeddings for rotary attention
            lm_head: Shared LM head if using shared embeddings
            mask: Attention mask
            
        Returns:
            multi_token_logits: (batch_size, seq_len, num_prediction_tokens, vocab_size)
        """
        batch_size, seq_len, dim = input_embeddings.shape
        
        # Step 1: Normalize input embedding and transformer output
        norm_input_emb = self.input_embedding_norm(input_embeddings)
        norm_transformer_out = self.transformer_output_norm(transformer_output)
        
        # Step 2: Concatenate normalized features
        concatenated = torch.cat([norm_input_emb, norm_transformer_out], dim=-1)
        
        # Step 3: Linear projection
        projected = self.projection(concatenated)
        
        # Step 4: Pass through single MTP transformer block
        mtp_output = self.mtp_transformer_block(projected, start_pos, freqs_cis, mask)
        
        # Step 5: Final normalization
        mtp_output = self.output_norm(mtp_output)
        
        # Step 6: Generate multi-token predictions
        if self.output_heads is None and lm_head is not None:
            # Use shared LM head - need to distinguish different prediction positions
            multi_token_logits = []
            for i in range(self.num_prediction_tokens):

                logits = lm_head(mtp_output)
                multi_token_logits.append(logits)
            multi_token_logits = torch.stack(multi_token_logits, dim=2)
        else:
            # Use separate heads for each prediction position
            if self.output_heads is None:
                raise ValueError("output_heads should not be None when not sharing embeddings")
            
            multi_token_logits = []
            for i, head in enumerate(self.output_heads):
                logits = head(mtp_output)
                multi_token_logits.append(logits)
            multi_token_logits = torch.stack(multi_token_logits, dim=2)
        
        return multi_token_logits


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        use_j_linear: bool = False
    ):
        """
        Initialize a unified TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.
            use_j_linear (bool): Whether to include the J linear transformation.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (nn.Module): Attention module.
            feed_forward (nn.Module): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            linear_j (Optional[nn.Linear]): Optional J linear transformation.
            logic_gate (Optional[LogicNetwork]): Optional logic network gate.
            use_logic_network (bool): Whether logic network is enabled.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.use_j_linear = use_j_linear
        
        # Create attention using factory function
        self.attention = _create_attention(layer_id, args)
        
        # Optional J linear transformation
        if use_j_linear:
            self.linear_j = nn.Linear(args.dim, args.dim)
        else:
            self.linear_j = None
        
        # Create feed forward - MoE only used with MLA attention
        if args.moe and args.attention_type == AttentionType.MLA:
            # Import MoE only when needed to avoid circular import
            from .moe import MoE
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )

        # Optional logic network
        if args.logic_network:
            self.logic_gate = LogicNetwork(args.dim)
            self.use_logic_network = True
        else:
            self.logic_gate = None
            self.use_logic_network = False

        # Normalization layers
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (Optional[torch.Tensor]): Precomputed cosine and sine frequencies.
            mask (Optional[torch.Tensor]): Masking tensor for attention.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        if isinstance(self.attention, MLA):
            # MLA handles normalization internally and has different signature
            h = x + self.attention(x, start_pos, freqs_cis, mask)[0]  # Take output, ignore scores
        else:
            # Standard attention flow
            x_norm = self.attention_norm(x)
            attn_out = self.attention(x_norm, start_pos, freqs_cis, mask)
            
            if self.use_j_linear and self.linear_j is not None:
                # TransformerBlockJ: attention + J linear
                h = x + attn_out + self.linear_j(x_norm)
            else:
                # Standard TransformerBlock: just attention
                h = x + attn_out

        # Apply logic network if enabled
        if self.use_logic_network and self.logic_gate is not None:
            h = h * self.logic_gate(h)

        # Feed forward
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SabiYarn(nn.Module):
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
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size,
            params.dim,
        )

        # Create layers with optional layer sharing 
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
        self.lm_head = nn.Linear(
            params.dim,
            params.vocab_size,
            bias=False,
        )
        if params.tie_weights == True:
            self.lm_head.weight = self.tok_embeddings.weight

        # Multi-token prediction module (only for MLA)
        if params.multi_token_prediction and params.attention_type == AttentionType.MLA:
            self.multi_token_predictor = MultiTokenPredictor(params)
            self.use_multi_token = True
        else:
            self.multi_token_predictor = None
            self.use_multi_token = False

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

    def _init_weights(self, module):
        """Initialize weights following DeepSeek paper recommendations."""
        # Handle standard PyTorch layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.params.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.params.init_std)
        
        # Handle MLA custom linear layers
        from .MLA import Linear, ColumnParallelLinear, RowParallelLinear
        if isinstance(module, (Linear, ColumnParallelLinear, RowParallelLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.params.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            # Initialize scale parameter for quantized weights
            if hasattr(module, 'scale') and module.scale is not None:
                torch.nn.init.ones_(module.scale)

    def get_model_size(self):
        # Calculate number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if self.params.layer_sharing:
            # Calculate effective vs actual depth
            effective_depth = self.params.n_layers
            actual_unique_layers = self.n_unique_layers
            memory_efficiency = f"{actual_unique_layers}×{self.repeat_factor} layers"
            return f"Trainable parameters: {trainable_params//1e6}M (Layer sharing: {memory_efficiency}, effective depth: {effective_depth})"
        else:
            return f"Trainable parameters: {trainable_params//1e6}M"

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
            
          
            if multi_token_logits is not None:
                # Gather multi-token logits across devices
                all_multi_logits = [torch.empty_like(multi_token_logits) for _ in range(self.params.world_size)]
                dist.all_gather(all_multi_logits, multi_token_logits)
                multi_token_logits = torch.cat(all_multi_logits, dim=-1)
        
        if multi_token_logits is not None:
            return hidden_states, logits, multi_token_logits
        else:
            return hidden_states, logits


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_multi_token=False):
        """
        Generate text using the model.
        Args:
            idx: Input token indices (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            use_multi_token: Whether to use multi-token prediction for faster generation
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.params.max_seq_len
                else idx[:, -self.params.max_seq_len :]
            )
            
            # forward the model to get the logits for the index in the sequence
            if use_multi_token and self.use_multi_token:
                _, logits, multi_token_logits = self(idx_cond, start_pos=0, return_multi_token=True)
                
                logits = logits[:, -1, :] / temperature

            else:
                _, logits = self(idx_cond, start_pos=0)
              
                logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



