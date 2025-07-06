import math
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
        # We know mla_config is not None due to validation
        return MLA(args.mla_config)  # type: ignore
    
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
    logic_network: Optional[bool] = False
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_j: bool = True
    tie_weights: bool = True
    attention_type: str = AttentionType.SELF_ATTENTION
    diff_attn_args: Optional[DiffAttnArgs] = None
    mla_config: Optional[MLAConfig] = None


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
        # Calculate number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Trainable parameters: {trainable_params//1e6}M"

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.params.max_seq_len
                else idx[:, -self.params.max_seq_len :]
            )
            # forward the model to get the logits for the index in the sequence
            _, logits = self(idx_cond, start_pos=0)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def forward(self, tokens: torch.Tensor, start_pos: int, mask=None):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.
            mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Hidden states and output logits.
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
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
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        hidden_states = self.norm(h)
        logits = self.lm_head(hidden_states).float()
        return hidden_states, logits
