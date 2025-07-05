## Adapted from https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention/blob/main/Multi-Head%20Latent%20Attention.py
## Adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from .kernel import act_quant, weight_dequant, fp8_gemm
import math

world_size = 1 # the number of GPUs
rank = 0 # the rank of the current GPU
block_size = 128 # the block size of the input tensor. useful when using fp8
gemm_impl: Literal["bf16", "fp8"] = "bf16"

@dataclass
class MLAConfig:
    hidden_size: int
    num_heads: int
    max_batch_size: int
    original_seq_len: int #Original sequence length.  
    max_seq_len: int # Maximum sequence length.
    rope_theta: float  # frequency, usually large
    attention_dropout: float
    q_lora_rank: int  # latent shape, usually >10k
    qk_rope_head_dim: int  # 64
    kv_lora_rank: int  # 512
    v_head_dim: int  # 128
    qk_nope_head_dim: int
    attention_bias: bool
    mscale: float =1. # Scaling factor for extended attention.
    rope_factor: float = 40 #Scaling factor for extended sequence lengths.
    beta_fast: int = 32
    beta_slow: int = 1

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    # Ensure input and weight have the same dtype
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype or Linear.dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)

class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y

class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y
    
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.dim = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            hidden_states (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # return F.rms_norm(hidden_states, (self.dim,), self.weight, self.eps)
        # # Custom RMS norm implementation
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

def precompute_freqs_cis(args: MLAConfig) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelConfig): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    # Convert to float32 for complex operations
    x_float = x.float()
    x_complex = torch.view_as_complex(x_float.view(*x_float.shape[:-1], -1, 2))
    
    # Slice freqs_cis to match the actual sequence length
    seq_len = x_complex.size(1)
    freqs_cis = freqs_cis[:seq_len]  # Take only the needed sequence positions
    freqs_cis = freqs_cis.view(1, seq_len, 1, x_complex.size(-1))
    
    y = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return y.to(dtype)

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        num_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, config):
        super().__init__()
        # 1. MHA
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.n_local_heads = config.num_heads // world_size
        self.v_head_dim = config.v_head_dim

        self.out_proj = RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size)

        # 2. MLA compression
        # 2.1 down compression
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        # 2 parts
        # 2.1 down compression
        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
            dtype=torch.bfloat16,
        )
        self.q_down_norm = DeepseekV2RMSNorm(self.q_lora_rank)

        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=torch.bfloat16,
        )  # qk_rope_head_dim usually 64
        self.kv_down_norm = DeepseekV2RMSNorm(self.kv_lora_rank)
        # after down, two parts, need to split

        # 2.2 up compression
        # q, k shape is same
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_up_proj = ColumnParallelLinear(self.q_lora_rank, self.num_heads * self.qk_head_dim)  # also split

        self.kv_up_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads
            * (
                self.qk_nope_head_dim + self.v_head_dim
            ),  
        )

        self.softmax_scale = self.qk_head_dim**0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.kv_lora_rank, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.qk_rope_head_dim, dtype=torch.bfloat16), persistent=False)     

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        # x (b, seq_len, hidden_dim)
        bsz, q_len, _ = x.size()
        end_pos = start_pos + q_len

        # 1. q compression
        q = self.q_down_proj(x)
        q = self.q_down_norm(q)
        q = self.q_up_proj(q)# q shape: self.num_heads * self.qk_head_dim,(b, seq_len, self.num_heads * self.qk_head_dim,)
        q = q.view(bsz, q_len, self.n_local_heads, self.qk_head_dim)#.transpose(1, 2)
        # (b, num_head, seq_len, qk_head_dim)

        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        
        # kv part
        # c_kv: compressed kv
        c_kv = self.kv_down_proj(x)
        c_kv, k_rope = torch.split(
            c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # k_rope shape: (b, seq_len, self.qk_rope_head_dim)
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        kv_up_proj = self.kv_up_proj.weight if self.kv_up_proj.scale is None else weight_dequant(self.kv_up_proj.weight, self.kv_up_proj.scale, block_size)
        kv_up_proj = kv_up_proj.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, kv_up_proj[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_down_norm(c_kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_rope.squeeze(2)
        
        # Transpose query tensors to have correct dimension order for einsum
        q_nope = q_nope.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        q_rope = q_rope.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        scores = (torch.einsum("bhsc,btc->bhst", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bhsr,btr->bhst", q_rope, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        else:
            # Causal mask: only allow attending to current and previous positions
            q_len = scores.size(2)
            k_len = scores.size(3)
            causal_mask = torch.tril(torch.ones((q_len, k_len), device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        
        scores = F.softmax(scores, dim=-1).type_as(x)
        # scores = F.dropout(scores, p=self.attention_dropout, training=self.training)
        
        # Project kv_cache through up-projection to get the full value representations
        kv_cache_projected = torch.einsum("btc,hdc->bhtd", self.kv_cache[:bsz, :end_pos], kv_up_proj[:, -self.v_head_dim:])
        
        output = torch.einsum("bhst,bhtd->bhsd", scores, kv_cache_projected)
        output = output.transpose(1, 2).flatten(2)  # Transpose back and flatten
        output = self.out_proj(output)
        return output


