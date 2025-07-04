## Adapted from https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention/blob/main/Multi-Head%20Latent%20Attention.py
## Adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from kernel import act_quant, weight_dequant, fp8_gemm
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
            self.bias = nn.Parameter(torch.empty(out_features))
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
        return F.rms_norm(hidden_states, (self.dim,), self.weight, self.eps)

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # use a different permutation to obtain same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

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

    t = torch.arange(seqlen)
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
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
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
        )
        self.q_down_norm = DeepseekV2RMSNorm(self.q_lora_rank)

        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
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

        # 3. rope
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            config.qk_rope_head_dim,
            config.original_seq_len,
            config.rope_theta,
        )

        self.softmax_scale = self.qk_head_dim**0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.qk_rope_head_dim), persistent=False)     

    def forward(self, hidden_states, start_pos, freqs_cis, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            hidden states (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            attention mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # hidden_states (b, seq_len, hidden_dim)
        bsz, q_len, _ = hidden_states.size()
        end_pos = start_pos + q_len

        # 1. q compression
        q = self.q_down_proj(hidden_states)
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
        c_kv = self.kv_down_proj(hidden_states)
        c_kv, k_rope = torch.split(
            c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # k_rope shape: (b, seq_len, self.qk_rope_head_dim)
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        kv_up_proj = self.kv_up_proj.weight if self.kv_up_proj.scale is None else weight_dequant(self.kv_up_proj.weight, self.kv_up_proj.scale, block_size)
        kv_up_proj = kv_up_proj.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, kv_up_proj[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_down_norm(c_kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_rope.squeeze(2)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_rope, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        else:
            q_len = scores.size(2)
            k_len = self.kv_cache.size(2)
            scores = scores[:, :, :q_len, :k_len]
        scores = F.softmax(scores, dim=-1).type_as(hidden_states)
        # scores = F.dropout(scores, p=self.attention_dropout, training=self.training)
        output = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        output = torch.einsum("bshc,hdc->bshd", output, kv_up_proj[:, -self.v_head_dim:])
        output = output.view(bsz, q_len, -1)
        output = self.out_proj(output)
        return output, scores


# ----
def test_mla():
    config = MLAConfig(
        hidden_size=7168,
        num_heads=16,
        max_seq_len=1024*4,
        max_batch_size=2,
        attention_dropout=0.1,
        #mla
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        v_head_dim=128,
        qk_nope_head_dim=128,
        attention_bias=False,
        # yarn
        original_seq_len= 4096,
        rope_theta = 10000.0,
        rope_factor = 40,
        beta_fast = 32,
        beta_slow = 1,
        mscale = 1.
    )
    mla = MLA(config)
    x = torch.randn(2, 1024, 7168)
    start_pos = 0
    freqs_cis = precompute_freqs_cis(config)
    # position_ids = (
    #     torch.arange(
    #         config.original_seq_len,
    #     )
    #     .unsqueeze(0)
    #     .expand(x.size(0), -1)
    # )  # (batch_size, seq_len)
    attn_output, attn_weights = mla(x, start_pos, freqs_cis)
    print(attn_output.shape)
    print(attn_weights.shape)


test_mla()
