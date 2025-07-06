"""Implementation of Multihead Differential Attention """

import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffAttnArgs:
    depth: Optional[int] = None  # will be determined in the model
    max_batch_size: int = 32
    n_heads: int = 32  # half of the transformers num_head
    embed_dim: int = 4096
    n_kv_heads: Optional[int] = None
    max_seq_len: int = 2048
    norm_eps: int = 1e-5


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def lambda_init_fn(depth):
    """
    Function for calculating Lambda_init
    Args:
          depth (int): Decoder layer index containing the attention mechanism.
    Returns:
          float: lambda init value.
    """

    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DiffAttention(nn.Module):
    def __init__(self, args: DiffAttnArgs):
        """
        Initialize the Differential Attention Module.

        Args:
            args (DiffAttnArgs): Model configuration parameters.
        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            out_proj (nn.Linear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
            lambda_init: initial lambda value
            lambda_q1 (nn.Parameter): lambda for queries in first attention map
            lambda_q2 (nn.Parameter): lambda for queries in second attention map
            lambda_k1 (nn.Parameter): lambda for keys in first attention map
            lambda_k2 (nn.Parameter): lambda for keys in second attention map
            sublayer_norm: RMSNorm for sub attention layers
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.num_heads = args.n_heads  # half of transformers head
        self.head_dim = args.embed_dim // args.n_heads // 2
        self.scaling = self.head_dim**-0.5
        self.depth = args.depth
        self.wq = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.wk = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.wv = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.out_proj = nn.Linear(args.embed_dim, args.embed_dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, args.n_heads * 2, self.head_dim)
        )

        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, args.n_heads, self.head_dim * 2)
        )

        self.lambda_init = lambda_init_fn(self.depth if self.depth is not None else 1)
        self.lambda_q1 = nn.Parameter(
            torch.normal(mean=0, std=0.1, size=(self.head_dim,), dtype=torch.float32)
        )
        self.lambda_q2 = nn.Parameter(
            torch.normal(mean=0, std=0.1, size=(self.head_dim,), dtype=torch.float32)
        )
        self.lambda_k1 = nn.Parameter(
            torch.normal(mean=0, std=0.1, size=(self.head_dim,), dtype=torch.float32)
        )
        self.lambda_k2 = nn.Parameter(
            torch.normal(mean=0, std=0.1, size=(self.head_dim,), dtype=torch.float32)
        )

        self.sublayer_norm = RMSNorm(
            2 * self.head_dim,
            eps=1e-5,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the differential attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            attn_mask (torch.Tensor, Optional): Attention mask tensor.
        Returns:
            torch.Tensor: Output tensor after attention
        """
        bsz, tgt_len, _ = x.shape
        src_len = tgt_len

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        self.cache_k = self.cache_k.to(k)
        self.cache_v = self.cache_v.to(v)

        # self.cache_k[:bsz, start_pos : start_pos + src_len] = k
        # self.cache_v[:bsz, start_pos : start_pos + src_len] = v

        # keys = self.cache_k[:bsz, : start_pos + src_len]
        # values = self.cache_v[:bsz, : start_pos + src_len]

        offset = src_len - tgt_len
        q *= self.scaling

        q = q.transpose(1, 2)
        keys = k.transpose(1, 2)
        values = v.transpose(1, 2)

        attn_scores = torch.matmul(q, keys.transpose(2, 3))
        if mask is None:
            # Create proper causal mask with -inf for future positions
            mask = torch.triu(
                torch.ones((tgt_len, tgt_len), device=attn_scores.device) * float('-inf'),
                diagonal=1 + offset,
            )
        
        attn_scores = torch.nan_to_num(attn_scores)
        attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(
            attn_scores
        )
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Split attention weights for differential attention
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights_diff = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # Apply attention to values (values shape: bsz, num_heads, tgt_len, 2*head_dim)
        ctx_vec = torch.matmul(attn_weights_diff, values)
        ctx_vec = self.sublayer_norm(ctx_vec)
        ctx_vec = ctx_vec.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )
        ctx_vec = self.out_proj(ctx_vec)

        return ctx_vec
