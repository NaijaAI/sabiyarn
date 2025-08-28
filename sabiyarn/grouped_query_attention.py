
""" Implementation of Grouped Query Attention"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class GQAArgs:
    dim: int = 2048
    n_kv_heads: int = 8 #n_kv_heads can be less than n_heads
    n_heads: int = 16 # number of query heads
    max_seq_len: int = 2048
    max_batch_size: int = 32
    use_kv_cache = True


def precompute_freqs_cis(dim: int, end:int, theta: float=10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension "dim"
    and the end index "end". The 'Theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

     Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

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

    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
    ):

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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""

    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:,:,None, :].expand(
            bs, slen, n_kv_heads, n_rep, head_dim
        ).reshape(
            bs, slen, n_kv_heads*n_rep, head_dim
        )
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, args: GQAArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "number of query heads must be divisible by n_kv_heads"
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_kv_cache = args.use_kv_cache

        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False
        )

        self.wk = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wv = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        if self.use_kv_cache:
            self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim
            )
            )
            self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                args.n_kv_heads,
                self.head_dim
            )
        )

        self.wo = nn.Linear(
            args.dim,
            args.dim,
            bias=False
        )

    def forward(self, 
        x: torch.Tensor, 
        start_pos:int,
        freqs_cis: torch.Tensor, mask:Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the  GroupedQueryAttention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        if self.use_kv_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xv)

            self.cache_k[:bsz, start_pos: start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos: start_pos + seq_len] = xv

            xk = self.cache_k[:bsz, :start_pos+seq_len]
            xv = self.cache_l[:bsz, :start_pos+seq_len]
        
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.to(scores.device)
        
        else:
            seq_len = scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.bool(), float("-inf"))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)
            output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)
            return self.ffn(output)



    




