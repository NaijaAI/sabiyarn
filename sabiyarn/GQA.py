
""" Implementation of Grouped Query Attention"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from dataclasses import dataclass
from utils import precompute_freqs_cis, apply_rotary_emb, reshape_for_broadcast, repeat_kv

@dataclass
class GQAArgs:
    dim: int = 2048
    n_kv_heads: int = 8 #n_kv_heads can be less than n_heads
    n_heads: int = 16 # number of query heads
    max_seq_len: int = 2048
    max_batch_size: int = 32
    use_kv_cache = True

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
        freqs_cis: torch.Tensor, 
        mask:Optional[torch.Tensor]) -> torch.Tensor:
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
            xv = self.cache_v[:bsz, :start_pos+seq_len]
        
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
        return self.wo(output)



    




