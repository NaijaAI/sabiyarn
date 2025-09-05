"""Implementation of Standard Multi-Head Attention"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from dataclasses import dataclass
from .utils import apply_rotary_emb

@dataclass
class SelfAttnArgs:
    dim: int = 4096
    n_heads: int = 32
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_kv_cache: bool = True
    bias: bool = False
    dropout: bool= 0.1

class CausalSelfAttention(nn.Module):

    def __init__(self, config: SelfAttnArgs):
        super().__init__()
        assert config.dim % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_heads
        self.n_embd = config.dim
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                        .view(1, 1, config.max_seq_len, config.max_seq_len))
        
        self.use_kv_cache = config.use_kv_cache  
        if self.use_kv_cache:
            self.cache_k = torch.zeros(
            (
                config.max_batch_size,
                config.max_seq_len,
                self.n_head,
                config.dim // config.n_heads
            )
            )
            
            self.cache_v = torch.zeros(
            (
                config.max_batch_size,
                config.max_seq_len,
                self.n_head,
                config.dim // config.n_heads
            )
        )


    def forward(self, x: torch.Tensor, 
                    start_pos:int,
                    freqs_cis: torch.Tensor, 
                    mask:Optional[torch.Tensor]= None)-> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        k= k.transpose(1, 2) # (B, nh, T, hs)
        
        if self.use_kv_cache:
            self.cache_k = self.cache_k.to(q)
            self.cache_v = self.cache_v.to(q)

            self.cache_k[:B, start_pos: start_pos + T] = k
            self.cache_v[:B, start_pos: start_pos + T] = v

            k = self.cache_k[:B, :start_pos + T]
            v = self.cache_v[:B, :start_pos + T]
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = mask or self.mask
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



# class GQA(nn.Module):
#     """Multi-head attention module."""

#     def __init__(self, config: SelfAttnArgs):
#         """
#         Initialize the Attention module.

#         Args:
#             args (SelfAttnArgs): Model configuration parameters.

#         Attributes:
#             n_kv_heads (int): Number of key and value heads.
#             n_local_heads (int): Number of local query heads.
#             n_local_kv_heads (int): Number of local key and value heads.
#             n_rep (int): Number of repetitions for local heads.
#             head_dim (int): Dimension size of each attention head.
#             wq (nn.Linear): Linear transformation for queries.
#             wk (nn.Linear): Linear transformation for keys.
#             wv (nn.Linear): Linear transformation for values.
#             wo (nn.Linear): Linear transformation for output.
#             cache_k (torch.Tensor): Cached keys for attention.
#             cache_v (torch.Tensor): Cached values for attention.

#         """
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         model_parallel_size = 1
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads

#         self.wq = nn.Linear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#         )
#         self.wk = nn.Linear(
#             args.dim,
#             args.dim,
#             bias=False,
#         )
#         self.wv = nn.Linear(
#             args.dim,
#             args.dim,
#             bias=False,
#         )
#         self.wo = nn.Linear(
#             args.n_heads * self.head_dim,
#             args.dim,
#             bias=False,
#         )

#         self.cache_k = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()
#         self.cache_v = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         """
#         Forward pass of the attention module.

#         Args:
#             x (torch.Tensor): Input tensor.
#             start_pos (int): Starting position for caching.
#             freqs_cis (torch.Tensor): Precomputed frequency tensor.
#             mask (torch.Tensor, optional): Attention mask tensor.

#         Returns:
#             torch.Tensor: Output tensor after attention.

#         """
#         bsz, seqlen, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

#         # self.cache_k = self.cache_k.to(xq)
#         # self.cache_v = self.cache_v.to(xq)

#         # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
#         # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

#         # # repeat k/v heads if n_kv_heads < n_heads
#         # keys = repeat_kv(
#         #     keys, self.n_rep
#         # )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
#         # values = repeat_kv(
#         #     values, self.n_rep
#         # ) # (bs, cache_len + seqlen, n_local_heads, head_dim)

#         xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
#         xk = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         xv = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask.to(
#                 scores.device
#             )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
#         else:
#             # Apply causal masking
#             seq_len = scores.size(-1)
#             causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
#             scores = scores.masked_fill(causal_mask.bool(), float('-inf'))      
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#         output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
#         return self.wo(output)