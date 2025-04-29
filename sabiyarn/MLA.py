import torch
import math
from torch import nn
import torch.nn.functional as F
from differential_attention import RMSNorm
from dataclasses import dataclass
from typing import Literal, Optional


attn_impl = Literal["naive", "absorb"] = "absorb"


@dataclass
class MLAArgs:
    pass


class MLA(nn.Module):
    """
    Multihead Latent Attention Layer

    Attributes:
    dim: Model dimensions
    n_heads(int): Number of attention heads
    q_lora_rank(int): Rank for low-rank query projection.
    kv_lora_rank(int): Rank for low-rank key/value projection.
    qk_nope_head_dim(int): Dimensionality for non-positional query/key projections.
    qk_rope_head_dim(int): Dimensionality of rotary-positional query/key projections.
    qk_head_dim(int): Dimenisonality of query_key projection.
    v_head_dim(int): Dimensionality of value projection
    softmax_scale (float): Scaling factor for softmax in attention.
    
    """
    def __init__(self, args: MLAArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kq_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wQ_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the multi-head latent attention module
        Args:
            x( torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis(torch.tensor): Precomputed complex exponential values for rotary embeddings.
            mask(optinal[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim])