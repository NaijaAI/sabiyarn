import torch
from torch import nn
import math

class DiffAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, head_dim, context_length, depth):
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.head_dim = head_dim # half of transformers head
    self.scaling = head_dim **0.5

    self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
    self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
    self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
    def lambda_init_fn(depth):
      return 0.8 - 0.6 * math.exp(-0.3 * depth)

    self.lambda_init = lambda_init_fn(depth)
    self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
    self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
    self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
    self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))

    self.subln = RMSNorm(2 * self.head_dim, eps=1e-5,)
  def forward(self, x, rel_pos, attn_mask=None):
    bsz, tgt_len, embed_dim = x.shape
    src_len = tgt_len

    q = self.w_q(x)
    k = self.w_k(X)
    v = self.w_v(X)
    q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
    v = v.view(bsz, tgt_len, self.num_heads, self.head_dim)
    k = k.view(bsz, tgt_len, self.num_heads, self.head_dim)

    cls_freq = precompute_freqs_cis(self.head_dim, )
    q,k = apply_rotary_emb(q,k *rel_pos, interleaved=True)

    offset = src_len - tgt_len
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    attn_weights = torch.matmul(q, k.transpose(2,3)) / self.scaling
    if attn_mask is None:
      attn_mask = torch.triu(
          torch.zeros((tgt_len, tgt_len)
          ).float()
          .type_as(attn_weights),
          diagonal=1+offset
      )
    attn_weights = torch.nan_to_num(attn_weights)
    attn_weights += attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dytpe=torch.float32).type_As(
        attn_weights
    )

    lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
    lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
    lambda_full = lambda_1 - lambda_2 + self.lambda_init
    attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
    attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
    attn = torch.matmul(attn_weights, v)
    attn = self.subln(attn)
    attn = attn.transpose(1,2).reshape(bsz, tgt_len, self.num_heads*2*self.head_dim)
    attn = self.out_proj(attn)
