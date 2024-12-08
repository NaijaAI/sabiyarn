import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple

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


class DiffAttention(nn.Module):
  def __init__(self, embed_dim, head_dim, context_length, depth):
    super().__init__()
    self.embed_dim = embed_dim
    self.head_dim = head_dim // 2 # half of transformers head
    self.num_heads = self.embed_dim // self.head_dim
    self.scaling = head_dim ** -0.5
    print(f" The model now has {self.num_heads} heads with dimension {self.head_dim}")

    self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
    self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
    self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    def lambda_init_fn(depth):
      return 0.8 - 0.6 * math.exp(-0.3 * depth)

    self.lambda_init = lambda_init_fn(depth)
    self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
    self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
    self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
    self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))

    self.subln = RMSNorm(2 * self.head_dim, eps=1e-5,)
  def forward(self, x,attn_mask=None):
    bsz, tgt_len, _ = x.shape
    src_len = tgt_len

    q = self.wq(x)
    k = self.wk(x)
    v = self.wv(x)
    q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
    k = k.view(bsz, tgt_len, self.num_heads, self.head_dim)
    v = v.view(bsz, tgt_len, self.num_heads, self.head_dim)
    cls_freq = precompute_freqs_cis(self.head_dim, tgt_len)
    q,k = apply_rotary_emb(q, k, cls_freq)

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
    print(f"after masking{attn_weights} with shape {attn_weights.shape} ")
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
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
