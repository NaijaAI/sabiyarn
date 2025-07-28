import dataclasses
from typing import List, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from .MLA import Linear, ColumnParallelLinear, RowParallelLinear, linear

# Avoid circular import while maintaining type checking
if TYPE_CHECKING:
    from .model import ModelArgs
else:
    ModelArgs = None

# Default values for non-distributed setup
def get_distributed_info():
    """Get distributed training info with fallback for non-distributed setup."""
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        else:
            return 1, 0
    except:
        # Fallback for environments without torch.distributed
        return 1, 0

world_size, rank = get_distributed_info()

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """

        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        expert_bias (torch.nn.Parameter): Learnable bias per expert for load balancing.
        bias_update_speed (float): Learning rate multiplier for expert bias updates.
    """
    def __init__(self, args):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        
        #  expert bias for load balancing
        self.expert_bias = nn.Parameter(torch.zeros(args.n_routed_experts))
        self.bias_update_speed = args.bias_update_speed
        
        # Initialize weights
        nn.init.normal_(self.weight, std=args.init_std)
        
        # For auxiliary loss computation
        self.n_routed_experts = args.n_routed_experts
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Routing weights, selected expert indices, and raw scores.
        """
       
        scores = linear(x, self.weight)
        scores = scores.sigmoid()
        original_scores = scores
        
        # Add expert bias for load balancing
        scores = scores + self.expert_bias
        raw_scores = scores.clone()
        
        # Expert group routing (if using groups)
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # Select top-k experts
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights *= self.route_scale
        
        return weights.type_as(x), indices, raw_scores

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """

        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def compute_auxiliary_loss(gate_logits: torch.Tensor, expert_indices: torch.Tensor, 
                          num_experts: int, aux_loss_weight: float = 0.01) -> torch.Tensor:
    """
    Compute sequence-wise auxiliary loss for expert load balancing.
    
    Args:
        gate_logits: Raw gate logits (batch_size, seq_len, num_experts)
        expert_indices: Selected expert indices (batch_size, seq_len, topk)
        num_experts: Total number of experts
        aux_loss_weight: Weight for auxiliary loss
        
    Returns:
        Auxiliary loss tensor
    """
    batch_size, seq_len = gate_logits.shape[:2]
    
    # Compute expert probabilities from logits
    expert_probs = F.softmax(gate_logits, dim=-1)  # (batch_size, seq_len, num_experts)
    
    # Sequence-wise expert usage frequency
    # Sum probabilities across sequence dimension
    seq_expert_freq = expert_probs.sum(dim=1)  # (batch_size, num_experts)
    
    # Target uniform distribution across experts
    target_freq = seq_len / num_experts  # Each expert should get seq_len/num_experts tokens
    
    # Compute variance of expert usage (higher variance = poor load balancing)
    expert_variance = torch.var(seq_expert_freq, dim=-1)  # (batch_size,)
    
    # Alternative: Use coefficient of variation for scale-invariant loss
    expert_mean = torch.mean(seq_expert_freq, dim=-1)  # (batch_size,)
    coefficient_of_variation = expert_variance / (expert_mean + 1e-8)
    
    # Return mean auxiliary loss across batch
    aux_loss = coefficient_of_variation.mean() * aux_loss_weight
    
    return aux_loss

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
    
    def forward(self, x: torch.Tensor, return_aux_loss: bool = False) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.
            return_aux_loss (bool): Whether to return auxiliary loss for load balancing.

        Returns:
            torch.Tensor or Tuple: Output tensor after expert routing and computation.
                                  If return_aux_loss=True, returns (output, aux_loss).
        """
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        weights, indices, raw_scores = self.gate(x_flat)
        y = torch.zeros_like(x_flat)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x_flat[idx]) * weights[idx, top, None]
        
        z = self.shared_experts(x_flat)
        
        # All-reduce only in distributed setup
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            try:
                dist.all_reduce(y)
            except RuntimeError as e:
                # Fallback: if all_reduce fails, just use local results
                print(f"Warning: MoE all_reduce failed, using local results: {e}")
        
        output = (y + z).view(shape)
        
        if return_aux_loss:
            # Compute auxiliary loss for load balancing
            # Reshape raw_scores to match original input shape
            gate_logits = raw_scores.view(shape[0], -1, self.n_routed_experts)
            expert_indices_reshaped = indices.view(shape[0], -1, indices.size(-1))
            aux_loss = compute_auxiliary_loss(gate_logits, expert_indices_reshaped, self.n_routed_experts)
            return output, aux_loss
        
        return output