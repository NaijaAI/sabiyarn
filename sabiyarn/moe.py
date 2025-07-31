# sequence wise loss adapted from https://gist.github.com/TeaPoly/b5e046d9efa93fa7e38880b4c7e5ec5f
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
        self.score_function = args.score_function
        self.aux_loss_weight = args.moe_aux_loss_weight
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.n_routed_experts = args.n_routed_experts
        
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
        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)

        scores = linear(x, self.weight)
        if self.score_function == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(dim=-1, dtype=torch.float32)

        original_scores = scores
        
        # Add expert bias for load balancing
        scores = scores + self.expert_bias

        
        # Expert group routing (if using groups)
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # Select top-k experts
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # complementary sequence-wise auxiliary loss
        if self.training and self.aux_loss_weight > 0.0:
            scores_for_aux = original_scores
            topk_idx_for_aux_loss = indices.view(bsz, -1)
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=x.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * self.topk, device=x.device),
            ).div_(seq_len * self.topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.aux_loss_weight
        else:
            aux_loss = None
        
        if self.training and self.expert_bias is not None:
            with torch.no_grad():
                counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
                self.update_expert_bias(counts)
            
        weights = original_scores.gather(1, indices)
        if self.score_function == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        
        return weights.type_as(x), indices, aux_loss
    
    def update_expert_bias(self, counts: torch.Tensor):
        """
        Update expert bias based on counts.
        """
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, dist.ReduceOp.SUM)
        avg_count = counts.float().mean()
        error = avg_count - counts.float()
        self.expert_bias.add_(torch.sign(error) * self.bias_update_speed)
        

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        weights, indices, aux_loss = self.gate(x)
        x_flat = x.view(-1, self.dim)
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
        if self.training and aux_loss is not None:
            y= AddAuxLoss.apply(y, aux_loss)
        output = (y + z).view(shape)
        
        return output
    
class AddAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, aux_loss):
        assert aux_loss.numel() == 1, "Auxiliary loss must be a scalar"
        ctx.dtype = aux_loss.dtype
        ctx.required_aux_loss = aux_loss.requires_grad
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss =  torch.ones(1, dtype=ctx.dtype, device= grad_output.device)
        return grad_output, grad_loss
