import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from ..cut_cross_entropy import linear_cross_entropy
    CCE_AVAILABLE = True
except ImportError:
    CCE_AVAILABLE = False
    linear_cross_entropy = None


class MultiTokenLoss(nn.Module):
    """
    Loss function for DeepSeek-style multi-token prediction.
    Computes the average cross-entropy loss across all prediction depths,
    multiplied by a weighting factor. Supports Cut Cross Entropy for memory efficiency.
    """
    
    def __init__(self, num_prediction_tokens: int, mtp_loss_weight: float = 1.0, use_cut_cross_entropy: bool = True):
        super().__init__()
        self.num_prediction_tokens = num_prediction_tokens
        self.mtp_loss_weight = mtp_loss_weight
        self.use_cut_cross_entropy = use_cut_cross_entropy and CCE_AVAILABLE
    
    def forward(self, multi_token_logits: torch.Tensor, targets: torch.Tensor, 
                mtp_hidden_states: Optional[torch.Tensor] = None, 
                mtp_output_heads: Optional[nn.ModuleList] = None) -> torch.Tensor:
        """
        Compute multi-token prediction loss.
        
        Args:
            multi_token_logits: (batch_size, seq_len, num_prediction_tokens, vocab_size)
            targets: (batch_size, seq_len + num_prediction_tokens) - extended targets
            mtp_hidden_states: Optional hidden states for Cut Cross Entropy
            mtp_output_heads: Optional output heads for Cut Cross Entropy
            
        Returns:
            Multi-token prediction loss (average across depths * weight)
        """
        batch_size, seq_len, num_pred_tokens, vocab_size = multi_token_logits.shape
        
        losses = []
        
        for i in range(num_pred_tokens):
            # Get targets shifted by (i+1) positions
            target_start = i + 1
            target_end = target_start + seq_len
            
            if target_end <= targets.shape[1]:
                pred_targets = targets[:, target_start:target_end]  # (B, S)
                
                if self.use_cut_cross_entropy and mtp_hidden_states is not None and mtp_output_heads is not None and linear_cross_entropy is not None:
                    # Use Cut Cross Entropy for memory efficiency
                    try:
                        loss = linear_cross_entropy(
                            mtp_hidden_states,  # Use MTP hidden states
                            mtp_output_heads[i].weight,  # Use specific head weights
                            pred_targets,
                            shift=False,  # Targets are already aligned
                            impl="torch_compile"
                        )
                    except Exception:
                        # Fallback to standard cross-entropy
                        pred_logits = multi_token_logits[:, :, i, :]  # (B, S, V)
                        loss = F.cross_entropy(
                            pred_logits.reshape(-1, vocab_size),
                            pred_targets.reshape(-1),
                            ignore_index=-1
                        )
                else:
                    # Standard cross-entropy
                    pred_logits = multi_token_logits[:, :, i, :]  # (B, S, V)
                    loss = F.cross_entropy(
                        pred_logits.reshape(-1, vocab_size),
                        pred_targets.reshape(-1),
                        ignore_index=-1
                    )
                losses.append(loss)
                
        # Average across all depths and multiply by weighting factor
        if losses:
            avg_loss = torch.stack(losses).mean()
            return self.mtp_loss_weight * avg_loss
        else:
            return torch.tensor(0.0, device=multi_token_logits.device)


def compute_combined_loss(next_token_logits: torch.Tensor, 
                         multi_token_logits: Optional[torch.Tensor],
                         targets: torch.Tensor,
                         multi_token_loss_fn: Optional[MultiTokenLoss] = None) -> dict:
    """
    Compute combined loss for standard next-token prediction and multi-token prediction.
    
    Args:
        next_token_logits: (batch_size, seq_len, vocab_size) - Standard next-token logits
        multi_token_logits: (batch_size, seq_len, num_prediction_tokens, vocab_size) - MTP logits
        targets: (batch_size, seq_len + num_prediction_tokens) - Extended targets
        multi_token_loss_fn: Multi-token loss function
        
    Returns:
        Dictionary with loss components
    """
    # Standard next-token loss
    next_token_loss = F.cross_entropy(
        next_token_logits.reshape(-1, next_token_logits.size(-1)),
        targets[:, 1:targets.shape[1] - (multi_token_logits.shape[2] if multi_token_logits is not None else 0)].reshape(-1),
        ignore_index=-1
    )
    
    loss_dict = {
        'next_token_loss': next_token_loss,
        'total_loss': next_token_loss
    }
    
    # Multi-token loss (if enabled)
    if multi_token_logits is not None and multi_token_loss_fn is not None:
        multi_token_loss = multi_token_loss_fn(multi_token_logits, targets)
        loss_dict['multi_token_loss'] = multi_token_loss
        loss_dict['total_loss'] = next_token_loss + multi_token_loss
    
    return loss_dict
