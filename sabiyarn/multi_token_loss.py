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
                            ignore_index=-100
                        )
                else:
                    # Standard cross-entropy
                    pred_logits = multi_token_logits[:, :, i, :]  # (B, S, V)
                    loss = F.cross_entropy(
                        pred_logits.reshape(-1, vocab_size),
                        pred_targets.reshape(-1),
                        ignore_index=-100
                    )
                losses.append(loss)
                
        # Average across all depths and multiply by weighting factor
        if losses:
            avg_loss = torch.stack(losses).mean()
            return self.mtp_loss_weight * avg_loss
        else:
            return torch.tensor(0.0, device=multi_token_logits.device)