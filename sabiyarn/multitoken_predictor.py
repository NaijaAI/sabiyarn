from typing import Optional
import torch
from torch import nn


class MultiTokenPredictor(nn.Module):
    """
    Multi-Token Prediction module following DeepSeek's architecture.
    Uses a single transformer block as per the original design.
    """

    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.dim = args.dim
        self.num_prediction_tokens = args.num_prediction_tokens

        # RMS norms for input embedding and transformer output
        self.input_embedding_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.transformer_output_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Linear projection after concatenation
        # Input: concatenated [normalized_embedding, normalized_transformer_output]
        # Output: projected features for MTP transformer
        projection_input_dim = 2 * args.dim  # Concatenated features
        self.projection = nn.Linear(projection_input_dim, args.dim, bias=False)

        # Single MTP transformer block - use TransformerBlock for consistent initialization
        self.mtp_transformer_block = TransformerBlock(0, args)

        # Final norm
        self.output_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Multi-token output heads (if not sharing embeddings)
        if not args.mtp_share_embeddings:
            self.output_heads = nn.ModuleList(
                [
                    nn.Linear(args.dim, args.vocab_size, bias=False)
                    for _ in range(self.num_prediction_tokens)
                ]
            )
        else:
            self.output_heads = None

    def forward(
        self,
        input_embeddings: torch.Tensor,
        transformer_output: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        lm_head: Optional[nn.Linear] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (batch_size, seq_len, dim) - Original input embeddings
            transformer_output: (batch_size, seq_len, dim) - Output from main transformer
            start_pos: Starting position for attention caching
            freqs_cis: Frequency embeddings for rotary attention
            lm_head: Shared LM head if using shared embeddings
            mask: Attention mask

        Returns:
            multi_token_logits: (batch_size, seq_len, num_prediction_tokens, vocab_size)
        """
        batch_size, seq_len, dim = input_embeddings.shape

        # Step 1: Normalize input embedding and transformer output
        norm_input_emb = self.input_embedding_norm(input_embeddings)
        norm_transformer_out = self.transformer_output_norm(transformer_output)

        # Step 2: Concatenate normalized features
        concatenated = torch.cat([norm_input_emb, norm_transformer_out], dim=-1)

        # Step 3: Linear projection
        projected = self.projection(concatenated)

        # Step 4: Pass through single MTP transformer block
        mtp_output = self.mtp_transformer_block(projected, start_pos, freqs_cis, mask)

        # Step 5: Final normalization
        mtp_output = self.output_norm(mtp_output)

        # Step 6: Generate multi-token predictions
        if self.output_heads is None and lm_head is not None:
            # Use shared LM head - need to distinguish different prediction positions
            multi_token_logits = []
            for i in range(self.num_prediction_tokens):

                logits = lm_head(mtp_output)
                multi_token_logits.append(logits)
            multi_token_logits = torch.stack(multi_token_logits, dim=2)
        else:
            # Use separate heads for each prediction position
            if self.output_heads is None:
                raise ValueError(
                    "output_heads should not be None when not sharing embeddings"
                )

            multi_token_logits = []
            for i, head in enumerate(self.output_heads):
                logits = head(mtp_output)
                multi_token_logits.append(logits)
            multi_token_logits = torch.stack(multi_token_logits, dim=2)

        return multi_token_logits
