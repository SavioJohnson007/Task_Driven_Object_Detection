from __future__ import annotations

import torch
import torch.nn as nn


class ObjectSelector(nn.Module):
    """
    Applies threshold filtering followed by ArgMax selection to identify
    the best matching object proposal.
    """

    def __init__(self, default_threshold: float = 0.0) -> None:
        super().__init__()
        self.default_threshold = default_threshold

    def forward(
        self,
        logits: torch.Tensor,
        threshold: float | None = None,
    ) -> tuple[int, float]:
        """
        Selects the index of the highest scoring object proposal.
        
        Args:
            logits: Tensor of shape [M] containing preference scores (logits)
            threshold: Optional threshold override. If the highest score is
                       less than this threshold, returns index -1.
                       
        Returns:
            A tuple containing:
                best_idx: Index of the selected object, or -1 if none exceed threshold or list is empty.
                best_score: The relevance score of the selected object (or max score if empty/filtered).
        """
        M = logits.shape[0]
        if M == 0:
            return -1, float("-inf")

        t = self.default_threshold if threshold is None else threshold

        # Find argmax and max value
        max_val, argmax_idx = torch.max(logits, dim=0)
        best_score = float(max_val.item())
        best_idx = int(argmax_idx.item())

        if best_score < t:
            # Under threshold: no suitable object found
            return -1, best_score

        return best_idx, best_score
