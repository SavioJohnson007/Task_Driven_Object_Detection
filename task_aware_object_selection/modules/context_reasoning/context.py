from __future__ import annotations

import math
import torch
import torch.nn as nn


class ResidualContextPropagation(nn.Module):
    """
    Implements per-object relational context propagation over object proposal features.
    """

    def __init__(self, feature_dim: int = 192) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.context_transform = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        object_feats: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            object_feats: Tensor of shape [M, 192]
            scores: Tensor of shape [M] containing YOLO confidence scores
            
        Returns:
            Tensor of shape [M, 192] with per-object propagated context
        """
        M = object_feats.shape[0]
        if M == 0:
            return object_feats
        if M == 1:
            return object_feats

        # Build a linear O(M) context aggregation weighted by detector confidence.
        # Each object receives the pooled signal from all other objects, excluding itself.
        weighted_feats = scores.unsqueeze(1) * object_feats  # Shape [M, 192]
        global_context = weighted_feats.sum(dim=0, keepdim=True) - weighted_feats

        # Normalize by total confidence mass across other objects for numerical stability.
        total_confidence = scores.sum().clamp_min(1e-6)
        normalized_context = global_context / total_confidence

        transformed_context = self.context_transform(normalized_context)
        object_feats_updated = object_feats + transformed_context

        return object_feats_updated
