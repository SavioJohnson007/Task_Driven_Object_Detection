from __future__ import annotations

import torch
import torch.nn as nn


class RankingLayer(nn.Module):
    """
    Computes preference logits for object proposals by combining
    task similarity, detector confidence, and class priors through a small MLP.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        similarity_scores: torch.Tensor,
        confidences: torch.Tensor,
        category_priors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            similarity_scores: Tensor of shape [M]
            confidences: Tensor of shape [M]
            category_priors: Optional tensor of shape [M]
            
        Returns:
            Tensor of shape [M] containing preference logit scores
        """
        M = similarity_scores.shape[0]
        if M == 0:
            return torch.zeros((0,), dtype=torch.float32)

        similarity_scores = similarity_scores.view(-1, 1)
        confidences = confidences.view(-1, 1)

        if category_priors is None:
            category_priors = torch.zeros_like(similarity_scores)
        else:
            category_priors = category_priors.view(-1, 1)

        features = torch.cat([similarity_scores, confidences, category_priors], dim=1)
        logits = self.mlp(features).view(-1)
        return logits
