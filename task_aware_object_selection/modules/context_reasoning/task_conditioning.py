from __future__ import annotations

import torch
import torch.nn as nn


class TaskConditioning(nn.Module):
    """FiLM-style modulation of visual features using task embeddings."""

    def __init__(self, task_dim: int = 192, visual_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim),
        )

    def forward(self, visual_feats: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: Tensor of shape [M, visual_dim]
            task_emb: Tensor of shape [task_dim] or [1, task_dim]

        Returns:
            Tensor of shape [M, visual_dim]
        """
        if visual_feats.numel() == 0:
            return visual_feats

        if task_emb.ndim == 2 and task_emb.shape[0] == 1:
            task_emb = task_emb.squeeze(0)

        gamma = self.gamma(task_emb)
        beta = self.beta(task_emb)

        # Preserve a baseline scale and then apply task modulation.
        gamma = 1.0 + gamma

        return visual_feats * gamma.unsqueeze(0) + beta.unsqueeze(0)
