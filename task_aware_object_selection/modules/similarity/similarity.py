from __future__ import annotations

import torch
import torch.nn as nn


class SimilarityCalculation(nn.Module):
    """Computes cosine-like similarity between context-aware object features and the task embedding."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, object_feats: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_feats: Tensor of shape [M, 192]
            task_emb: Tensor of shape [192] or [1, 192] or [B, 192]
            
        Returns:
            Tensor of shape [M] containing similarity scores
        """
        M = object_feats.shape[0]
        if M == 0:
            return torch.zeros((0,), dtype=torch.float32)

        # Align shapes of task_emb to [192]
        if task_emb.ndim == 2:
            if task_emb.shape[0] == 1:
                task_emb = task_emb.squeeze(0)
            else:
                task_emb = task_emb[0]

        object_feats = torch.nn.functional.normalize(object_feats, dim=1, eps=1e-8)
        task_emb = torch.nn.functional.normalize(task_emb, dim=0, eps=1e-8)

        return torch.matmul(object_feats, task_emb)
