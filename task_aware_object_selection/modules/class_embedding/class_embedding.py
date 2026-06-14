from __future__ import annotations

import torch
import torch.nn as nn


class ClassEmbedding(nn.Module):
    """Learns an embedding representation for category/class IDs."""

    def __init__(self, num_classes: int = 91, embedding_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_ids: LongTensor of shape [M]
            
        Returns:
            Tensor of shape [M, embedding_dim]
        """
        return self.embedding(class_ids)
