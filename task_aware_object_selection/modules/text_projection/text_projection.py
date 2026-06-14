from __future__ import annotations

import torch
import torch.nn as nn


class TextProjection(nn.Module):
    """Projects a 384-D SentenceTransformer prompt embedding to a 192-D task embedding."""

    def __init__(self, input_dim: int = 384, output_dim: int = 192) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [384] or [B, 384]
            
        Returns:
            Tensor of shape [192] or [B, 192]
        """
        return self.projection(x)
