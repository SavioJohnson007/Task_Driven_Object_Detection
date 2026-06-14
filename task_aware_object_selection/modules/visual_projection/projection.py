from __future__ import annotations

import torch
import torch.nn as nn


class VisualProjection(nn.Module):
    """Projects RoI-aligned visual features to a 128-D visual embedding using convolutions and GAP."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [M, 256, 5, 5]
            
        Returns:
            Tensor of shape [M, output_dim]
        """
        # Expect RoI features already in [M, 256, 5, 5] form.
        if x.ndim != 4 or x.shape[1] != 256:
            raise ValueError(f"Expected input shape [M,256,5,5], got {tuple(x.shape)}")
        
        # Apply convolutions
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        
        # Global Average Pool to [M, 64, 1, 1]
        x = self.gap(x)
        
        # Flatten to [M, 64]
        x = x.view(x.size(0), -1)
        
        # Final projection to [M, output_dim]
        return self.fc(x)
