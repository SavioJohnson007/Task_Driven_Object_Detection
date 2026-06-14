from __future__ import annotations

import torch
import torch.nn as nn


class SpatialEmbedding(nn.Module):
    """Normalizes bounding box coordinates and projects them to a 32-D spatial embedding."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, output_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            boxes: Tensor of shape [M, 4] containing absolute coordinates [x1, y1, x2, y2]
            img_size: Tuple containing (height, width) of the image
            
        Returns:
            Tensor of shape [M, output_dim]
        """
        if len(boxes) == 0:
            return torch.zeros((0, self.mlp[-1].out_features), dtype=torch.float32)
            
        H, W = img_size
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Normalize
        x1_norm = x1 / W
        y1_norm = y1 / H
        x2_norm = x2 / W
        y2_norm = y2 / H
        w_norm = w / W
        h_norm = h / H
        area_norm = area / (W * H)

        spatial_feats = torch.stack(
            [x1_norm, y1_norm, x2_norm, y2_norm, w_norm, h_norm, area_norm],
            dim=1
        )

        return self.mlp(spatial_feats)
