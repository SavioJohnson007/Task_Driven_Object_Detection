from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torchvision
from modules.detector.detector import YOLODetector


def extract_roi_features(
    detector: YOLODetector,
    image_path: str | Path,
    boxes: torch.Tensor,
    output_size: tuple[int, int] = (5, 5),
) -> torch.Tensor:
    """
    Extracts RoI visual features from the final neck layer of the YOLOv8 model.
    
    Args:
        detector: An instance of YOLODetector.
        image_path: Path to the input image.
        boxes: PyTorch tensor of shape [M, 4] containing bounding box coordinates in xyxy format.
        output_size: Spatial dimensions of the pooled output features (height, width).
        
    Returns:
        A PyTorch tensor of shape [M, 256, H, W] containing the pooled RoI features.
    """
    if len(boxes) == 0:
        return torch.zeros((0, 256 * output_size[0] * output_size[1]), dtype=torch.float32)

    feature_maps: dict[str, torch.Tensor] = {}

    def hook_fn(module: torch.nn.Module, input: Any, output: torch.Tensor) -> None:
        feature_maps["neck_features"] = output

    # In YOLOv8 model, the Detect head is the last layer (model[-1]).
    # The final neck layer output before the head is model[-2] (layer 21).
    yolo_module = detector.model.model
    target_layer = yolo_module.model[-2]
    
    # Register the forward hook
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Trigger the forward pass
        detector.model.predict(
            source=str(image_path),
            imgsz=detector.image_size,
            device=detector.device or "cpu",
            verbose=False,
        )
    finally:
        # Ensure the hook is always removed
        handle.remove()

    neck_feat = feature_maps.get("neck_features")
    if neck_feat is None:
        raise ValueError("Failed to extract neck features from the YOLOv8 forward pass.")

    # Ensure boxes are on the same device as neck features
    boxes = boxes.to(neck_feat.device)

    # The final neck layer has a stride of 32 relative to the input image size.
    # Therefore, the spatial scale is 1 / 32 = 0.03125.
    spatial_scale = 1.0 / 32.0

    # torchvision.ops.roi_align expects a list of tensors, one tensor per batch element,
    # containing bounding boxes in [x1, y1, x2, y2] format.
    roi_feats = torchvision.ops.roi_align(
        neck_feat,
        [boxes],
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True,
    )

    # Return RoI features in [M, 256, H, W] shape directly.
    return roi_feats
