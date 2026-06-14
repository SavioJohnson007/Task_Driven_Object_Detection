from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Detection:
    """Single object detection in absolute image coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return x1, y1, x2 - x1, y2 - y1

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_xywh": list(self.bbox_xywh),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO for object proposal generation."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self._configure_local_cache()

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for YOLODetector. "
                "Install it before running detection."
            ) from exc

        self.model = YOLO(str(self.model_path))

    def detect(self, image_path: str | Path) -> list[Detection]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        return self._parse_result(results[0])

    def detect_as_dicts(self, image_path: str | Path) -> list[dict[str, Any]]:
        return [detection.as_dict() for detection in self.detect(image_path)]

    def _parse_result(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        detections: list[Detection] = []

        for box in boxes:
            xyxy = self._to_float_tuple(box.xyxy[0], expected_len=4)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id))

            detections.append(
                Detection(
                    bbox_xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections

    @staticmethod
    def _to_float_tuple(values: Iterable[Any], expected_len: int) -> tuple[float, ...]:
        converted = tuple(float(value) for value in values)
        if len(converted) != expected_len:
            raise ValueError(f"Expected {expected_len} values, got {len(converted)}")
        return converted

    @staticmethod
    def _configure_local_cache() -> None:
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / ".cache"
        matplotlib_dir = cache_dir / "matplotlib"

        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "yolo" / "yolov8n.pt"


def create_default_detector(
    confidence_threshold: float = 0.25,
    image_size: int = 640,
    device: str | None = None,
) -> YOLODetector:
    return YOLODetector(
        model_path=default_model_path(),
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        device=device,
    )
