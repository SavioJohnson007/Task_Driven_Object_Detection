from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from modules.task_encoder.task_encoder import TASK_DESCRIPTIONS


DEFAULT_PRIOR_VALUE = 0.5
DEFAULT_NUM_CLASSES = 91


class TaskCategoryPrior:
    """Stores task-category compatibility priors for COCO-Tasks."""

    def __init__(self, prior_tensor: torch.Tensor | None = None) -> None:
        if prior_tensor is None:
            self.prior = torch.full(
                (len(TASK_DESCRIPTIONS), DEFAULT_NUM_CLASSES),
                DEFAULT_PRIOR_VALUE,
                dtype=torch.float32,
            )
        else:
            self.prior = prior_tensor.clone().detach().float()

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "TaskCategoryPrior":
        counts = torch.zeros((len(TASK_DESCRIPTIONS), DEFAULT_NUM_CLASSES), dtype=torch.float32)
        totals = torch.zeros((len(TASK_DESCRIPTIONS), DEFAULT_NUM_CLASSES), dtype=torch.float32)

        for item in dataset:
            task_index = item["task_id"] - 1
            class_ids = item["class_ids"].tolist()
            labels = item["labels"].tolist()
            for class_id, label in zip(class_ids, labels):
                totals[task_index, class_id] += 1.0
                counts[task_index, class_id] += float(label)

        prior = torch.where(
            totals > 0.0,
            (counts + 1.0) / (totals + 2.0),
            torch.zeros_like(totals),
        )
        return cls(prior_tensor=prior)

    def lookup(self, task_id: int, class_ids: torch.Tensor) -> torch.Tensor:
        """Returns a prior score for each object in the batch."""
        task_index = task_id - 1
        if task_index < 0 or task_index >= len(TASK_DESCRIPTIONS):
            raise ValueError(f"Task ID must be between 1 and {len(TASK_DESCRIPTIONS)}")

        class_ids = class_ids.long()
        return self.prior[task_index, class_ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.prior, path)

    @classmethod
    def load(cls, path: Path) -> "TaskCategoryPrior":
        prior_tensor = torch.load(path, map_location="cpu")
        return cls(prior_tensor=prior_tensor)

    @classmethod
    def from_annotations_dir(cls, annotations_dir: Path) -> "TaskCategoryPrior":
        """Builds priors from raw COCO-Tasks JSON annotation files."""
        import json

        counts = torch.zeros((len(TASK_DESCRIPTIONS), DEFAULT_NUM_CLASSES), dtype=torch.float32)
        totals = torch.zeros((len(TASK_DESCRIPTIONS), DEFAULT_NUM_CLASSES), dtype=torch.float32)

        for task_id in range(1, len(TASK_DESCRIPTIONS) + 1):
            path = annotations_dir / f"task_{task_id}_train.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for ann in data["annotations"]:
                class_id = int(ann["COCO_category_id"])
                label = float(ann["category_id"])
                totals[task_id - 1, class_id] += 1.0
                counts[task_id - 1, class_id] += label

        prior = torch.where(
            totals > 0.0,
            (counts + 1.0) / (totals + 2.0),
            torch.zeros_like(totals),
        )
        return cls(prior_tensor=prior)
