from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.utils.data import Dataset
from PIL import Image

TASK_DESCRIPTIONS = [
    "Step on something to reach top of a shelf",  # Task 1
    "Sit comfortably",                           # Task 2
    "Place flowers",                             # Task 3
    "Get potatoes out of fire",                  # Task 4
    "Water plant",                               # Task 5
    "Get lemon out of tea",                      # Task 6
    "Dig hole",                                  # Task 7
    "Open bottle of beer",                       # Task 8
    "Open parcel",                               # Task 9
    "Serve wine",                                # Task 10
    "Pour sugar",                                # Task 11
    "Smear butter",                              # Task 12
    "Extinguish fire",                           # Task 13
    "Pound carpet",                              # Task 14
]


class COCOTasksItem(TypedDict):
    image_path: str
    task_prompt: str
    task_id: int
    boxes: torch.Tensor       # Shape [M, 4], xyxy format (absolute coordinates)
    class_ids: torch.Tensor   # Shape [M], COCO class IDs
    scores: torch.Tensor      # Shape [M], confidence scores (1.0 for GT)
    labels: torch.Tensor      # Shape [M], target binary preference labels (0 or 1)


class COCOTasksDataset(Dataset):
    """PyTorch Dataset for loading the COCO-Tasks dataset."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        split: str = "train",
        task_ids: list[int] | None = None,
    ) -> None:
        """
        Args:
            data_dir: Root directory of the project containing annotations/images.
            split: 'train' or 'test'.
            task_ids: List of task IDs to load (1-based, e.g. [1, 2, 10]). If None, loads all 14.
        """
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "coco_tasks" / "annotations"
        self.images_root = self.data_dir / "images" / "coco_images"
        self.split = split

        if task_ids is None:
            self.task_ids = list(range(1, 15))
        else:
            self.task_ids = task_ids

        self.items: list[dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        for task_id in self.task_ids:
            task_desc = TASK_DESCRIPTIONS[task_id - 1]
            json_path = self.annotations_dir / f"task_{task_id}_{self.split}.json"

            if not json_path.exists():
                # Try fallback without task_aware_object_selection prefix if called from root
                alternative_path = Path(__file__).resolve().parents[2] / json_path
                if alternative_path.exists():
                    json_path = alternative_path
                else:
                    raise FileNotFoundError(f"Annotation file not found: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Map image_id -> image metadata
            images_map = {img["id"]: img for img in data["images"]}

            # Group annotations by image_id
            annotations_by_image: dict[int, list[dict[str, Any]]] = {}
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)

            # Create items
            for img_id, anns in annotations_by_image.items():
                img_meta = images_map.get(img_id)
                if img_meta is None:
                    continue

                file_name = img_meta["file_name"]
                
                # Determine which folder the image is in (train2014 or val2014)
                if "train2014" in file_name:
                    image_path = self.images_root / "train2014" / file_name
                elif "val2014" in file_name:
                    image_path = self.images_root / "val2014" / file_name
                else:
                    # Fallback check
                    if (self.images_root / "train2014" / file_name).exists():
                        image_path = self.images_root / "train2014" / file_name
                    else:
                        image_path = self.images_root / "val2014" / file_name

                self.items.append(
                    {
                        "image_path": image_path,
                        "task_prompt": task_desc,
                        "task_id": task_id,
                        "annotations": anns,
                        "width": img_meta["width"],
                        "height": img_meta["height"],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> COCOTasksItem:
        item_meta = self.items[idx]
        image_path = item_meta["image_path"]

        boxes_list = []
        class_ids_list = []
        labels_list = []

        for ann in item_meta["annotations"]:
            x, y, w, h = ann["bbox"]
            # Convert [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
            boxes_list.append([x, y, x + w, y + h])
            class_ids_list.append(ann["COCO_category_id"])
            # `category_id` is the COCO-Tasks preference label: 1 means this object
            # was chosen by humans as the preferred object for the current task.
            labels_list.append(ann["category_id"])

        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        class_ids = torch.tensor(class_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.float32)
        scores = torch.ones(len(boxes_list), dtype=torch.float32)

        return {
            "image_path": str(image_path),
            "task_prompt": item_meta["task_prompt"],
            "task_id": item_meta["task_id"],
            "boxes": boxes,
            "class_ids": class_ids,
            "scores": scores,
            "labels": labels,
        }
