from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont

import torch

from modules.detector.detector import create_default_detector
from modules.detector.roi_features import extract_roi_features
from modules.task_encoder.task_category_prior import TaskCategoryPrior
from modules.task_encoder.task_encoder import create_default_task_encoder
from modules.reasoning_network import ReasoningNetwork


COCO_NAME_TO_ID = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
    'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
    'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
    'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44,
    'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55,
    'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79,
    'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run task-aware object selection to find the best object for a prompt in an image."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("prompt", type=str, help="Text prompt describing the task.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to a custom YOLO model checkpoint.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image size used by the detector.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the models on (e.g. cpu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON output. If not set, prints to stdout.",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=None,
        help="Optional SentenceTransformer model name for text encoding.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "checkpoints" / "best_model.pt",
        help="Path to trained reasoning network checkpoints.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Relevance score threshold for selection.",
    )
    parser.add_argument(
        "--save-visual",
        action="store_true",
        default=True,
        help="Whether to save the annotated visual result image in outputs/.",
    )
    return parser.parse_args()


def build_detector(args: argparse.Namespace):
    if args.model_path is not None:
        from modules.detector.detector import YOLODetector

        return YOLODetector(
            model_path=args.model_path,
            confidence_threshold=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )

    return create_default_detector(
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        device=args.device,
    )


def draw_and_save_result(
    image_path: Path,
    selected_object: dict[str, Any] | None,
    detections: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Draws visual annotations on the image and saves the file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a font, otherwise fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw all detected object proposals in red
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        draw.text((x1 + 2, y1 + 2), f"{det['class_name']}", fill="red", font=font)

    # Draw the chosen task-aware object in thick green
    if selected_object is not None:
        x1, y1, x2, y2 = selected_object["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
        draw.text(
            (x1 + 4, y1 + 4),
            f"SELECTED ({selected_object['class_name']}): {selected_object['relevance_score']:.2f}",
            fill="green",
            font=font,
        )

    # Save the output file
    out_name = f"{image_path.stem}_result{image_path.suffix}"
    out_path = output_dir / out_name
    img.save(out_path)
    return out_path


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # 1. Initialize Detector and Task Encoder
    detector = build_detector(args)
    encoder = create_default_task_encoder(model_name=args.text_model, device=args.device)

    # 2. Match text prompt to nearest COCO-Tasks description and get text embedding
    match = encoder.match_prompt_to_task(args.prompt)
    task_raw_emb = torch.from_numpy(match.prompt_embedding).float().to(args.device)

    # 3. Detect candidate object proposals
    detections = detector.detect(args.image)

    selected_object = None
    best_score = float("-inf")
    best_idx = -1
    logits = []

    if len(detections) > 0:
        # Construct tensors for reasoning network
        boxes_list = [det.bbox_xyxy for det in detections]
        class_ids_list = [COCO_NAME_TO_ID.get(det.class_name, det.class_id) for det in detections]
        scores_list = [det.confidence for det in detections]

        boxes = torch.tensor(boxes_list, dtype=torch.float32).to(args.device)
        class_ids = torch.tensor(class_ids_list, dtype=torch.long).to(args.device)
        scores = torch.tensor(scores_list, dtype=torch.float32).to(args.device)

        # Extract neck visual features for RoIs
        roi_features = extract_roi_features(detector, args.image, boxes).to(args.device)

        # Image size (height, width)
        with Image.open(args.image) as img:
            img_size = (img.height, img.width)

        # 4. Load Reasoning Network
        prior_path = Path(__file__).resolve().parent / "models" / "task_category_prior.pt"
        prior_tensor = None
        if prior_path.exists():
            prior_tensor = TaskCategoryPrior.load(prior_path).prior
            print(f"Loaded task-category priors from {prior_path}", file=sys.stderr)

        model = ReasoningNetwork(default_threshold=args.threshold, prior_tensor=prior_tensor).to(args.device)
        if args.checkpoint.exists():
            model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
            print(f"Loaded reasoning network weights from {args.checkpoint}", file=sys.stderr)
        else:
            print(
                f"Warning: Reasoning network checkpoint not found at {args.checkpoint}. Running with random weights.",
                file=sys.stderr,
            )

        model.eval()

        # 5. Run inference
        with torch.no_grad():
            logits_tensor, best_idx, best_score = model(
                roi_features=roi_features,
                boxes=boxes,
                class_ids=class_ids,
                scores=scores,  # Pass the raw detector confidence scores
                task_raw_emb=task_raw_emb,
                task_id=match.task_id,
                img_size=img_size,
                threshold=args.threshold,
            )
            logits = logits_tensor.tolist()

        if best_idx != -1:
            selected_object = detections[best_idx].as_dict()
            selected_object["relevance_score"] = best_score
            selected_object["logit"] = logits[best_idx]

    # Convert detections to standard dict representations for output
    detections_dicts = [det.as_dict() for det in detections]
    for idx, d in enumerate(detections_dicts):
        if len(logits) > idx:
            d["logit"] = logits[idx]

    output_data: dict[str, Any] = {
        "image": str(args.image),
        "prompt": args.prompt,
        "selected_task_id": match.task_id,
        "selected_task_description": match.task_description,
        "task_similarity": match.similarity,
        "selected_object": selected_object,
        "detections": detections_dicts,
    }

    # Generate and save visual annotated image
    if args.save_visual and len(detections) > 0:
        output_dir = Path(__file__).resolve().parent / "outputs"
        visual_path = draw_and_save_result(
            args.image, selected_object, detections_dicts, output_dir
        )
        print(f"Visual annotated result saved to: {visual_path}", file=sys.stderr)

    output_json = json.dumps(output_data, indent=2)

    if args.output is not None:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Saved JSON output to {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
