from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from modules.dataset.coco_tasks_dataset import COCOTasksDataset
from modules.detector.detector import create_default_detector
from modules.detector.roi_features import extract_roi_features
from modules.task_encoder.task_category_prior import TaskCategoryPrior
from modules.task_encoder.task_encoder import create_default_task_encoder
from modules.reasoning_network import ReasoningNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Task-Aware Object Selection Reasoning Network.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training to a subset of samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation to a subset of samples.")
    parser.add_argument("--checkpoint-dir", type=Path, default=project_root / "models" / "checkpoints", help="Directory to save weights.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Logit threshold for selection.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run training on (e.g. cpu, cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Enable deterministic flags to reduce CPU/GPU differences (may slow training).")
    return parser.parse_args()


def simulate_confidences(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates realistic detector confidence scores for training and validation.
    - If label is 1.0 (preferred object): assign high confidence Uniform(0.7, 1.0)
    - If label is 0.0 (non-preferred object):
      - 30% chance to simulate a weak/false detection: confidence Uniform(0.1, 0.4), label forced to 0.0.
      - 70% chance to simulate a clear non-target detection: confidence Uniform(0.7, 1.0), label remains 0.0.
    """
    device = scores.device
    perturbed_scores = torch.empty_like(scores)
    perturbed_labels = labels.clone()
    for i in range(len(scores)):
        if labels[i] == 1.0:
            perturbed_scores[i] = torch.empty(1, device=device).uniform_(0.7, 1.0).item()
        else:
            if torch.rand(1).item() < 0.3:
                perturbed_scores[i] = torch.empty(1, device=device).uniform_(0.1, 0.4).item()
                perturbed_labels[i] = 0.0
            else:
                perturbed_scores[i] = torch.empty(1, device=device).uniform_(0.7, 1.0).item()
    return perturbed_scores, perturbed_labels


def evaluate(
    model: ReasoningNetwork,
    loader: DataLoader,
    detector: Any,
    encoder: Any,
    max_samples: int | None = None,
    threshold: float = 0.0,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Evaluates the model performance on validation set using:
      1. BCE Loss
      2. Top-1 Accuracy: is the chosen argmax object a preferred one (label == 1.0)?
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    total_valid_samples = 0
    
    count = 0
    with torch.no_grad():
        for item in tqdm(loader, desc="Evaluating", leave=False):
            if max_samples is not None and count >= max_samples:
                break

            # Squeeze batch dimension (DataLoader wraps items in batch size 1)
            image_path = item["image_path"][0]
            task_prompt = item["task_prompt"][0]
            task_id = int(item["task_id"][0])
            boxes = item["boxes"][0].to(device)
            class_ids = item["class_ids"][0].to(device)
            scores = item["scores"][0].to(device)
            labels = item["labels"][0].to(device)

            M = boxes.shape[0]
            if M == 0:
                continue

            scores, labels = simulate_confidences(scores, labels)

            # Load image size
            try:
                with Image.open(image_path) as img:
                    img_size = (img.height, img.width)
            except Exception:
                continue

            # Extract features (frozen models)
            try:
                roi_features = extract_roi_features(detector, image_path, boxes).to(device)
                task_raw_emb = torch.from_numpy(encoder.encode_prompt(task_prompt)).to(device)
            except Exception:
                continue

            # Forward pass through the reasoning network
            logits, best_idx, best_score = model(
                roi_features=roi_features,
                boxes=boxes,
                class_ids=class_ids,
                scores=scores,
                task_raw_emb=task_raw_emb,
                task_id=task_id,
                img_size=img_size,
                threshold=threshold,
            )

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()

            # Compute Top-1 Accuracy
            # Under threshold selection (best_idx == -1) is correct if there are no positive labels
            has_positive = (labels == 1.0).any().item()
            if best_idx == -1:
                if not has_positive:
                    correct_top1 += 1
            else:
                if labels[best_idx] == 1.0:
                    correct_top1 += 1

            total_valid_samples += 1
            count += 1

    avg_loss = total_loss / max(total_valid_samples, 1)
    top1_accuracy = correct_top1 / max(total_valid_samples, 1)
    
    return avg_loss, top1_accuracy


def main() -> int:
    args = parse_args()

    # Setup reproducibility
    import random, numpy as _np
    random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        # Make cudnn deterministic where possible (may slow training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not support this API; it's optional
            pass

    # Ensure checkpoint directory exists
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_dataset = COCOTasksDataset(data_dir=project_root / "data", split="train")
    val_dataset = COCOTasksDataset(data_dir=project_root / "data", split="test")
    
    print(f"Loaded {len(train_dataset)} training items and {len(val_dataset)} validation items.")

    print("Building task-category priors from training data...")
    prior = TaskCategoryPrior.from_dataset(train_dataset)
    prior_path = args.checkpoint_dir / "task_category_prior.pt"
    prior.save(prior_path)
    print(f"Saved task-category prior tensor to {prior_path}")

    # Batch size is 1 because of variable proposals per image.
    # For deterministic runs set num_workers=0; use pin_memory when using CUDA.
    num_workers = 0 if args.deterministic else 4
    pin_memory = True if args.device.startswith("cuda") and torch.cuda.is_available() else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Initializing YOLO detector and SentenceTransformer task encoder (frozen layers) on {args.device}...")
    detector = create_default_detector(device=args.device)
    encoder = create_default_task_encoder(device=args.device)

    print("Initializing Reasoning Network...")
    model = ReasoningNetwork(default_threshold=args.threshold, prior_tensor=prior.prior).to(args.device)
    checkpoint_path = args.checkpoint_dir / "best_model.pt"
    best_val_acc = 0.0
    if checkpoint_path.exists():
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            print(f"Loaded existing model checkpoint from {checkpoint_path}")
            print("Evaluating loaded checkpoint...")
            _, best_val_acc = evaluate(
                model,
                val_loader,
                detector,
                encoder,
                max_samples=args.max_val_samples,
                threshold=args.threshold,
                device=args.device,
            )
            print(f"Loaded model validation accuracy: {best_val_acc:.4f}")
        except RuntimeError as exc:
            print("WARNING: Failed to load existing checkpoint due to model mismatch.")
            print("This can happen when the code has changed since the checkpoint was saved.")
            print(f"Checkpoint path: {checkpoint_path}")
            print(f"Error: {exc}")
            print("Training will continue from scratch with a fresh model.")
            best_val_acc = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("\nStarting Stage A training (YOLO and MiniLM frozen)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        processed_samples = 0
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, item in enumerate(train_pbar):
            if args.max_train_samples is not None and step >= args.max_train_samples:
                break

            image_path = item["image_path"][0]
            task_prompt = item["task_prompt"][0]
            boxes = item["boxes"][0].to(args.device)
            class_ids = item["class_ids"][0].to(args.device)
            scores = item["scores"][0].to(args.device)
            labels = item["labels"][0].to(args.device)

            M = boxes.shape[0]
            if M == 0:
                continue

            scores, labels = simulate_confidences(scores, labels)

            try:
                with Image.open(image_path) as img:
                    img_size = (img.height, img.width)
            except Exception:
                continue

            # Extract features under no_grad (frozen)
            with torch.no_grad():
                try:
                    roi_features = extract_roi_features(detector, image_path, boxes).to(args.device)
                    task_raw_emb = torch.from_numpy(encoder.encode_prompt(task_prompt)).to(args.device)
                except Exception:
                    continue

            # Forward pass
            logits, best_idx, best_score = model(
                roi_features=roi_features,
                boxes=boxes,
                class_ids=class_ids,
                scores=scores,
                task_raw_emb=task_raw_emb,
                task_id=int(item["task_id"][0]),
                img_size=img_size,
                threshold=args.threshold,
            )

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            processed_samples += 1

            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{epoch_loss / processed_samples:.4f}"})

        # Run validation
        print(f"\nRunning validation for epoch {epoch}...")
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            detector,
            encoder,
            max_samples=args.max_val_samples,
            threshold=args.threshold,
            device=args.device,
        )
        print(f"Validation results: Loss = {val_loss:.4f}, Top-1 Accuracy = {val_acc:.4f}")

        # Save checkpoint if accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = args.checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best validation accuracy! Saved model weights to {checkpoint_path}")

    print("\nTraining completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
