#!/usr/bin/env python3
"""Evaluate an AD classifier.

Examples:
    python test.py --model resnet --split test
    python test.py --model vgg16 --checkpoint checkpoints/vgg16/vgg16_best.pth
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.dataset import ADNIDataset, load_config
from utils.metrics import get_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AD classifier.")
    parser.add_argument(
        "--model",
        default="resnet",
        choices=["resnet", "vgg16", "hcct_vit", "vit_3d", "swin_3d_cnn", "mvst"],
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model weights path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    config_path = args.config or Path("configs") / "training" / f"{args.model}.yaml"
    config = load_config(str(config_path))
    config["model"]["name"] = args.model

    device = torch.device(args.device)
    dataset = ADNIDataset(config["model"]["name"]).load_data(args.split)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        config["model"]["name"],
        num_classes=config["model"].get("num_classes", 5),
        in_channels=config["model"].get("in_channels"),
    ).to(device)

    checkpoint = args.checkpoint or Path(config["training"]["save_path"])
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)

            probs = torch.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(dataset)
    metrics = get_classification_metrics(all_labels, all_preds, all_probs)

    print(f"Checkpoint: {checkpoint}")
    print(f"Split: {args.split} ({len(dataset)} samples)")
    print(f"Loss: {loss:.4f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
