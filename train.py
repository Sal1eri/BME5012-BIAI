#!/usr/bin/env python3
"""Train an AD classifier.

Examples:
    python train.py --model resnet --epochs 50
    python train.py --model vgg16 --batch-size 16 --lr 1e-4
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.dataset import ADNIDataset, load_config
from utils.metrics import get_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AD classifier.")
    parser.add_argument(
        "--model",
        default="resnet",
        choices=["resnet", "vgg16", "hcct_vit", "vit_3d", "swin_3d_cnn", "mvst"],
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--weighted-loss", action="store_true", help="Use class-weighted cross entropy.")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--monitor", choices=["val_loss", "val_f1"], default="val_f1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_training_config(args: argparse.Namespace) -> dict:
    config_path = args.config or Path("configs") / "training" / f"{args.model}.yaml"
    config = load_config(str(config_path))
    config["model"]["name"] = args.model
    training = config["training"]

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "save_path": args.save_path,
    }
    for key, value in overrides.items():
        if value is not None:
            training[key] = value

    optimizer = training["optimizer"]
    if args.lr is not None:
        optimizer["lr"] = args.lr
    if args.weight_decay is not None:
        optimizer["weight_decay"] = args.weight_decay
    if args.patience is not None:
        training["early_stopping"]["patience"] = args.patience

    return config


def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = get_classification_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    config = load_training_config(args)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    model_cfg = config["model"]
    dataset = ADNIDataset(model_cfg["name"])
    train_dataset = dataset.load_data("train")
    val_dataset = dataset.load_data("val")

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_cfg["name"],
        num_classes=model_cfg.get("num_classes", 5),
        in_channels=model_cfg.get("in_channels"),
    ).to(device)

    if args.weighted_loss:
        labels = [sample["label"] for sample in train_dataset.samples]
        counts = np.bincount(labels, minlength=model_cfg.get("num_classes", 5))
        weights = len(labels) / (len(counts) * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping"]["patience"])
    save_path = Path(config["training"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = float("-inf") if args.monitor == "val_f1" else float("inf")
    stale_epochs = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, use_amp)
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={metrics['Accuracy']:.4f} macro_f1={metrics['Macro-F1']:.4f}"
        )

        if args.monitor == "val_f1":
            score = metrics["Macro-F1"]
            improved = score > best_score
        else:
            score = val_loss
            improved = score < best_score

        if improved:
            best_score = score
            stale_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path} ({args.monitor}={score:.4f})")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
