#!/usr/bin/env python3
"""Parse training logs and draw one loss comparison plot."""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] train_loss=([0-9.]+) val_loss=([0-9.]+)"
)
LOG_RE = re.compile(r"(.+)_seed(\d+)_train\.log$")
DISPLAY_NAMES = {
    "vgg16": "VGG16",
    "resnet": "3D-ResNet",
    "hcct_vit": "HCCT-ViT",
    "vit_3d": "3D-ViT",
    "swin_3d_cnn": "Swin-CNN",
    "mvst": "MVST",
}
COLORS = {
    "vgg16": "#1f77b4",
    "resnet": "#ff7f0e",
    "hcct_vit": "#2ca02c",
    "vit_3d": "#d62728",
    "swin_3d_cnn": "#9467bd",
    "mvst": "#8c564b",
}


def parse_logs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*_train.log")):
        match = LOG_RE.match(path.name)
        if not match:
            continue
        model, seed = match.group(1), int(match.group(2))
        text = path.read_text(errors="ignore")
        for epoch, total, train_loss, val_loss in EPOCH_RE.findall(text):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, metric: str):
    grouped = frame.groupby("epoch")[metric]
    keep = grouped.count()[grouped.count() >= 3].index
    return (
        grouped.median().loc[keep],
        grouped.quantile(0.25).loc[keep],
        grouped.quantile(0.75).loc[keep],
    )


def plot_curves(frame: pd.DataFrame, output: Path, max_epoch: int) -> None:
    models = [m for m in DISPLAY_NAMES if m in set(frame["model"])]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in models:
        subset = frame[frame["model"] == model]
        color = COLORS.get(model)
        median, q25, q75 = summarize(subset, "train_loss")
        ax.plot(median.index, median.values, color=color, linewidth=2.0, label=DISPLAY_NAMES[model])
        ax.fill_between(median.index, q25.values, q75.values, color=color, alpha=0.10)

    upper = max(
        np.percentile(summarize(frame[frame["model"] == m], "train_loss")[0].values, 98)
        for m in models
    )
    ax.set_ylim(0, upper * 1.12)
    ax.set_xlim(1, max_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training loss, median over 5 seeds (epochs 1-{max_epoch}; band: IQR)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from training logs.")
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--csv", type=Path, default=Path("results/loss_history.csv"))
    parser.add_argument("--from-csv", type=Path, default=None, help="Read an existing loss CSV.")
    parser.add_argument("--max-epoch", type=int, default=25)
    parser.add_argument("--figure", type=Path, default=Path("figures/loss_curves.png"))
    args = parser.parse_args()

    if args.from_csv and args.from_csv.exists():
        frame = pd.read_csv(args.from_csv)
    else:
        frame = parse_logs(args.runs)
    frame = frame[frame["epoch"] <= args.max_epoch]
    if frame.empty:
        raise SystemExit("No epoch logs found.")

    if not args.from_csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
    plot_curves(frame, args.figure, args.max_epoch)

    best = (
        frame.loc[frame.groupby(["model", "seed"])["val_loss"].idxmin()]
        .groupby("model")["val_loss"]
        .agg(["mean", "std", "min"])
        .sort_values("mean")
    )
    print(best.round(4).to_string())
    if not args.from_csv:
        print(f"Saved: {args.csv}")
    print(f"Saved: {args.figure}")


if __name__ == "__main__":
    main()
