#!/usr/bin/env python3
"""Three-view MVST Grad-CAM for one AD case (one PNG per view)."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from utils.dataset import ADNIDataset, load_config

LABELS = {0: "CN", 1: "MCI", 2: "EMCI", 3: "LMCI", 4: "AD"}
VIEW_OFFSET = {"sagittal": 0, "coronal": 1, "axial": 2}


def compute_cam(model, dataset, index, token, device):
    image, label = dataset[index]
    image = image.unsqueeze(0).to(device)
    activations, gradients = {}, {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    layer = model.backbone.layer4[-1].conv2
    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)

    image.requires_grad_(True)
    logits = model(image)
    pred = int(logits.argmax(dim=1))
    logits[0, pred].backward()
    fh.remove()
    bh.remove()

    weights = gradients["value"][token].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activations["value"][token]).sum(dim=0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(
        cam[None, None], size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0]
    base = image.detach().cpu()[0, token, 0].numpy()
    return base, cam.detach().cpu().numpy(), int(label), pred


def save_panel(base, cam, output: Path) -> None:
    fig = plt.figure(figsize=(2.4, 2.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(base, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(output, dpi=300, pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="AD three-view MVST Grad-CAM.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mvst/mvst_case.pth"))
    parser.add_argument("--index", type=int, default=19)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("assets/mvst_attention_ad.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config("configs/training/mvst.yaml")
    num_slices = int(config["model"]["num_slices"])
    dataset = ADNIDataset("mvst").load_data("test")

    model = build_model("mvst", num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    panels = [("axial", 2), ("coronal", 2), ("sagittal", 2)]
    stem = args.output.with_suffix("")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    label, pred = None, None
    for view, slice_index in panels:
        token = VIEW_OFFSET[view] * num_slices + slice_index
        base, cam, label, pred = compute_cam(model, dataset, args.index, token, device)
        path = stem.parent / f"{stem.name}_{view}.png"
        save_panel(base, cam, path)
        print(f"Saved: {path}")

    print(f"true={LABELS[label]} pred={LABELS[pred]}")


if __name__ == "__main__":
    main()
