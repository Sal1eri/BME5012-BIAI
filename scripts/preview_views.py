import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from utils.dataset import ADNIDataset, load_config
from models.vgg16 import VGG16ForAD
from models.resnet import ResNet3D
import numpy as np


# ==========================
# Build model from config
# ==========================
def build_model(name):
    name = name.lower()
    if name == "vgg16":
        return VGG16ForAD()
    elif name in ["resnet", "resnet3d"]:
        return ResNet3D()
    else:
        raise ValueError("Unknown model: " + name)



# ==========================
# Build dataset from config
# ==========================
def build_dataset(config, split="test"):

    ds = ADNIDataset(config["model"]["name"])
    return ds.load_data(split)



# ==========================
# Load model weights
# ==========================
def load_weights(model, config):
    path = config["training"]["save_path"]
    print("loading:", path)

    model.load_state_dict(
        torch.load(path, map_location="cpu")
    )
    model.eval()
    return model



# ==========================
# Save stacked three-view image
# ==========================
def save_triple_view(img, save_path):
    """img: (3,H,W)"""
    vutils.save_image(img.unsqueeze(0), save_path, normalize=True)
    print("saved:", save_path)



# ==========================
# Save individual slices
# ==========================
def save_each_view(img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = ["axial","coronal","sagittal"]

    for i in range(3):
        p = os.path.join(out_dir, names[i] + ".png")
        vutils.save_image(img[i].unsqueeze(0), p, normalize=True)
        print("saved:", p)



# ==========================
# Visualize one sample
# ==========================
def visualize_sample(dataset, idx, model=None, device="cpu"):

    img, label = dataset[idx]
    print("true label =", label)

    # Predict
    if model is not None:
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = torch.argmax(logits, dim=1).item()
        print("pred =", pred)

    save_triple_view(img, "triple.png")
    save_each_view(img, "views/")



# ==========================
# main
# ==========================
if __name__ == "__main__":

    model_name = "resnet"
    cfg_path = os.path.join("configs","training",model_name+".yaml")

    config = load_config(cfg_path)

    # Data
    dataset = build_dataset(config, "test")

    # Model
    model = build_model(config["model"]["name"])
    model = load_weights(model, config)

    # Visualize
    visualize_sample(dataset, idx=0, model=model)
