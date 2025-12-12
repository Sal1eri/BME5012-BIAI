import os
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from utils.dataset import ADNIDataset, load_config
from model.vgg16 import VGG16ForAD
from model.ResNet import ResNet3D
import numpy as np


# ==========================
# 按配置构建模型
# ==========================
def build_model(name):
    name = name.lower()
    if name == "vgg16":
        return VGG16ForAD()
    elif name in ["resnet", "resnet3d"]:
        return ResNet3D()
    else:
        raise ValueError("未知模型: " + name)



# ==========================
# 按配置构建数据集
# ==========================
def build_dataset(config, split="test"):

    ds = ADNIDataset(config["model"]["name"])
    return ds.load_data(split)



# ==========================
# 记载模型权重
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
# 保存三视图合成图
# ==========================
def save_triple_view(img, save_path):
    """img: (3,H,W)"""
    vutils.save_image(img.unsqueeze(0), save_path, normalize=True)
    print("saved:", save_path)



# ==========================
# 保存单独 slice
# ==========================
def save_each_view(img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = ["axial","coronal","sagittal"]

    for i in range(3):
        p = os.path.join(out_dir, names[i] + ".png")
        vutils.save_image(img[i].unsqueeze(0), p, normalize=True)
        print("saved:", p)



# ==========================
# 可视化一个样本
# ==========================
def visualize_sample(dataset, idx, model=None, device="cpu"):

    img, label = dataset[idx]
    print("true label =", label)

    # 预测
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

    model_name = "resnet"      # 或 resnet
    cfg_path = os.path.join("configs","training",model_name+".yaml")

    config = load_config(cfg_path)

    # 数据
    dataset = build_dataset(config, "test")

    # 模型
    model = build_model(config["model"]["name"])
    model = load_weights(model, config)

    # 可视化
    visualize_sample(dataset, idx=0, model=model)
