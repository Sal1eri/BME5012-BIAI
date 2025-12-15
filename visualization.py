import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib import cm

from utils.dataset import ADNIDataset, load_config
from model.ResNet import ResNet3D


class GradCAM3D:
    """Minimal Grad-CAM for 3D CNNs."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x, target_category=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if target_category is None:
            target_category = logits.argmax(dim=1)
        loss = logits.gather(1, target_category.view(-1, 1)).squeeze()
        loss.backward(torch.ones_like(loss))

        weights = self.gradients.mean(dim=(2, 3, 4))
        cam = torch.relu((weights[:, :, None, None, None] * self.activations).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2:], mode="trilinear", align_corners=False).squeeze(1)

        cams = []
        for single in cam:
            min_v, max_v = single.min(), single.max()
            cams.append((single - min_v) / (max_v - min_v + 1e-6))
        cam = torch.stack(cams)
        return cam, logits.detach()


def build_model(config, device, checkpoint_path=None):
    name = config["model"]["name"].lower()
    if name not in ["resnet", "resnet3d"]:
        raise ValueError("Only 3D ResNet is supported in this visualization script.")

    model = ResNet3D()
    ckpt = checkpoint_path or config["training"]["save_path"]
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model):
    # Use the last conv layer of the final residual block.
    return model.layer4[-1].conv2


def to_numpy(volume):
    # volume: (1, D, H, W)
    arr = volume.squeeze(0).cpu().numpy()
    return arr


def normalize_slice(slice_2d):
    min_v, max_v = slice_2d.min(), slice_2d.max()
    return (slice_2d - min_v) / (max_v - min_v + 1e-6)


def apply_colormap(heat, cmap_name="turbo"):
    cmap = cm.get_cmap(cmap_name)
    heat_rgb = cmap(heat)[:, :, :3]  # drop alpha
    return heat_rgb


def plot_cam(volume, cam, save_path, alpha=0.45, cmap_name="turbo"):
    """Plot axial, coronal, sagittal overlays with equal aspect and no resizing."""
    vol_np = to_numpy(volume)
    cam_np = cam.cpu().numpy()
    d, h, w = vol_np.shape

    axial = normalize_slice(vol_np[d // 2])
    coronal = normalize_slice(vol_np[:, h // 2, :])
    sagittal = normalize_slice(vol_np[:, :, w // 2])

    cam_axial = normalize_slice(cam_np[d // 2])
    cam_coronal = normalize_slice(cam_np[:, h // 2, :])
    cam_sagittal = normalize_slice(cam_np[:, :, w // 2])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = [(axial, cam_axial, "Axial"), (coronal, cam_coronal, "Coronal"), (sagittal, cam_sagittal, "Sagittal")]

    for ax, (img, heat, title) in zip(axes, slices):
        ax.imshow(img, cmap="gray", interpolation="nearest", aspect="equal")
        ax.imshow(heat, cmap=cmap_name, alpha=alpha, interpolation="nearest", aspect="equal")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cam_to_mask(cam, thresh=0.4):
    """Turn CAM (D,H,W) into binary mask using percentile threshold."""
    cam_np = cam.cpu().numpy()
    t = np.percentile(cam_np, thresh * 100)
    mask = (cam_np >= t).astype(np.uint8)
    return mask


def plot_montage(volume, cam, save_path, num_slices=4, cmap_name="turbo", alpha=0.45):
    """Grid of slices across depth to show 3D attention footprint."""
    vol_np = to_numpy(volume)
    cam_np = cam.cpu().numpy()
    d, h, w = vol_np.shape
    axial_idxs = np.linspace(0, d - 1, num_slices, dtype=int)
    cor_idxs = np.linspace(0, h - 1, num_slices, dtype=int)
    sag_idxs = np.linspace(0, w - 1, num_slices, dtype=int)

    rows = [("Axial", axial_idxs, lambda i: (normalize_slice(vol_np[i]), normalize_slice(cam_np[i]))),
            ("Coronal", cor_idxs, lambda i: (normalize_slice(vol_np[:, i, :]), normalize_slice(cam_np[:, i, :]))),
            ("Sagittal", sag_idxs, lambda i: (normalize_slice(vol_np[:, :, i]), normalize_slice(cam_np[:, :, i])))]

    fig, axes = plt.subplots(len(rows), num_slices, figsize=(3 * num_slices, 3 * len(rows)))
    for r, (title, idxs, getter) in enumerate(rows):
        for c, idx in enumerate(idxs):
            img, heat = getter(idx)
            ax = axes[r, c] if len(rows) > 1 else axes[c]
            ax.imshow(img, cmap="gray", interpolation="nearest", aspect="equal")
            ax.imshow(heat, cmap=cmap_name, alpha=alpha, interpolation="nearest", aspect="equal")
            ax.set_title(f"{title} {idx}")
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mip(cam, save_path, cmap_name="turbo"):
    """Save maximum intensity projections of CAM along three axes."""
    cam_np = cam.cpu().numpy()
    mips = [cam_np.max(axis=0), cam_np.max(axis=1), cam_np.max(axis=2)]
    titles = ["MIP Axial", "MIP Coronal", "MIP Sagittal"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mip, title in zip(axes, mips, titles):
        ax.imshow(normalize_slice(mip), cmap=cmap_name, interpolation="nearest", aspect="equal")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cam_nifti(cam, out_path):
    cam_np = cam.cpu().numpy().astype(np.float32)
    nib.save(nib.Nifti1Image(cam_np, affine=np.eye(4)), out_path)


def load_sample(config, split, index):
    dataset = ADNIDataset(config["model"]["name"]).load_data(split)
    image, label = dataset[index]
    return image.unsqueeze(0), int(label)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for 3D CNN AD classifier")
    parser.add_argument("--config", default=os.path.join("configs", "training", "resnet.yaml"))
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint. Defaults to config training.save_path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="Sample index inside the split dataset")
    parser.add_argument("--output", default="outputs", help="Directory to save visualization")
    parser.add_argument("--threshold", type=float, default=0.4, help="CAM percentile threshold for mask, 0-1")
    parser.add_argument("--save-nifti", action="store_true", help="Save CAM and mask as NIfTI volumes")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for CAM")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap for CAM (e.g., turbo, jet, magma)")
    parser.add_argument("--montage", action="store_true", help="Save multi-slice montage to show 3D attention spread")
    parser.add_argument("--mip", action="store_true", help="Save maximum intensity projection of CAM")
    parser.add_argument("--slices", type=int, default=4, help="Number of slices per view in montage")
    parser.add_argument("--device", default=None, help="Force device, e.g., cuda:0 or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(args.config)

    image, label = load_sample(config, args.split, args.index)
    model = build_model(config, device, args.checkpoint)

    target_layer = get_target_layer(model)
    cam = GradCAM3D(model, target_layer)

    os.makedirs(args.output, exist_ok=True)
    png_path = os.path.join(args.output, f"cam_{args.split}_{args.index}.png")

    with torch.enable_grad():
        image = image.to(device)
        cams, logits = cam.generate(image)
        pred = logits.argmax(dim=1).item()
        cam_single = cams[0]
        mask = cam_to_mask(cam_single, args.threshold)

        plot_cam(image.cpu()[0], cam_single, png_path, alpha=args.alpha, cmap_name=args.cmap)

        if args.montage:
            montage_path = os.path.join(args.output, f"cam_montage_{args.split}_{args.index}.png")
            plot_montage(image.cpu()[0], cam_single, montage_path, num_slices=args.slices, cmap_name=args.cmap, alpha=args.alpha)

        if args.mip:
            mip_path = os.path.join(args.output, f"cam_mip_{args.split}_{args.index}.png")
            save_mip(cam_single, mip_path, cmap_name=args.cmap)

        if args.save_nifti:
            cam_nii = os.path.join(args.output, f"cam_{args.split}_{args.index}.nii.gz")
            mask_nii = os.path.join(args.output, f"cam_mask_{args.split}_{args.index}.nii.gz")
            save_cam_nifti(cam_single, cam_nii)
            save_cam_nifti(torch.from_numpy(mask.astype(np.float32)), mask_nii)

    cam.remove()
    print(f"saved overlay: {png_path}")
    if args.montage:
        print("saved montage")
    if args.mip:
        print("saved MIP")
    if args.save_nifti:
        print("saved NIfTI volumes next to overlay")
    print(f"true label: {label}, pred: {pred}")


if __name__ == "__main__":
    main()
