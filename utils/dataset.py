from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
import yaml
import glob
import nibabel as nib
import numpy as np
from torchvision import transforms
import cv2

import torchvision.utils as vutils

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "3d"
CACHE_2D_DIR = Path(__file__).resolve().parents[1] / "cache" / "2d"
CACHE_MVST_DIR = Path(__file__).resolve().parents[1] / "cache" / "mvst"

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "EMCI": 2,
    "LMCI": 3,
    "AD": 4
}
def find_nii_path(root_dir, subject_id, image_id):
    """Find the NIfTI file for a subject/image ID."""
    subject_dir = os.path.join(root_dir, subject_id)
    if not os.path.exists(subject_dir):
        return None
    patterns = [
        os.path.join(subject_dir, "**", f"*{image_id}*.nii"),
        os.path.join(subject_dir, "**", f"*{image_id}*.nii.gz")
    ]
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        if files:
            return files[0]
    return None

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ADNI3DDataset(Dataset):
    def __init__(self, csv_file, root_dir, resize=(32, 112, 112), transform=None):
        """
        Args:
            csv_file: CSV path
            root_dir: MRI root directory
            resize: Target (D, H, W)
            transform: Optional torchvision transform
        """
        self.resize = resize
        self.transform = transform
        self.samples = []
        self._mem_cache = {}
        self._cache_file = CACHE_DIR / f"{Path(csv_file).stem}.npy"
        self._cache = None

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            group = row['Group']
            if group not in LABEL_MAP:
                continue
            path = find_nii_path(root_dir, row['Subject'], row['Image Data ID'])
            if path:
                self.samples.append({
                    'path': path,
                    'label': LABEL_MAP[group],
                    'image_id': str(row['Image Data ID']),
                })

        if self._cache_file.exists():
            cache = np.load(self._cache_file, mmap_mode="r")
            if len(cache) == len(self.samples):
                self._cache = cache

    def __len__(self):
        return len(self.samples)

    def _load_and_resize(self, nii_path):
        # Load NIfTI
        img = nib.load(nii_path)
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata(dtype=np.float32)

        # Percentile clip
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)

        # Resize depth (D)
        D, H, W = self.resize
        z = np.linspace(0, volume.shape[2]-1, D).astype(np.int32)
        volume = volume[:, :, z]

        # Resize H/W
        resized_slices = []
        for i in range(volume.shape[2]):
            slice_resized = cv2.resize(volume[:, :, i], (W, H), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(slice_resized)
        volume_resized = np.stack(resized_slices, axis=2)  # (H, W, D)

        # Normalize
        volume_resized = (volume_resized - volume_resized.mean()) / (volume_resized.std() + 1e-5)

        # (C, D, H, W)
        volume_resized = volume_resized[np.newaxis, ...]  # (1, H, W, D)
        volume_resized = np.transpose(volume_resized, (0, 3, 1, 2))  # (1, D, H, W)
        return volume_resized.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]

        if idx not in self._mem_cache:
            if self._cache is not None:
                volume = np.array(self._cache[idx])
            else:
                volume = self._load_and_resize(item['path'])
            self._mem_cache[idx] = torch.from_numpy(volume)

        volume = self._mem_cache[idx]
        if self.transform:
            volume = self.transform(volume)

        label = torch.tensor(item['label'], dtype=torch.long)
        return volume, label
    
class ADNI2DDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, single_slice_mode=False):
        """
        Args:
            csv_file (str): CSV split path
            root_dir (str): ADNI root directory
        """
        self.transform = transform
        self.single_slice_mode = single_slice_mode
        self.samples = []
        self._mem_cache = {}
        mode = "single" if single_slice_mode else "triple"
        self._cache_file = CACHE_2D_DIR / f"{Path(csv_file).stem}_{mode}.npy"
        self._cache = None
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            group = row['Group']
            if group not in LABEL_MAP:
                continue
            path = find_nii_path(root_dir, row['Subject'], row['Image Data ID'])
            if path:
                self.samples.append({
                    'path': path,
                    'label': LABEL_MAP[group],
                    'image_id': str(row['Image Data ID']),
                })

        if self._cache_file.exists():
            cache = np.load(self._cache_file, mmap_mode="r")
            if len(cache) == len(self.samples):
                self._cache = cache

    def __len__(self):
        return len(self.samples)
    def _extract_single_slice(self, nii_path):
        img = nib.load(nii_path)
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata(dtype=np.float32)

        z_mid = volume.shape[2] // 2
        axial = volume[:, :, z_mid]

        # Percentile clip
        p1, p99 = np.percentile(axial, (1, 99))
        axial = np.clip(axial, p1, p99)

        # Resize
        axial_resized = cv2.resize(axial, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize
        axial_resized = (axial_resized - axial_resized.mean()) / (axial_resized.std() + 1e-5)

        # 3 channels
        axial_resized = np.stack([axial_resized]*3, axis=0)  # (3,H,W)
        return axial_resized.astype(np.float32)
    def _extract_3_slices(self, nii_path):
        img = nib.load(nii_path)
        img = nib.as_closest_canonical(img)

        volume = img.get_fdata(dtype=np.float32)

        x_mid = volume.shape[0] // 2
        y_mid = volume.shape[1] // 2
        z_mid = volume.shape[2] // 2

        sagittal = volume[x_mid, :, :]      # (Y,Z)
        coronal  = volume[:, y_mid, :]      # (X,Z)
        axial    = volume[:, :, z_mid]      # (X,Y)

        # ---- 1. percent clip ----
        all_slices = [axial, coronal, sagittal]
        p1, p99 = np.percentile(np.concatenate([s.flatten() for s in all_slices]), (1, 99))
        slices = [np.clip(s, p1, p99) for s in all_slices]

        # ---- 2. Resize to 224x224 ----
        resized = []
        for s in slices:
            s = cv2.resize(s, (224, 224), interpolation=cv2.INTER_LINEAR)
            resized.append(s)

        resized = np.stack(resized, axis=0)  # (3, 224, 224)

        # ---- 3. normalize ----
        resized = (resized - resized.mean()) / (resized.std() + 1e-5)

        return resized.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mode = "single" if self.single_slice_mode else "triple"

        if idx not in self._mem_cache:
            if self._cache is not None:
                img = np.array(self._cache[idx])
            else:
                extractor = (
                    self._extract_single_slice
                    if self.single_slice_mode
                    else self._extract_3_slices
                )
                img = extractor(item["path"])
            self._mem_cache[idx] = torch.from_numpy(img)

        img = self._mem_cache[idx]  # (3, H, W)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(item["label"], dtype=torch.long)

        return img, label


class ADNIMultiSliceDataset(Dataset):
    """Multi-view slice dataset for the transformer model."""

    def __init__(self, csv_file, root_dir, num_slices=4, image_size=224, transform=None):
        self.num_slices = num_slices
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        self._cache_file = CACHE_MVST_DIR / f"{Path(csv_file).stem}_{num_slices}.npy"
        self._cache = None

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            group = row["Group"]
            if group not in LABEL_MAP:
                continue
            path = find_nii_path(root_dir, row["Subject"], row["Image Data ID"])
            if path:
                self.samples.append(
                    {
                        "path": path,
                        "label": LABEL_MAP[group],
                        "image_id": str(row["Image Data ID"]),
                    }
                )

        if self._cache_file.exists():
            cache = np.load(self._cache_file, mmap_mode="r")
            if len(cache) == len(self.samples):
                self._cache = cache

    def __len__(self):
        return len(self.samples)

    def _extract(self, nii_path):
        img = nib.as_closest_canonical(nib.load(nii_path))
        volume = img.get_fdata(dtype=np.float32)
        slices = []

        for axis in range(3):
            positions = np.linspace(0, volume.shape[axis] - 1, self.num_slices).astype(int)
            for index in positions:
                if axis == 0:
                    image = volume[index, :, :]
                elif axis == 1:
                    image = volume[:, index, :]
                else:
                    image = volume[:, :, index]

                p1, p99 = np.percentile(image, (1, 99))
                image = np.clip(image, p1, p99)
                image = cv2.resize(
                    image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
                )
                image = (image - image.mean()) / (image.std() + 1e-5)
                slices.append(np.stack([image] * 3, axis=0))

        return np.stack(slices).astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self._cache is not None:
            volume = np.array(self._cache[idx])
        else:
            volume = self._extract(item["path"])

        volume = torch.from_numpy(volume)
        if self.transform:
            volume = self.transform(volume)
        return volume, torch.tensor(item["label"], dtype=torch.long)


class ADNIDataset:
    def __init__(self,model_name):
        self.model_name = model_name
        cfg_pth = os.path.join("configs", "training", f"{model_name}.yaml")
        self.config = load_config(cfg_pth)
        self._loader_map = {
            "vgg16": self._load_vgg16_data,
            "resnet": self._load_resnet_data,
            "hcct_vit": self._load_resnet_data,
            "vit_3d": self._load_resnet_data,
            "swin_3d_cnn": self._load_resnet_data,
            "mvst": self._load_mvst_data,
        }
    def load_data(self, split='train'):
        return self._loader_map[self.model_name](split)
    
    def _load_resnet_data(self, split='train'):
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"

        csv_path = self.config['data'][f"{split}_csv"]
        root_dir = self.config['data']['data_root_dir']
        dataset = ADNI3DDataset(csv_file=csv_path, root_dir=root_dir)
        return dataset
    def _load_mvst_data(self, split='train'):
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        csv_path = self.config['data'][f"{split}_csv"]
        root_dir = self.config['data']['data_root_dir']
        num_slices = int(self.config['model'].get('num_slices', 4))
        return ADNIMultiSliceDataset(
            csv_file=csv_path,
            root_dir=root_dir,
            num_slices=num_slices,
        )

    def _load_vgg16_data(self, split='train'):
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        csv_path = self.config['data'][f"{split}_csv"]
        root_dir = self.config['data']['data_root_dir']
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        dataset = ADNI2DDataset(csv_file=csv_path, root_dir=root_dir, transform=transform)
        return dataset


# -------------------------
# Visualization helpers
# -------------------------
def save_example_view(dataset, idx=0, save_path="view_sample.png"):
    """Save the stacked three-view image."""
    img, label = dataset[idx]    # img: (3,224,224)

    # (3, H, W) -> (1, 3, H, W)
    img = img.unsqueeze(0)

    vutils.save_image(img, save_path, normalize=True)
    print(f"Saved: {save_path}, label={label}")


def save_each_slice(dataset, idx=0, out_dir="slices_view"):
    """Save axial, coronal, and sagittal slices separately."""
    os.makedirs(out_dir, exist_ok=True)

    img, _ = dataset[idx]    # img: (3,224,224)
    names = ["axial", "coronal", "sagittal"]

    for i in range(3):
        out = os.path.join(out_dir, f"{names[i]}.png")
        vutils.save_image(img[i].unsqueeze(0), out, normalize=True)
        print(f"Saved: {out}")

if __name__ == "__main__":
    dataset = ADNIDataset("vgg16").load_data("val")

    save_example_view(dataset, 0, "assets/slice_views/view_sample.png")
    save_each_slice(dataset, 0, "assets/slice_views")