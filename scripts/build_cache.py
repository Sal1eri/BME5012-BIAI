#!/usr/bin/env python3
"""Build compact per-split caches for 3D volumes and 2D slices."""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset import (
    ADNI2DDataset,
    ADNIDataset,
    ADNIMultiSliceDataset,
    CACHE_2D_DIR,
    CACHE_DIR,
    CACHE_MVST_DIR,
    load_config,
)


def build_3d() -> None:
    for split in ["train", "val", "test"]:
        dataset = ADNIDataset("resnet").load_data(split)
        arrays = []
        for item in tqdm(dataset.samples, desc=f"3d/{split}"):
            legacy = CACHE_DIR / f"{item['image_id']}.npy"
            arrays.append(np.load(legacy) if legacy.exists() else dataset._load_and_resize(item["path"]))

        dataset._cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(dataset._cache_file, np.stack(arrays))
        for item in dataset.samples:
            (CACHE_DIR / f"{item['image_id']}.npy").unlink(missing_ok=True)
        print(f"Saved {dataset._cache_file}")


def build_2d() -> None:
    config = load_config("configs/training/vgg16.yaml")
    root = config["data"]["data_root_dir"]
    for split in ["train", "val", "test"]:
        csv_path = config["data"][f"{split}_csv"]
        for single in [False, True]:
            mode = "single" if single else "triple"
            dataset = ADNI2DDataset(csv_path, root, single_slice_mode=single)
            extractor = dataset._extract_single_slice if single else dataset._extract_3_slices
            arrays = []
            for item in tqdm(dataset.samples, desc=f"2d/{mode}/{split}"):
                legacy = CACHE_2D_DIR / f"{item['image_id']}_{mode}.npy"
                arrays.append(np.load(legacy) if legacy.exists() else extractor(item["path"]))

            dataset._cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(dataset._cache_file, np.stack(arrays))
            for item in dataset.samples:
                (CACHE_2D_DIR / f"{item['image_id']}_{mode}.npy").unlink(missing_ok=True)
            print(f"Saved {dataset._cache_file}")


def build_mvst() -> None:
    config = load_config("configs/training/mvst.yaml")
    root = config["data"]["data_root_dir"]
    num_slices = int(config["model"].get("num_slices", 4))
    for split in ["train", "val", "test"]:
        csv_path = config["data"][f"{split}_csv"]
        dataset = ADNIMultiSliceDataset(csv_path, root, num_slices=num_slices)
        arrays = []
        for item in tqdm(dataset.samples, desc=f"mvst/{split}"):
            arrays.append(dataset._extract(item["path"]))
        dataset._cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(dataset._cache_file, np.stack(arrays))
        print(f"Saved {dataset._cache_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact image caches.")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-mvst", action="store_true")
    args = parser.parse_args()

    if not args.skip_3d:
        build_3d()
        print(f"3D cache ready: {CACHE_DIR}")
    if not args.skip_2d:
        build_2d()
        print(f"2D cache ready: {CACHE_2D_DIR}")
    if not args.skip_mvst:
        build_mvst()
        print(f"MVST cache ready: {CACHE_MVST_DIR}")


if __name__ == "__main__":
    main()
