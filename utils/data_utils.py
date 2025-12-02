#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: data_utils.py
====================================
Dataset Utilities for Pet Recognition System

Purpose:
- Split already standardized YOLO formatted data into Train/Val/Test
- Generate dataset.yaml for YOLOv8

yolo pose train data=data/yolo_pose_data/dataset.yaml model=yolov8n-pose.pt epochs=100 imgsz=640
"""

import os
import shutil
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import random
import numpy as np


def ensure_directory(path: str, description: str = "directory") -> Path:
    """Ensure directory exists, create if not exists."""
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def create_yolo_pose_dataset(
        output_dir: str = "data/yolo_pose_data",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create YOLO pose dataset by combining multiple PRE-PROCESSED datasets.

    IMPORTANT: This function assumes that source labels in 'yolo_keypoints'
    are already in the correct YOLO Pose format:
    <class_id> <cx> <cy> <w> <h> <k1_x> <k1_y> <k1_v> ... <k5_x> <k5_y> <k5_v>
    """
    random.seed(seed)
    train_ratio, val_ratio, test_ratio = split_ratios

    # Define source datasets
    # We trust yolo_utils.py has already cleaned and formatted these
    datasets = {}

    # 1. AP-10K
    if Path("data/ap-10k/yolo_keypoints").exists():
        datasets["ap10k"] = {
            "images_dir": "data/ap-10k/data",
            "labels_dir": "data/ap-10k/yolo_keypoints",
            "extensions": [".jpg", ".jpeg"]
        }

    # 2. Stanford Dogs
    if Path("data/stanford_dogs/yolo_keypoints").exists():
        datasets["stanford_dogs"] = {
            "images_dir": "data/stanford_dogs/Images",
            "labels_dir": "data/stanford_dogs/yolo_keypoints",
            "extensions": [".jpg", ".jpeg"]
        }

    # 3. Self Collected (Includes Animal Pose & Placeholders)
    # Check both potential locations just in case
    sc_labels_path = Path("data/Self_collected_Images/yolo_keypoints")
    if sc_labels_path.exists():
        datasets["self_collected"] = {
            "images_dir": "data/Self_collected_Images",
            "labels_dir": str(sc_labels_path),
            "extensions": [".jpg", ".jpeg", ".png"]
        }

    if not datasets:
        print("❌ No datasets found! Please run utils/yolo_utils.py first.")
        return {}

    # Create output directory structure
    output_path = ensure_directory(output_dir, "YOLO pose dataset root")

    # Clean output dir to avoid mixing old data
    if output_path.exists():
        # shutil.rmtree(output_path) # Optional: Careful with delete
        pass

    image_dirs = {
        "train": ensure_directory(output_path / "images" / "train"),
        "val": ensure_directory(output_path / "images" / "val"),
        "test": ensure_directory(output_path / "images" / "test")
    }

    label_dirs = {
        "train": ensure_directory(output_path / "labels" / "train"),
        "val": ensure_directory(output_path / "labels" / "val"),
        "test": ensure_directory(output_path / "labels" / "test")
    }

    # Collect all valid image-label pairs
    all_samples = []

    print("\n📦 Collecting samples...")
    for dataset_name, config in datasets.items():
        images_dir = Path(config["images_dir"])
        labels_dir = Path(config["labels_dir"])

        # Gather images (handling subdirectories recursively)
        found_images = []
        for ext in config["extensions"]:
            found_images.extend(images_dir.rglob(f"*{ext}"))
            # Also handle upper case extensions if needed, usually glob is case sensitive on Linux
            found_images.extend(images_dir.rglob(f"*{ext.upper()}"))

        valid_count = 0
        for img_path in found_images:
            # Match label file by stem
            label_path = labels_dir / f"{img_path.stem}.txt"

            # Basic validation: Label must exist and not be empty
            if label_path.exists() and label_path.stat().st_size > 0:
                all_samples.append({
                    "image_path": img_path,
                    "label_path": label_path,
                    "dataset": dataset_name
                })
                valid_count += 1

        print(f"   🔹 {dataset_name}: {valid_count} pairs found.")

    # Shuffle and Split
    total = len(all_samples)
    if total == 0:
        print("❌ Total samples is 0. Check your paths.")
        return {}

    random.shuffle(all_samples)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:]
    }

    print(f"\n✂️  Splitting data: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # Copy files
    for split_name, samples in splits.items():
        print(f"   🚀 Generating {split_name} set...")
        for sample in samples:
            # Define destination paths
            # Use strict name to avoid collision? Assuming stems are unique enough or folders separate them.
            # If collision is a worry, prefix with dataset name.
            # file_name = f"{sample['dataset']}_{sample['image_path'].name}"
            file_name = sample['image_path'].name

            src_img = sample['image_path']
            dst_img = image_dirs[split_name] / file_name

            src_lbl = sample['label_path']
            dst_lbl = label_dirs[split_name] / f"{src_img.stem}.txt"

            # Copy files
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Simply copy the label, DO NOT convert again
            if not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    # Create dataset.yaml
    yaml_config = {
        "path": str(output_path.absolute()),  # Use absolute path to be safe
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "cat",
            1: "dog"
        },
        # Keypoint definition
        "kpt_shape": [5, 3],
        "flip_idx": [0, 2, 1, 4, 3]  # Nose, R_Eye, L_Eye, R_Ear, L_Ear (Symmetric flip)
    }

    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    print(f"\n✅ Ready for training! Config saved to: {yaml_path}")
    return {}


# --- Retain classification utils just in case, but they are separate ---
def get_num_classes(dataset_name: str, project_root: Path = None) -> int:
    # ... (Keep original logic or simple return if not used)
    return 2


def get_class_names(dataset_name: str, project_root: Path = None) -> Dict[int, str]:
    return {0: 'cat', 1: 'dog'}


if __name__ == "__main__":
    # Test run
    create_yolo_pose_dataset()