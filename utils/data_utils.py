#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: data_utils.py
====================================
Dataset Utilities for Pet Recognition System

Purpose:
- Comprehensive dataset management and preprocessing utilities
- Support for pet classification and detection model training
- Integration with YOLOv8 and custom PetNet architectures
- Multi-dataset handling (Oxford Pets, Stanford Dogs, custom collections)

Key Features:
1. Directory Management: Automated folder creation and validation
2. Data Partitioning: Train/val/test split with configurable ratios
3. Configuration Handling: YAML-based dataset configuration processing
4. Class Information: Dynamic class count and name mapping retrieval
5. Cross-dataset Support: Unified interface for multiple pet datasets

Dependencies:
- PyYAML for configuration parsing
- Pathlib for cross-platform path handling
- Standard library utilities for file operations
"""

import os
import shutil
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import random
import numpy as np

def get_num_classes(dataset_name: str, project_root: Path = None) -> int:
    """
    Get the number of classes from dataset.yaml configuration.

    Args:
        dataset_name: Dataset name (e.g., 'merged_cls_dataset')
        project_root: Project root directory path

    Returns:
        Number of classes

    Raises:
        FileNotFoundError: If dataset configuration file does not exist
        ValueError: If 'nc' field is missing in configuration
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    dataset_yaml = project_root / "data" / dataset_name / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {dataset_yaml}")

    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)

    if 'nc' not in config:
        raise ValueError(f"Missing 'nc' (number of classes) field in configuration")

    return config['nc']

def get_class_names(dataset_name: str, project_root: Path = None) -> Dict[int, str]:
    """
    Get class names mapping from dataset configuration.

    Args:
        dataset_name: Dataset name
        project_root: Project root directory path

    Returns:
        Dictionary: {class_index: class_name}

    Raises:
        FileNotFoundError: If dataset configuration file does not exist
        ValueError: If 'names' field is missing or invalid in configuration
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    dataset_yaml = project_root / "data" / dataset_name / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {dataset_yaml}")

    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)

    if 'names' not in config or not isinstance(config['names'], list):
        raise ValueError(f"Missing valid 'names' list in configuration")

    return {i: name for i, name in enumerate(config['names'])}

def ensure_directory(path: str, description: str = "directory") -> Path:
    """
    Ensure directory exists, create if not exists.

    Args:
        path: Directory path
        description: Directory description (for logging)

    Returns:
        Path object of the directory
    """
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {description}: {path}")
    else:
        print(f"⏩ {description} already exists: {path}")
    return dir_path

def split_classification_data(
    source_dir: str = "data/merged_cls_dataset",
    target_dir: str = "data/pet_cls_training",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split classification data into train/val/test directories by ratio.

    Args:
        source_dir: Source directory containing subdirectories by class
        target_dir: Target directory where train/val/test subdirectories will be created
        train_ratio: Training set ratio (0.0-1.0)
        val_ratio: Validation set ratio (0.0-1.0)
        test_ratio: Test set ratio (0.0-1.0)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Create target directory structure
    target_path = ensure_directory(target_dir, "target directory")
    for split in ['train', 'val', 'test']:
        ensure_directory(target_path / split, f"{split} split directory")

    # Iterate through each class in source directory
    for class_dir in Path(source_dir).iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        # Get all image files for this class
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(image_files)

        # Calculate split points
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        # Split files
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }

        # Copy files to target directory
        for split, files in splits.items():
            split_dir = ensure_directory(target_path / split / class_name,
                                       f"{split}/{class_name} class directory")

            for src_file in files:
                dst_file = split_dir / src_file.name
                if not dst_file.exists():
                    shutil.copy(src_file, dst_file)
                    print(f"  Copied: {src_file.name} -> {split_dir}")

    # Copy dataset.yaml configuration file to target directory
    source_config = Path(source_dir) / "dataset.yaml"
    target_config = target_path / "dataset.yaml"
    if source_config.exists() and not target_config.exists():
        shutil.copy(source_config, target_config)
        print(f"✅ Copied config file: {source_config} -> {target_config}")

    print("\n✅ Data splitting completed!")

def prepare_dataset_from_config(config_path: str = "configs/petnet_base.yaml"):
    """
    Prepare dataset based on configuration file.

    Args:
        config_path: Configuration file path
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    split_classification_data(
        train_ratio=data_config['train_split'],
        val_ratio=data_config['val_split'],
        test_ratio=data_config['test_split']
    )

def create_yolo_pose_dataset(
    output_dir: str = "data/yolo_pose_data",
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create YOLO pose dataset by combining multiple datasets for YOLOv8m-pose training.

    Args:
        output_dir: Output directory for the combined dataset
        split_ratios: (train_ratio, val_ratio, test_ratio)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with image paths for each split
    """
    random.seed(seed)
    train_ratio, val_ratio, test_ratio = split_ratios

    # Define source datasets and their paths (only include existing directories)
    datasets = {}

    # Check and include AP-10K if exists (only cat and dog categories)
    ap10k_images = Path("data/ap-10k/data")
    ap10k_labels = Path("data/ap-10k/yolo_keypoints")
    if ap10k_images.exists() and ap10k_labels.exists():
        # AP-10K Cat/Dog Category IDs
        cat_dog_category_ids = {8, 23, 24}  # dog:8, bobcat:23, cat:24

        datasets["ap10k"] = {
            "images_dir": "data/ap-10k/data",
            "labels_dir": "data/ap-10k/yolo_keypoints",
            "extensions": [".jpg", ".JPG", ".jpeg", ".JPEG"],
            "filter_categories": cat_dog_category_ids  # Include these categories only
        }
        print("✅ Included AP-10K dataset (cat and dog only)")
    else:
        print("⚠️  AP-10K dataset not found, skipping")

    # Check and include Stanford Dogs
    stanford_images = Path("data/stanford_dogs/Images")
    stanford_labels = Path("data/stanford_dogs/yolo_keypoints")
    if stanford_images.exists() and stanford_labels.exists():
        datasets["stanford_dogs"] = {
            "images_dir": "data/stanford_dogs/Images",
            "labels_dir": "data/stanford_dogs/yolo_keypoints",
            "extensions": [".jpg", ".JPG", ".jpeg", ".JPEG"]
        }
        print("✅ Included Stanford Dogs dataset")
    else:
        print("⚠️  Stanford Dogs dataset not found, skipping")

    # Check and include Self Collected
    self_images = Path("data/Self_collected_Images")
    self_labels = Path("data/Self_collected_Images/yolo_keypoints")
    if self_images.exists() and self_labels.exists():
        datasets["self_collected"] = {
            "images_dir": "data/Self_collected_Images",
            "labels_dir": "data/Self_collected_Images/yolo_keypoints",
            "extensions": [".jpeg", ".jpg", ".JPG", ".JPEG"]
        }
        print("✅ Included Self Collected dataset")
    else:
        print("⚠️  Self Collected dataset not found, skipping")

    if not datasets:
        print("❌ No datasets found! Please check the dataset directories.")
        return {}

    # Create output directory structure
    output_path = ensure_directory(output_dir, "YOLO pose dataset root")
    image_dirs = {
        "train": ensure_directory(output_path / "images" / "train", "train images"),
        "val": ensure_directory(output_path / "images" / "val", "validation images"),
        "test": ensure_directory(output_path / "images" / "test", "test images")
    }

    label_dirs = {
        "train": ensure_directory(output_path / "labels" / "train", "train labels"),
        "val": ensure_directory(output_path / "labels" / "val", "validation labels"),
        "test": ensure_directory(output_path / "labels" / "test", "test labels")
    }

    # Collect all image-label pairs from all datasets
    all_samples = []

    for dataset_name, config in datasets.items():
        images_dir = Path(config["images_dir"])
        labels_dir = Path(config["labels_dir"])

        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  Skipping {dataset_name}: missing directories")
            continue

        # Find all image files (supporting subdirectories)
        image_files = []
        for ext in config["extensions"]:
            image_files.extend(images_dir.glob(f"**/*{ext}"))

        print(f"📊 {dataset_name}: Found {len(image_files)} images")

        for img_path in image_files:
            # Find corresponding label file
            label_filename = f"{img_path.stem}.txt"
            label_path = labels_dir / label_filename

            if not label_path.exists():
                continue

            # Filter for AP-10K Cat/Dog categories
            if dataset_name == "ap10k" and "filter_categories" in config:
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()

                    # Check if label file contains cat/dog categories
                    has_cat_dog = False
                    for line in content.split('\n'):
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                class_id = int(parts[0])
                                if class_id in config["filter_categories"]:
                                    has_cat_dog = True
                                    break

                    if not has_cat_dog:
                        continue  # Skip non-cat/dog samples

                except Exception as e:
                    print(f"⚠️  Error reading AP-10K label {label_path}: {e}")
                    continue

            all_samples.append({
                "image_path": img_path,
                "label_path": label_path,
                "dataset": dataset_name
            })

    print(f"📊 Total samples collected: {len(all_samples)}")

    # Shuffle and split samples
    random.shuffle(all_samples)

    # Filter invalid data (all-zero keypoints)
    print("\n🧹 Filtering invalid data...")
    filtered_samples = []
    invalid_count = 0

    for sample in all_samples:
        label_path = sample["label_path"]
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()

            if content:
                parts = content.split()
                if len(parts) >= 11:  # Requires class ID + 10 coordinates minimum
                    # Check if keypoints are all zero
                    keypoints = list(map(float, parts[1:11]))
                    all_zero = all(kp == 0 for kp in keypoints)

                    if not all_zero:
                        filtered_samples.append(sample)
                    else:
                        invalid_count += 1
            else:
                invalid_count += 1

        except Exception as e:
            print(f"  Error reading {label_path.name}: {e}")
            invalid_count += 1

    print(f"  Total samples: {len(all_samples)}")
    print(f"  Filtered invalid samples: {invalid_count} ({invalid_count/len(all_samples)*100:.1f}%)")
    print(f"  Remaining valid samples: {len(filtered_samples)}")

    all_samples = filtered_samples
    total = len(all_samples)

    if total == 0:
        print("❌ No samples remaining after filtering")
        return {}

    # Resplit samples
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:]
    }

    # Copy files to output directories and create relative paths
    relative_paths = {"train": [], "val": [], "test": []}

    for split_name, samples in splits.items():
        print(f"\n📁 Processing {split_name} split ({len(samples)} samples)...")

        for i, sample in enumerate(samples):
            # Copy image file
            img_dest = image_dirs[split_name] / sample["image_path"].name
            if not img_dest.exists():
                shutil.copy(sample["image_path"], img_dest)

            # Copy and convert label file to YOLOv8 pose format
            label_dest = label_dirs[split_name] / sample["label_path"].name
            if not label_dest.exists():
                converted_label = convert_label_to_yolo_pose_format(sample["label_path"])
                if converted_label:
                    with open(label_dest, 'w') as f:
                        f.write(converted_label)
                else:
                    print(f"⚠️  Skipping invalid label: {sample['label_path'].name}")
                    continue

            # Store relative path for YAML config
            rel_path = f"images/{split_name}/{sample['image_path'].name}"
            relative_paths[split_name].append(rel_path)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")

    # Create dataset.yaml configuration
    yaml_config = {
        "path": str(output_path),
        "train": f"images/train",
        "val": f"images/val",
        "test": f"images/test",
        "names": {
            0: "cat",
            1: "dog"
        },
        "nc": 2,
        "kpt_shape": [5, 3]  # 5 keypoints, each with (x, y, visibility)
    }

    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ YOLO pose dataset created successfully!")
    print(f"   Total samples: {total}")
    print(f"   Train: {len(splits['train'])}")
    print(f"   Val: {len(splits['val'])}")
    print(f"   Test: {len(splits['test'])}")
    print(f"   Config: {yaml_path}")

    return relative_paths


def convert_label_to_yolo_pose_format(label_path):
    """
    Convert labels to YOLOv8 pose format: class + bbox + keypoints (x, y, visibility)
    YOLOv8 pose requires 15 columns: class, cx, cy, w, h, x1, y1, v1, x2, y2, v2, x3, y3, v3
    """
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()

        if not content:
            return None

        # Parse original label format
        parts = content.split()
        if len(parts) < 11:  # Requires at least 11 columns: Class + 5 keypoints * 2 coordinates
            return None

        # Extract class and keypoints
        class_id = int(parts[0])
        keypoints = list(map(float, parts[1:11]))

        # Calculate bounding box (derived from keypoints)
        x_coords = keypoints[::2]  # All x coordinates
        y_coords = keypoints[1::2]  # All y coordinates

        # Filter out zero-value keypoints
        valid_x = [x for x in x_coords if x > 0]
        valid_y = [y for y in y_coords if y > 0]

        if not valid_x or not valid_y:
            return None

        x_min, x_max = min(valid_x), max(valid_x)
        y_min, y_max = min(valid_y), max(valid_y)

        # Calculate bbox center, width, and height
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        # Construct YOLOv8 pose format (Class + BBox + 5 Keypoints + Visibility)
        yolo_pose_label = [f"{class_id:.6f}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

        # Ensure exactly 15 keypoint values (5 keypoints * 3 values)
        processed_keypoints = []
        for i in range(5):
            if i*2 + 1 < len(keypoints):
                x = keypoints[i*2]
                y = keypoints[i*2 + 1]
            else:
                x, y = 0.0, 0.0  # Pad missing keypoints

            # Ensure coordinates are within [0,1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            # Visibility: 2 if coordinates exist, else 0
            visibility = 2 if (x > 0 and y > 0) else 0
            processed_keypoints.extend([x, y, visibility])

        # Ensure exactly 15 values
        if len(processed_keypoints) != 15:
            processed_keypoints = processed_keypoints[:15]  # Truncate excess values
            while len(processed_keypoints) < 15:
                processed_keypoints.extend([0.0, 0.0, 0.0])  # Pad missing values

        # Add to label list
        yolo_pose_label.extend([f"{v:.6f}" for v in processed_keypoints])

        return " ".join(yolo_pose_label)

    except Exception as e:
        print(f"Error converting label {label_path}: {e}")
        return None

def generate_yolo_pose_dataset_command():
    """
    Generate command to run the YOLO pose dataset creation
    """
    return """
# Run the following command to create YOLO pose dataset:
python3 -c "
from utils.data_utils import create_yolo_pose_dataset
create_yolo_pose_dataset(
    output_dir='data/yolo_pose_data',
    split_ratios=(0.8, 0.1, 0.1),
    seed=42
)
print('🎯 YOLO pose dataset ready for YOLOv8m-pose training!')
"
"""