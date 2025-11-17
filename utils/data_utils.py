
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
from typing import Dict, List
import random
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

if __name__ == "__main__":
    prepare_dataset_from_config()
