#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: data_loader.py
====================================
Unified Pet Data Loader

Purpose:
- Support both single-pet and multi-pet loading modes
- Provide data augmentation for pet recognition training
- Integrate with YOLOv8 and PetNet model architectures
- Handle classification and detection dataset formats

Features:
1. Multi-modal Support: Single-pet and multi-pet training modes
2. Data Augmentation: Comprehensive augmentation pipeline
3. Format Compatibility: YOLO, COCO, and custom dataset formats
4. Performance Optimization: Efficient data loading and preprocessing
5. Cross-dataset: Support for Oxford Pets, Stanford Dogs, and custom datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFile
from typing import Dict, Optional
import logging

from torchvision.transforms import autoaugment

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup logger
logger = logging.getLogger(__name__)

try:
    from models.modules.ldre import LDRETransform, LDRETransformCompose
    LDRE_AVAILABLE = True
except ImportError:
    LDRE_AVAILABLE = False
    logger.warning("LDRE transform module not available, using simplified implementation")

class PetDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 224, split: str = "train",
                 transform_type: str = "default", ldre_cfg: Optional[Dict] = None):
        """
        Pet classification dataset loader

        Args:
            root_dir: Dataset root directory (points to data/pet_cls_training)
            img_size: Output image size
            split: Dataset split ("train"/"val"/"test")
            transform_type: Transformation type ("default"/"yolo_to_cls")
            ldre_cfg: LDRE configuration dictionary (grid_size, drop_count, prob)
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type

        # Initialize LDRE transform
        self.ldre_transform = None
        if ldre_cfg and ldre_cfg.get('enable', False) and LDRE_AVAILABLE:
            ldre_params = {k: v for k, v in ldre_cfg.items() if k != 'enable'}
            self.ldre_transform = LDRETransform(**ldre_params)

        # Handle relative paths: if path is not absolute, resolve based on project root
        if not self.root_dir.is_absolute():
            # Try to resolve path from project root
            project_root = Path(__file__).parent.parent
            absolute_root_dir = project_root / self.root_dir
            if absolute_root_dir.exists():
                self.root_dir = absolute_root_dir
            else:
                # If path doesn't exist, try adding data/ prefix
                data_dir = project_root / "data" / self.root_dir
                if data_dir.exists():
                    self.root_dir = data_dir

        # Get image paths (new structure directly contains train/val/test)
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise ValueError(f"Dataset split directory does not exist: {split_dir}")

        self.img_paths = []
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                self.img_paths.extend(list(class_dir.glob("*.jpg")))
                self.img_paths.extend(list(class_dir.glob("*.png")))
        if not self.img_paths:
            raise ValueError(f"No jpg image files found in {split_dir}")

        # Randomly shuffle image paths during initialization to ensure class mixing
        import random
        random.shuffle(self.img_paths)
        if split == 'train':
            # Strong augmentations for the training set
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size + 32, img_size + 32)),  # Resize to a larger size first
                transforms.RandomCrop(img_size),  # Then crop to the target size
                transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
                autoaugment.RandAugment(num_ops=2, magnitude=9),  # RandAugment (m=9, n=2 from your doc)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
                # Simulates Cutout/GridMask
            ])
            print("🚀 Using strong augmentations for training.")
        else:
            # Simple resizing and normalization for validation and testing
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("🔬 Using standard transforms for validation/testing.")




    def _read_image_robust(self, img_path: Path, apply_ldre: bool = False) -> Optional[np.ndarray]:
        """
        Robustly read image file, handling corrupted JPEG files

        Args:
            img_path: Image file path
            apply_ldre: Whether to apply LDRE preprocessing

        Returns:
            np.ndarray or None: Read image data, returns None if reading fails
        """
        try:
            # Method 1: Use PIL reading (more friendly to corrupted files)
            pil_img = Image.open(img_path)
            img_array = np.array(pil_img.convert('RGB'))

            # Apply LDRE preprocessing
            if apply_ldre and self.ldre_transform is not None:
                img_array = self.ldre_transform(img_array)

            return img_array
        except Exception as e:
            logger.warning(f"PIL read failed {img_path}: {e}")
            try:
                # Method 2: Use OpenCV as backup
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Apply LDRE preprocessing
                    if apply_ldre and self.ldre_transform is not None:
                        img_array = self.ldre_transform(img_array)

                    return img_array
                return None
            except Exception as e2:
                logger.warning(f"OpenCV read failed {img_path}: {e2}")
                return None

    def _get_class_id_from_path(self, img_path: Path) -> int:
        """Get class ID from file path

        Get class ID based on directory name format, expected format: pets_0001_staffordshire_bull_terrier
        The numeric part is the class ID (converted to 0-based)
        """
        class_name = img_path.parent.name

        if class_name.startswith('pets_'):
            try:
                # Extract numeric part from pets_0001_staffordshire_bull_terrier format
                class_id_str = class_name.split('_')[1]
                # Convert 1-based class ID to 0-based
                return int(class_id_str) - 1
            except (IndexError, ValueError):
                logger.warning(f"Unable to extract class ID from directory name: {class_name}")
                return 0
        else:
            logger.warning(f"Non-standard pets directory format: {class_name}")
            return 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Dict:
        """
        Get classification data sample

        Returns:
            dict: Contains the following key-value pairs
                - image: Image tensor
                - labels: Class label
                - image_path: Image file path
        """
        img_path = self.img_paths[idx]

        # Read image (apply LDRE preprocessing in training mode)
        apply_ldre = self.split == "train"  # Only apply LDRE during training
        img = self._read_image_robust(img_path, apply_ldre=apply_ldre)
        if img is None:
            logger.warning(f"Unable to read image file: {img_path}")
            return self._get_empty_sample()

        # Get class ID from file path
        class_id = self._get_class_id_from_path(img_path)

        result = {
            'image': self.image_transform(img),
            'labels': class_id,
            'image_path': str(img_path)
        }

        return result

    def _get_empty_sample(self):
        """Return empty sample"""
        return {
            'image': torch.zeros(3, self.img_size, self.img_size),
            'labels': 0,
            'image_path': ''
        }

def collate_fn(batch):
    """Collate function for classification data"""
    images = []
    labels = []
    meta_data = []

    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['labels'])
        meta_data.append({
            'image_path': sample['image_path']
        })

    if not images:
        return {
            'images': torch.zeros(0, 3, 224, 224),
            'labels': torch.zeros(0, dtype=torch.long),
            'meta_data': meta_data
        }

    return {
        'images': torch.stack(images),
        'labels': torch.tensor(labels, dtype=torch.long),
        'meta_data': meta_data
    }

def build_dataloader(root_dir, batch_size=32, shuffle=True,
                           num_workers=4, split="train", sampler=None, **kwargs):
    """
    Build pet classification data loader

    Args:
        root_dir: Dataset root directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        split: Dataset split ("train"/"val"/"test")
    """
    # Extract LDRE configuration from kwargs
    ldre_cfg = kwargs.pop('ldre_cfg', None)

    dataset = PetDataset(root_dir, split=split, ldre_cfg=ldre_cfg, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),  # Only enable pin_memory when CUDA is available
        persistent_workers=num_workers > 0  # Avoid frequent creation/destruction of worker processes
    )

def build_datasampler(root_dir, shuffle=True, split="train", **kwargs):
    """
    Build pet classification data sampler (only for distributed env)

    Args:
        root_dir: Dataset root directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        split: Dataset split ("train"/"val"/"test")
    """

    # Extract LDRE configuration from kwargs
    ldre_cfg = kwargs.pop('ldre_cfg', None)

    dataset = PetDataset(root_dir, split=split, ldre_cfg=ldre_cfg, **kwargs)

    return DistributedSampler(dataset, shuffle=shuffle)
