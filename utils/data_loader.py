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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFile

import logging
import json
from typing import Dict, Optional, List
from torchvision.transforms import autoaugment
from torch.utils.data.distributed import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup logger
logger = logging.getLogger(__name__)

# --- 1. Import LDRE Augmentation Modules ---
try:
    from models.modules.ldre import LDRETransform
    from models.modules.enhanced_ldre import EnhancedLDRETransform
    LDRE_AVAILABLE = True
except ImportError:
    LDRE_AVAILABLE = False
    logger.warning("LDRE or EnhancedLDRE transform module not available...")

class PetDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 224, split: str = "train",
                 transform_type: str = "default", ldre_cfg: Optional[Dict] = None):
        """
        Pet classification dataset loader with LDRE support

        Args:
            root_dir: Dataset root directory
            img_size: Output image size
            split: Dataset split ("train"/"val"/"test")
            transform_type: Transformation type
            ldre_cfg: LDRE configuration dictionary
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type

        if not self.root_dir.is_absolute():
            project_root = Path(__file__).parent.parent
            self.root_dir = project_root / "data" / self.root_dir

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory does not exist: {split_dir}")

        # --- Robust Label Mapping (Revised Logic) ---
        self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if not self.classes:
            raise FileNotFoundError(f"No class folders found in {split_dir}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.img_paths: List[Path] = []
        self.labels: List[int] = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.img_paths.append(img_path)
                    self.labels.append(class_idx)

        # --- 2. Load Keypoints Database (If required) ---
        self.keypoints_db = None
        if ldre_cfg and ldre_cfg.get('enable', False) and ldre_cfg.get('enhanced_mode', False):
            db_path = Path('data/keypoints_db.json')
            if not db_path.is_absolute():
                db_path = Path(__file__).parent.parent / db_path
            if db_path.exists():
                print(f"🔑 Loading keypoints database: {db_path}")
                with open(db_path, 'r') as f:
                    self.keypoints_db = json.load(f)
            else:
                logger.warning(f"Enhanced LDRE enabled but keypoints DB not found: {db_path}")

        # --- 3. Initialize LDRE Transform ---
        self.ldre_transform = None
        if ldre_cfg and ldre_cfg.get('enable', False) and LDRE_AVAILABLE:
            if ldre_cfg.get('enhanced_mode', False) and self.keypoints_db is not None:
                # Enable Enhanced Mode
                print("✨ Enhanced LDRE Enabled (Keypoint-guided)")
                self.ldre_transform = EnhancedLDRETransform(
                    grid_size=ldre_cfg.get('grid_size', 16),
                    top_k_ratio=ldre_cfg.get('top_k_ratio', 0.2),
                    prob=ldre_cfg.get('prob', 0.5)
                )
            else:
                # Enable Standard Mode
                print("🎲 Standard LDRE Enabled (Random Erasing)")
                self.ldre_transform = LDRETransform(
                    grid_size=ldre_cfg.get('grid_size', 16),
                    drop_count=ldre_cfg.get('drop_count', 2),
                    prob=ldre_cfg.get('prob', 0.5)
                )

        # Image transforms
        if split == 'train':
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])
            print("🚀 Using strong augmentations for training.")
        else:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("🔬 Using standard transforms for validation/testing.")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    # _read_image_robust handles image loading only, no transforms applied
    def _read_image_robust(self, img_path: Path) -> Optional[np.ndarray]:
        try:
            pil_img = Image.open(img_path).convert('RGB')
            return np.array(pil_img)
        except Exception as e:
            logger.warning(f"PIL read failed {img_path}: {e}")
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e2:
                logger.warning(f"OpenCV read failed {img_path}: {e2}")
            return None

    def _get_class_id_from_path(self, img_path: Path) -> int:
        class_name = img_path.parent.name
        if class_name.startswith('pets_'):
            try:
                class_id_str = class_name.split('_')[1]
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
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # --- 4. Core __getitem__ Logic ---

        # Step A: Read original image
        img = self._read_image_robust(img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}, returning empty sample")
            return self._get_empty_sample()

        # Step B: Apply LDRE (Only in training mode)
        if self.split == "train" and self.ldre_transform is not None:
            if isinstance(self.ldre_transform, EnhancedLDRETransform):
                # For Enhanced Mode, lookup keypoints
                key = str(img_path.relative_to(self.root_dir.parent)).replace('\\', '/')
                keypoints_info = self.keypoints_db.get(key)
                keypoints = keypoints_info[0]['keypoints'] if keypoints_info else None

                # Pass both image and keypoints
                img = self.ldre_transform(img, keypoints)
            else:
                # For Standard Mode, pass image only
                img = self.ldre_transform(img)

        # Step C: Apply standard image transforms
        transformed_img = self.image_transform(img)

        # Step D: Return data and label
        return {
            'image': transformed_img,
            'labels': label,
            'image_path': str(img_path)
        }

    def _get_empty_sample(self):
        return {
            'image': torch.zeros(3, self.img_size, self.img_size),
            'labels': 0,
            'image_path': ''
        }

def collate_fn(batch):
    images = []
    labels = []
    meta_data = []
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['labels'])
        meta_data.append({'image_path': sample['image_path']})

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
    ldre_cfg = kwargs.pop('ldre_cfg', None)
    img_size = kwargs.pop('img_size', 224)
    dataset = PetDataset(root_dir, img_size=img_size, split=split, ldre_cfg=ldre_cfg, **kwargs)

    if sampler is not None:
        shuffle = False  # Disable shuffle when using distributed sampler

    drop_last_batch = (split == 'train')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last_batch
    )

def build_datasampler(root_dir, split="train", shuffle=True, **kwargs):
    """
    Construct Sampler for distributed training
    """
    # Instantiate Dataset to get length, images are not loaded yet
    # kwargs passes through ldre_cfg etc.
    dataset = PetDataset(root_dir, split=split, **kwargs)

    return DistributedSampler(dataset, shuffle=shuffle)