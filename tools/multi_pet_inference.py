#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/multi_pet_inference.py
Location: tools/
====================================
Multi-Pet Inference System (MultiPetInference)

Purpose:
- Provide end-to-end inference pipeline for multiple pet detection and classification
- Support YOLO-based detection combined with PetNet classification
- Enable batch processing and comprehensive result visualization

Key Features:
1. YOLO Detection Integration: Leverage YOLO for accurate pet object detection
2. PetNet Classification: Use trained PetNet models for fine-grained pet classification
3. Multi-Object Handling: Support detection and classification of multiple pets in single image
4. Confidence Thresholding: Configurable confidence levels for both detection and classification
5. Comprehensive Visualization: Generate detailed result visualizations with bounding boxes and labels
6. Batch Processing: Support for processing multiple images in sequence
"""

import torch
import cv2
import numpy as np
import sys
import argparse
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.petnet import PetNet


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


class MultiPetDetector:
    def __init__(self, detector_path, classifier_path, conf_threshold=0.5, img_size=256):
        """
        Initialize multi-pet detection and classifier
        Args:
            img_size: Inference resolution, recommend keeping consistent with fine-tuning (256)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.img_size = img_size  # Explicitly save resolution

        # 1. Load YOLO detector
        self.detector_path = self._get_detector_path(detector_path)
        print(f"🔍 Loading detector: {self.detector_path}")
        self.detector = YOLO(self.detector_path)

        # 2. Load PetNet configuration
        print(f"🔍 Loading classifier weights: {classifier_path}")
        if not Path(classifier_path).exists():
            raise FileNotFoundError(f"Model file not found: {classifier_path}")

        checkpoint = torch.load(classifier_path, map_location=self.device)

        # Compatibility handling: If no config in checkpoint, try to load default
        if 'config' in checkpoint:
            full_config = checkpoint['config']
            model_config = full_config['model']
        else:
            print("⚠️ No config found in checkpoint, trying to load default config configs/petnet_base.yaml...")
            config_path = project_root / "configs" / "petnet_base.yaml"
            full_config = load_config(config_path)
            model_config = full_config['model']

        # 3. Initialize PetNet (fixed parameter list)
        # Note: Must disable drop_path_rate during inference
        self.classifier = PetNet(
            num_classes=model_config.get('num_classes', 144),
            stage_repeats=model_config.get('stage_repeats', [2, 3, 4]),
            attn_cfg=model_config.get('attn_cfg', None),
            selfkd_cfg=model_config.get('selfkd_cfg', None),
            max_pets_per_image=10,
            drop_path_rate=0.0
        )

        # 4. Load weights
        model_state_dict = self.classifier.state_dict()
        # Filter mismatched keys (e.g., aux head or teacher buffer)
        pretrained_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
        self.classifier.load_state_dict(pretrained_dict, strict=False)
        self.classifier.to(self.device)
        self.classifier.eval()

        # 5. Define preprocessing
        print(f"📏 Inference input resolution: {self.img_size}x{self.img_size}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load class names
        self.class_names = self._load_class_names(model_config.get('num_classes', 144))

    def _get_detector_path(self, detector_path):
        """Get detector model path, prefer pretrained models"""
        if Path(detector_path).exists():
            return detector_path
        if Path('./yolov8n.pt').exists():
            return './yolov8n.pt'
        return 'yolov8n.pt'  # Let ultralytics auto-download

    def _load_class_names(self, num_classes):
        """Try to load class mapping"""
        try:
            from utils.data_utils import get_class_names
            # Assume dataset name is pet_cls_training, you can modify as needed
            return get_class_names("pet_cls_training")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load class mapping ({e}), Using default IDs.")
            return {i: f"Class_{i}" for i in range(num_classes)}
    def detect_pets(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        detections = []

        # YOLO inference
        # classes=[15, 16] Only detect cats(15) and dogs(16)
        results = self.detector(img_rgb, conf=0.1, iou=0.45, classes=[15, 16], verbose=False)

        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Filter low confidence
                    if conf < self.conf_threshold:
                        continue

                    # 🛠️ Optimize: Expand detection box (Padding) to prevent cropping ears
                    # Expand top, bottom, left, right by 5% each
                    pad_w = (x2 - x1) * 0.05
                    pad_h = (y2 - y1) * 0.05

                    x1_pad = max(0, int(x1 - pad_w))
                    y1_pad = max(0, int(y1 - pad_h))
                    x2_pad = min(w, int(x2 + pad_w))
                    y2_pad = min(h, int(y2 + pad_h))

                    # Crop and predict
                    crop_img = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]

                    if crop_img.size == 0: continue

                    # PetNet inference
                    input_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.classifier(input_tensor)
                        if isinstance(logits, tuple):
                            logits = logits[0]

                        probs = torch.softmax(logits, dim=1)
                        class_conf, class_id = probs.max(1)

                    # Record results
                    class_name = self.class_names.get(class_id.item(), f"Class_{class_id.item()}")

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'yolo_class_name': 'cat' if cls_id == 15 else 'dog',
                        'detection_confidence': conf,
                        'class_id': class_id.item(),
                        'class_confidence': class_conf.item(),
                        'class_name': class_name
                    })

                    print(f"🐶 Detected: {detections[-1]['yolo_class_name']} -> {class_name} ({class_conf.item():.2%})")

        return detections

    def visualize_results(self, image_path, detections, output_path=None):
        """Visualize detection results and save"""
        img = cv2.imread(image_path)
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        ax = plt.gca()

        # Color mapping: red for cats, blue for dogs
        colors = {'cat': 'red', 'dog': 'blue'}

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width, height = x2 - x1, y2 - y1

            species = det['yolo_class_name']
            color = colors.get(species, 'green')

            # Draw bounding box
            rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Draw label background
            label_text = f"{det['class_name']} ({det['class_confidence']:.1%})"
            plt.text(x1, y1 - 5, label_text, color='white', fontsize=10, weight='bold',
                     bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))

        plt.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"✅ Result saved to: {output_path}")

        # If running on server, you can comment this line out
        # plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='PetBuddy Multi-Pet Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--classifier', type=str, required=True, help='Path to PetNet weights (.pt)')
    parser.add_argument('--detector', type=str, default='yolov8n.pt', help='Path to YOLO weights')
    parser.add_argument('--output', type=str, default='result.jpg', help='Path to save result image')
    parser.add_argument('--conf', type=float, default=0.4, help='Detection confidence threshold')
    parser.add_argument('--img_size', type=int, default=256, help='Inference image size (must match training)')

    args = parser.parse_args()

    # Initialize
    detector = MultiPetDetector(
        detector_path=args.detector,
        classifier_path=args.classifier,
        conf_threshold=args.conf,
        img_size=args.img_size
    )

    # Inference
    print("-" * 30)
    detections = detector.detect_pets(args.image)
    print("-" * 30)

    # Visualize
    if detections:
        detector.visualize_results(args.image, detections, args.output)
    else:
        print("⚠️ No pets detected.")


if __name__ == "__main__":
    main()