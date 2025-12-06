#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/multi_pet_inference.py
====================================
Multi-Pet Inference System (Final Fixed Version with Path Correction, Label, and Save Control)
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
from matplotlib import patheffects

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.petnet import PetNet


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        # Fallback if specific config not found
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MultiPetDetector:
    def __init__(self, detector_path, classifier_path, det_conf_threshold=0.5, cls_conf_threshold=0.7, img_size=224):
        """
        Initialize multi-pet detection and classifier
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_conf_threshold = det_conf_threshold
        self.cls_conf_threshold = cls_conf_threshold
        self.img_size = img_size

        # 1. Load YOLO detector
        self.detector_path = self._get_detector_path(detector_path)
        print(f"🔍 Loading detector: {self.detector_path}")
        self.detector = YOLO(self.detector_path)

        # 2. Load PetNet configuration and weights
        print(f"🔍 Loading classifier weights: {classifier_path}")
        if not Path(classifier_path).exists():
            raise FileNotFoundError(f"Model file not found: {classifier_path}")

        checkpoint = torch.load(classifier_path, map_location=self.device)

        # Try to get config from checkpoint or fallback
        if 'config' in checkpoint:
            full_config = checkpoint['config']
            model_config = full_config.get('model', {})
        else:
            print("⚠️ No config in checkpoint, loading default config for model setup...")
            config_path = project_root / "configs" / "petnet_base.yaml"
            full_config = load_config(config_path)
            model_config = full_config.get('model', {})

        # 3. Initialize PetNet
        num_classes = model_config.get('num_classes', 144)
        self.classifier = PetNet(
            num_classes=num_classes,
            stage_repeats=model_config.get('stage_repeats', [2, 3, 4]),
            attn_cfg=model_config.get('attn_cfg', None),
            selfkd_cfg=model_config.get('selfkd_cfg', None),
            max_pets_per_image=10,
        )

        # 4. Load weights (CRITICAL FIX FOR DDP)
        model_state_dict = self.classifier.state_dict()
        state_dict_to_load = {}

        # Train script saves under 'model_state_dict'
        ckpt_dict = checkpoint.get('model_state_dict', checkpoint)

        for k, v in ckpt_dict.items():
            # Remove 'module.' prefix if trained with DDP
            if k.startswith('module.'):
                k = k[7:]

            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                state_dict_to_load[k] = v

        if len(state_dict_to_load) == 0:
            print("❌ ERROR: No matching weights found! Please confirm checkpoint integrity and model structure.")
        else:
            print(f"✅ Loaded {len(state_dict_to_load)}/{len(model_state_dict)} layers.")

        self.classifier.load_state_dict(state_dict_to_load, strict=False)
        self.classifier.to(self.device)
        self.classifier.eval()

        # 5. Define preprocessing
        print(f"📏 Inference resolution: {self.img_size}x{self.img_size}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 6. Load Class Names (Path Fixed & Formatting Simplified)
        self.class_names = self._load_class_mapping(num_classes)

    def _get_detector_path(self, detector_path):
        if Path(detector_path).exists(): return detector_path
        return 'yolov8n.pt'

    def _load_class_mapping(self, num_classes):
        """
        Load correct breed names from dataset.yaml, clean the prefix, and return ONLY the simple breed name.
        """
        yaml_path = project_root / "data" / "merged_cls_dataset" / "dataset.yaml"
        print(f"Attempting to load class map from: {yaml_path}")

        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)

                names_list = config.get('names')
                if names_list and isinstance(names_list, list):

                    cleaned_names_list = []
                    for name in names_list:
                        name_str = name.strip()
                        # format: "pets_0001_staffordshire_bull_terrier"
                        parts = name_str.split('_', 2)

                        if len(parts) == 3 and parts[0] == 'pets' and parts[1].isdigit():

                            #Extracting Breed Names, Replacing Them with Underscores, and Capitalizing the First Letter of Each Word
                            breed_name_raw = parts[2].replace('_', ' ')
                            cleaned_name = breed_name_raw.title()
                        else:
                            # if format unmatched，use te original name
                            cleaned_name = name_str

                        # The store only contains the specific category name, and visualizes the label
                        cleaned_names_list.append(cleaned_name)

                    names_map = {i: name for i, name in enumerate(cleaned_names_list)}

                    if len(names_map) == num_classes:
                        print(f"✅ Loaded and cleaned {len(names_map)} simple class names.")
                        return names_map
                    else:
                        print(
                            f"⚠️ Warning: Found class names in YAML ({len(names_map)}), but count doesn't match model's expected classes ({num_classes}). Using loaded names.")
                        return names_map

            except Exception as e:
                print(f"❌ Error loading and parsing class names from YAML: {e}.")

        # 2. Fallback to generic names
        print(
            f"⚠️ Warning: Could not find or load correct breed map at expected location. Using ID format (e.g., Breed_1).")
        return {i: f"Breed_{i + 1}" for i in range(num_classes)}

    def detect_pets(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        detections = []

        # YOLO inference (Cats=15, Dogs=16)
        # Use low confidence for YOLO to capture most pets, classifier will filter the rest
        results = self.detector(img_rgb, conf=0.1, iou=0.45, classes=[15, 16], verbose=False)

        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    det_conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    if det_conf < self.det_conf_threshold:
                        continue

                    # crop image to bbox area
                    x1_crop = max(0, int(x1))
                    y1_crop = max(0, int(y1))
                    x2_crop = min(w, int(x2))
                    y2_crop = min(h, int(y2))

                    crop_img = img_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
                    if crop_img.size == 0: continue

                    # PetNet inference
                    input_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.classifier(input_tensor)
                        if isinstance(logits, tuple): logits = logits[0]
                        probs = torch.softmax(logits, dim=1)
                        class_conf, class_id = probs.max(1)

                    # Filter by classification confidence
                    if class_conf.item() < self.cls_conf_threshold:
                        continue

                    class_name = self.class_names.get(class_id.item(), f"Breed_{class_id.item() + 1}")

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'yolo_class_name': 'cat' if cls_id == 15 else 'dog',
                        'detection_confidence': det_conf,
                        'class_id': class_id.item(),
                        'class_confidence': class_conf.item(),
                        'class_name': class_name  # breed name
                    })

                    print(f"🐶 Detected: {class_name} ({class_conf.item():.2%})")

        return detections

    def visualize_results(self, image_path, detections, output_path=None):
        """Visualize detection results using Matplotlib with dynamic scaling and DPI control."""
        try:
            # 1. read images
            img = cv2.imread(image_path)  # BGR
            if img is None:
                print(f"❌ Error: Could not read image at {image_path}")
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Matplotlib uses RGB
            H, W, _ = img.shape

            #  Dynamic Size Calculation

            #Target Maximum Width (e.g., 1280 pixels, adjustable as needed)
            TARGET_MAX_WIDTH = 1280

            # If the original image width exceeds the target maximum width, calculate the scaling ratio
            if W > TARGET_MAX_WIDTH:
                scale_factor = TARGET_MAX_WIDTH / W
            else:
                scale_factor = 1.0  # No enlargement

            # Use a reasonable base DPI to ensure rendering quality
            BASE_DPI = 100

            # Dynamically adjust the DPI so that the width of the finally saved image is close to TARGET_MAX_WIDTH
            # figsize * dpi = pixel_size
            # (W / BASE_DPI) * new_dpi = W * scale_factor  => new_dpi = BASE_DPI * scale_factor
            display_dpi = BASE_DPI * scale_factor

            # Dynamically adjust the font size (proportional to the scaling factor)
            # SE_FONT_SIZE_PT: A value between 12–16pt is usually sufficiently clear
            BASE_FONT_SIZE_PT = 14
            dynamic_font_size = max(8, int(BASE_FONT_SIZE_PT * scale_factor))
            BOX_LINE_WIDTH = max(1, int(3 * scale_factor))  # the min value 1

            # 2.init  Matplotlib
            fig, ax = plt.subplots(1, figsize=(W / BASE_DPI, H / BASE_DPI), dpi=BASE_DPI)  # 使用基础 DPI 初始化
            ax.imshow(img)
            ax.set_axis_off()

            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                color_mpl = 'red' if det['yolo_class_name'] == 'cat' else 'blue'
                label = f"{det['class_name']} ({det['class_confidence']:.2f})"

                # 3. 绘制边界框
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=BOX_LINE_WIDTH,
                                 edgecolor=color_mpl,
                                 facecolor='none')
                ax.add_patch(rect)

                # 4. drwa
                text_y = y1 - 2
                if text_y < 0.05 * H:
                    text_y = y1 + 5

                ax.text(x1, text_y, label,
                        color='white',
                        fontsize=dynamic_font_size,  # use dynamic font size
                        path_effects=[
                            patheffects.withStroke(
                                linewidth=max(2, BOX_LINE_WIDTH + 1),  # create a thicker outline
                                foreground=color_mpl
                            )
                        ],
                        verticalalignment='top',
                        bbox={
                            'facecolor': color_mpl,
                            'alpha': 0.8,
                            'pad': 2,
                            'edgecolor': 'none'
                        })

            # 5. save results
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            ax.autoscale_view()

            if output_path:
                # key: plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=display_dpi)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=display_dpi)
                print(f"✅ Result saved to: {output_path}")

            plt.close(fig)

        except Exception as e:
            # make sure:from matplotlib import patheffects
            print(f"❌ Error during visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description='PetBuddy Multi-Pet Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--classifier', type=str, default='runs/petnet_fine_tune/best.pt', help='Path to PetNet weights (.pt)')
    parser.add_argument('--detector', type=str, default='yolov11n.pt', help='Path to YOLO weights')

    #
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save result image (Default: current directory)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Flag to enable saving the output visualization image. (Default: False)')

    parser.add_argument('--det_conf', type=float, default=0.2, help='YOLO Detection confidence threshold')
    parser.add_argument('--cls_conf', type=float, default=0.1, help='PetNet Classification confidence threshold')
    parser.add_argument('--img_size', type=int, default=224, help='Inference image size (Default is 224)')

    args = parser.parse_args()

    detector = MultiPetDetector(
        detector_path=args.detector,
        classifier_path=args.classifier,
        det_conf_threshold=args.det_conf,
        cls_conf_threshold=args.cls_conf,
        img_size=args.img_size
    )

    print("-" * 30)
    detections = detector.detect_pets(args.image)
    print("-" * 30)


    output_save_path = None
    if detections and args.save:

        original_path = Path(args.image)
        original_stem = original_path.stem
        original_suffix = original_path.suffix
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # account the number of pets detected
        if len(detections) == 1:
            # single pet: image_name_breed_name.ext

            breed_name = detections[0]['class_name'].replace(' ', '_')
            final_name = f"{original_stem}_{breed_name}{original_suffix}"
        elif len(detections) > 1:
            # multi pets: image_name_breed1_breed2_etc.ext
            all_breeds = [det['class_name'].replace(' ', '_') for det in detections]
            breed_suffix = "_".join(all_breeds[:2])
            if len(all_breeds) > 2:
                breed_suffix += f"_and_{len(all_breeds) - 2}more"

            final_name = f"{original_stem}_{breed_suffix}{original_suffix}"
        else:
            final_name = f"{original_stem}_no_pet_detected{original_suffix}"

        output_save_path = output_dir / final_name
        print(f"📝 Saving result to: {output_save_path}")

    if output_save_path:

        detector.visualize_results(args.image, detections, str(output_save_path))
    elif detections:

        print("💡 Result visualization/saving skipped (default behavior).")
        print("   To save the output image, use the --save flag.")
    else:
        print("⚠️ No pets detected or all filtered by confidence thresholds.")


if __name__ == "__main__":
    main()