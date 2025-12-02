#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/experiments/analysis/visualize_gradcam.py
Location: tools/experiments/analysis/
====================================
Grad-CAM Visualizer (VisualizeGradCAM)

Purpose:
- Generate Grad-CAM visualizations for model interpretability and explainability
- Support heatmap generation for pet recognition model predictions
- Enable comprehensive visualization of model attention mechanisms

Key Features:
1. Automatic Target Layer Detection: Smart detection of convolutional layers for Grad-CAM
2. Multiple Output Formats: Support for individual heatmaps and concatenated visualizations
3. Model Compatibility: Works with both baseline MobileNetV2 and enhanced PetNet architectures
4. Publication Quality: Generate high-quality visualizations suitable for academic papers
5. Flexible Configuration: Customizable image size and output format options
"""

import torch
import cv2
import numpy as np
import argparse
import sys
import yaml
from pathlib import Path
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image, ImageDraw, ImageFont

# Add project root directory to system path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from models.petnet import PetNet


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_target_layer(model):
    """Automatically find target convolutional layer for MobileNetV2/PetNet"""
    # Usually choose the last block of Stage 3
    # For PetNet, structure is typically stage3 -> list of blocks
    if hasattr(model, 'stage3'):
        last_block = model.stage3[-1]
        # If the last module is ECAPos (Attention), we need to look at the previous convolution
        if hasattr(last_block, 'eca'):
            # Try to find the second last (usually IRB)
            # If stage3 has only one block and it's ECA, this would error, but PetNet design prevents this
            if len(model.stage3) > 1:
                target_block = model.stage3[-2]
            else:
                target_block = last_block  # Fallback strategy
        else:
            target_block = last_block

        # Find conv layer in IRB Block
        if hasattr(target_block, 'conv'):
            # IRB's conv is a Sequential, take the Conv before the last BatchNorm
            # Or directly take the whole conv module, GradCAM library will handle it
            return target_block.conv

        return target_block

    # For native MobileNetV2 (Baseline)
    elif hasattr(model, 'features'):
        return model.features[-1]

    return None

def create_concat_image(original, heatmap, overlay, labels=["Original", "Heatmap", "Overlay"]):
    """
    Horizontally stitch three images and add labels above
    """
    h, w, _ = original.shape

    # Set font and margins
    padding_top = 40  # Top padding for text
    padding_between = 10  # Image spacing
    total_w = w * 3 + padding_between * 2
    total_h = h + padding_top

    # Create white background canvas
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Place images
    canvas[padding_top:total_h, 0:w] = original
    canvas[padding_top:total_h, w + padding_between:w * 2 + padding_between] = heatmap
    canvas[padding_top:total_h, w * 2 + padding_between * 2:] = overlay

    # Convert to PIL for better text rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    # Try to load font, fallback to default if fails
    try:
        # Common paths for Linux/Mac, Windows may need adjustment
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # Draw text (centered)
    centers = [w // 2, w + padding_between + w // 2, w * 2 + padding_between * 2 + w // 2]

    for i, label in enumerate(labels):
        # Get text size (compatible with different PIL versions)
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)

        x = centers[i] - text_w // 2
        y = (padding_top - text_h) // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    return np.array(pil_img)


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM for PetNet')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--config', type=str, default='configs/petnet_base.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='gradcam_result.jpg', help='Output filename')
    parser.add_argument('--img-size', type=int, default=256, help='Input size (must match training)')
    # ✨ Added parameter: Enable concatenation mode
    parser.add_argument('--concat', action='store_true', help='Save as stitched image with labels')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load model ---
    print(f"🔍 Loading model configuration from {args.config}...")
    config = load_config(args.config)
    model_config = config['model']

    # Initialize model
    model = PetNet(
        num_classes=model_config.get('num_classes', 144),
        stage_repeats=model_config.get('stage_repeats', [2, 3, 4]),
        attn_cfg=model_config.get('attn_cfg', None),
        selfkd_cfg=model_config.get('selfkd_cfg', None),
        max_pets_per_image=10
    )

    # Load weights
    print(f"⬇️  Loading weights from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Filter mismatched keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(device)
    model.eval()

    # --- 2. Image preprocessing ---
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Error: Could not read image {args.image}")
        return
    # Resize - maintain original aspect ratio or force scaling? For CAM accuracy, recommend force scaling to model input size
    img_resized = cv2.resize(img, (args.img_size, args.img_size))
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    rgb_img_float = np.float32(rgb_img) / 255.0  # For visualization

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(rgb_img).unsqueeze(0).to(device)

    # --- 3. Grad-CAM computation ---
    target_layers = [get_target_layer(model)]
    print(f"?? Target Layer: {target_layers[0]}")

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

        # Generate overlay image
        overlay_img = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

        # Generate pure heatmap (Jet colormap)
        heatmap_only = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap_only = cv2.cvtColor(heatmap_only, cv2.COLOR_BGR2RGB)  # Convert back to RGB for matplotlib/PIL processing

    # --- 4. Save results ---
    if args.concat:
        # Concatenation mode: Original | Heatmap | Overlay
        print("🖼️  Stitching images with labels...")
        final_image = create_concat_image(
            original=rgb_img,
            heatmap=heatmap_only,
            overlay=overlay_img,
            labels=["Original Input", "Grad-CAM Heatmap", "Overlay Result"]
        )
        # Save
        Image.fromarray(final_image).save(args.output)
    else:
        # Default mode: Only save overlay image (or you can rename to save only heatmap)
        Image.fromarray(overlay_img).save(args.output)

    print(f"✅ Result saved to: {args.output}")


if __name__ == "__main__":
    main()