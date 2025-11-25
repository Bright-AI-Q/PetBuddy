#!/usr/bin/env python3
"""
YOLO Format Annotation Visualizer
---------------------------------
Author: Bright Wang
Project: PetBuddy

Purpose:
1. Read YOLO format annotation files (.txt)
2. Draw bounding boxes and class labels on corresponding images
3. Support saving visualization results or direct display

Usage:
python3 visualize_yolo.py --img_dir [image_directory] --label_dir [label_directory] --output_dir [output_directory](optional)
example:python utils/visualize_yolo.py --img_dir data/merged_pet_dataset/images/train --label_dir data/merged_pet_dataset/labels/train
"""
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

# Class name mapping (adjust according to your project)
CLASS_NAMES = ['cat', 'dog']

def visualize_yolo(img_path, label_path, output_path=None):
    """
    Draw YOLO format bounding boxes on image

    Args:
        img_path (Path): Image file path
        label_path (Path): YOLO label file path
        output_path (Path, optional): Output image path
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return

    img_h, img_w = img.shape[:2]

    # Read label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, xc, yc, w, h = map(float, parts)
        cls_id = int(cls_id)

        # Convert to absolute coordinates
        x1 = int((xc - w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w)
        y2 = int((yc + h/2) * img_h)

        # Draw bounding box - increased border width to 4 for better visibility
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Green for cat, Red for dog
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

        # Add class label - increased font size to 1.5 and thickness to 3
        label = f"{CLASS_NAMES[cls_id]}"
        cv2.putText(img, label, (x1, y1 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Save result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

def main():
    parser = argparse.ArgumentParser(description='YOLO Format Annotation Visualizer')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='Directory containing YOLO labels')
    parser.add_argument('--output_dir', type=str, default='data/yolo',
                       help='Directory to save visualization results (default: data/yolo)')

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    img_files = list(img_dir.glob('*.jpg'))
    total_files = len(img_files)

    if total_files == 0:
        print("Error: No .jpg files found in image directory")
        return

    print(f"Found {total_files} image files, starting processing...")

    # Process each image with progress bar
    processed_count = 0
    for img_path in tqdm(img_files, desc="Processing images", unit="images"):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        output_path = output_dir / img_path.name

        visualize_yolo(img_path, label_path, output_path)
        processed_count += 1

    print(f"Processing completed! Processed {processed_count}/{total_files} images")
    if processed_count < total_files:
        print(f"Note: {total_files - processed_count} images have no corresponding label files")

if __name__ == "__main__":
    main()