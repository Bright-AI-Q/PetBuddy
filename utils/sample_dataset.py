#!/usr/bin/env python3
"""
Randomly sample images from the merged pet dataset and display them with their labels.
"""
import os
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_yolo_labels(label_path):
    """Load YOLO format labels"""
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, x_center, y_center, width, height = map(float, parts)
                    labels.append({
                        'class_id': int(cls_id),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'class_name': 'cat' if int(cls_id) == 0 else 'dog'
                    })
    return labels

def draw_bounding_boxes(image_path, labels, output_path=None):
    """Draw bounding boxes on image and save/display"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    img_height, img_width = img.shape[:2]

    # Convert to PIL for easier drawing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Draw bounding boxes
    for label in labels:
        # Convert YOLO coordinates to pixel coordinates
        x_center = label['x_center'] * img_width
        y_center = label['y_center'] * img_height
        width = label['width'] * img_width
        height = label['height'] * img_height

        # Calculate bounding box coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

        # Draw label
        label_text = f"{label['class_name']}"
        draw.text((x1, y1 - 15), label_text, fill='red')

    if output_path:
        # Save image
        pil_img.save(output_path)
        return output_path
    else:
        # Convert back to OpenCV format for display
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def sample_dataset(dataset_path, num_samples=10, output_dir=None, display=False):
    """Randomly sample images from dataset"""
    dataset_path = Path(dataset_path)

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist")
        return

    # Find all image files
    image_files = []
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        if images_dir.exists():
            image_files.extend(list(images_dir.glob('*.jpg')))

    if not image_files:
        print("No images found in the dataset")
        return

    # Randomly sample images
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"Sampling {len(sampled_images)} images from dataset:")
    print("=" * 50)

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for img_path in sampled_images:
        # Find corresponding label file
        label_path = None
        for split in ['train', 'val', 'test']:
            possible_label_path = (dataset_path / 'labels' / split /
                                 img_path.name.replace('.jpg', '.txt'))
            if possible_label_path.exists():
                label_path = possible_label_path
                break

        labels = load_yolo_labels(label_path) if label_path else []

        print(f"Image: {img_path.name}")
        print(f"Labels: {len(labels)}")
        for label in labels:
            print(f"  - {label['class_name']}: ({label['x_center']:.3f}, {label['y_center']:.3f}, "
                  f"{label['width']:.3f}, {label['height']:.3f})")
        print()

        # Draw bounding boxes if output directory specified
        if output_dir:
            output_image_path = output_dir / f"annotated_{img_path.name}"
            draw_bounding_boxes(img_path, labels, output_image_path)
            results.append({
                'original_image': str(img_path),
                'annotated_image': str(output_image_path),
                'labels': labels
            })

        # Display image if requested
        if display:
            annotated_img = draw_bounding_boxes(img_path, labels)
            if annotated_img is not None:
                cv2.imshow(f"Sample: {img_path.name}", annotated_img)
                cv2.waitKey(2000)  # Display for 2 seconds
                cv2.destroyAllWindows()

    # Save results to JSON file
    if output_dir:
        import json
        results_file = output_dir / "sampling_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Sample images from merged pet dataset')
    parser.add_argument('--dataset', '-d', default='data/merged_pet_dataset',
                       help='Path to merged dataset directory')
    parser.add_argument('--samples', '-n', type=int, default=10,
                       help='Number of samples to display')
    parser.add_argument('--output', '-o',
                       help='Output directory to save annotated images')
    parser.add_argument('--display', action='store_true',
                       help='Display images with OpenCV')

    args = parser.parse_args()

    sample_dataset(args.dataset, args.samples, args.output, args.display)

if __name__ == "__main__":
    main()