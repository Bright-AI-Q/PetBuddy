#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: utils/yolo_utils.py
Location: utils/
====================================
Multi-Dataset Annotation Converter for YOLO Format

Purpose:
- Normalize various pet datasets into a unified YOLO format
- Support bounding box and keypoint conversion
- Handle specific mappings for AP-10K, Stanford Dogs, and Animal Pose datasets

Key Features:
1. AP-10K Support: Extracts cat/dog categories and converts COCO-style keypoints
2. Stanford Dogs Support: Adapts StanfordExtra dataset annotations
3. SchemaLDRE Integration: Maps diverse keypoint definitions to a unified 5-point schema
4. Standardization: Uniforms file structure and coordinate normalization
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np

# Check for SchemaLDRE support
try:
    from models.modules.ldre import convert_to_schema_ldre, visualize_schema_ldre_keypoints, SCHEMA_LDRE

    HAS_SCHEMA_LDRE = True
except ImportError:
    HAS_SCHEMA_LDRE = False
    print("⚠️ SchemaLDRE module not found. Using full keypoints or skipping specific conversions.")


def get_image_dimensions(image_path):
    """Retrieve image dimensions (width, height)"""
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            return img.shape[1], img.shape[0]
        return 500, 375  # Default fallback
    except:
        return 500, 375


def check_ap10k_categories(json_path):
    """Identify Category IDs for Cat and Dog in AP-10K dataset"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    target_ids = {}
    for cat in data['categories']:
        name = cat['name'].lower()
        supercat = cat['supercategory'].lower()

        if 'dog' in name or 'canis' in name or 'canidae' in supercat:
            # print(f"🐶 Found Dog Family: ID={cat['id']}, Name={cat['name']}, Family={cat['supercategory']}")
            target_ids['dog'] = cat['id']

        if 'cat' in name or 'felis' in name or 'felidae' in supercat:
            if 'catus' in name or 'cat' == name or 'domestic' in name:
                # print(f"🐱 Found Domestic Cat: ID={cat['id']}, Name={cat['name']}, Family={cat['supercategory']}")
                target_ids['cat'] = cat['id']

    return target_ids


def convert_ap10k_cat_and_dog_to_yolo(json_path, output_dir, categories):
    """Convert AP-10K Cat/Dog bounding box annotations to YOLO format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    stats = {
        'total_images': 0,
        'converted_images': 0,
        'dog_annotations': 0,
        'cat_annotations': 0
    }

    class_mapping = {v: 0 if k == 'cat' else 1 for k, v in categories.items()}

    for img in data['images']:
        img_id = img['id']
        img_path = Path(output_dir) / f"{img['file_name'].split('.')[0]}.txt"
        stats['total_images'] += 1

        annotations = [
            ann for ann in data['annotations']
            if ann['image_id'] == img_id and ann['category_id'] in categories.values()
        ]

        if not annotations:
            continue

        stats['converted_images'] += 1
        yolo_annotations = []

        for ann in annotations:
            category_id = ann['category_id']
            class_id = class_mapping[category_id]
            bbox = ann['bbox']
            xmin, ymin, width, height = bbox

            img_width = img.get('width', 500)
            img_height = img.get('height', 375)

            x_center = (xmin + width / 2) / img_width
            y_center = (ymin + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height

            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

            if class_id == 0:
                stats['cat_annotations'] += 1
            else:
                stats['dog_annotations'] += 1

        with open(img_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    print("\n📊 AP-10K BBox Conversion Stats:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Converted: {stats['converted_images']}")
    print(f"Cat Annotations: {stats['cat_annotations']}")
    print(f"Dog Annotations: {stats['dog_annotations']}")


def find_stanford_dogs_annotations(base_dir='data/stanford_dogs'):
    """Locate Stanford Dogs annotation file"""
    annotation_path = Path(base_dir) / 'annotations' / 'StanfordExtra_v12.json'
    if annotation_path.exists():
        return [annotation_path]
    return []


def convert_stanford_dogs_pose_to_yolo(json_path, output_dir, class_id=0, images_dir=None, use_schema_ldre=True):
    """Convert Stanford Dogs pose annotations to YOLO format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error: Failed to parse JSON file {json_path} - {str(e)}")
            return

    stats = {
        'total_images': 0,
        'converted_images': 0,
        'total_keypoints': 0
    }

    # Handle StanfordExtra format (List of dicts)
    if isinstance(data, list):
        processed_count = 0
        skipped_count = 0
        for item in data:
            img_path = item.get('img_path')
            if not img_path:
                skipped_count += 1
                continue

            # Attempt to get actual image dimensions
            img_width = item.get('img_width', 500)
            img_height = item.get('img_height', 375)

            if images_dir:
                full_img_path = Path(images_dir) / img_path
                if full_img_path.exists():
                    try:
                        img = cv2.imread(str(full_img_path))
                        if img is not None:
                            img_width, img_height = img.shape[1], img.shape[0]
                    except:
                        pass

            output_path = Path(output_dir) / f"{Path(img_path).stem}.txt"
            keypoints = item.get('joints', [])

            if not keypoints or len(keypoints) < 24:
                skipped_count += 1
                continue

            stats['total_images'] += 1

            # SchemaLDRE Mode
            if use_schema_ldre and HAS_SCHEMA_LDRE:
                schema_ldre_keypoints = [0.0] * 10  # 5 keypoints * 2 coordinates

                # Mapping: Stanford Dogs indices to SchemaLDRE
                # Schema: nose, left_eye, right_eye, left_ear_tip, right_ear_tip
                # Indices based on StanfordExtra definition
                STANFORD_TO_LDRE = [16, 14, 15, 18, 19]

                schema_keypoint_count = 0
                for i, kp_idx in enumerate(STANFORD_TO_LDRE):
                    if kp_idx < len(keypoints):
                        kp = keypoints[kp_idx]
                        if len(kp) >= 3:
                            x, y, visible = kp[0], kp[1], kp[2]
                            if visible == 1 and x > 0 and y > 0:
                                x = np.clip(x, 0, img_width)
                                y = np.clip(y, 0, img_height)
                                schema_ldre_keypoints[i * 2] = x / img_width
                                schema_ldre_keypoints[i * 2 + 1] = y / img_height
                                schema_keypoint_count += 1

                # Write to file
                if schema_keypoint_count > 0:
                    yolo_kps = [str(class_id)] + [f"{coord:.6f}" for coord in schema_ldre_keypoints]
                    with open(output_path, 'w') as f:
                        f.write(" ".join(yolo_kps) + "\n")
                    stats['converted_images'] += 1
                    stats['total_keypoints'] += schema_keypoint_count
                    processed_count += 1
                else:
                    skipped_count += 1
            else:
                # Raw Mode: Output all visible keypoints
                parsed_kps = []
                for kp in keypoints:
                    if len(kp) >= 3:
                        x, y, visible = kp[0], kp[1], kp[2]
                        if visible == 1 and x > 0 and y > 0:
                            parsed_kps.append((x, y))

                if not parsed_kps:
                    skipped_count += 1
                    continue

                stats['converted_images'] += 1
                stats['total_keypoints'] += len(parsed_kps)
                yolo_kps = []

                for x, y in parsed_kps:
                    x = np.clip(x, 0, img_width)
                    y = np.clip(y, 0, img_height)
                    x_norm = x / img_width
                    y_norm = y / img_height
                    yolo_kps.append(f"{class_id} {x_norm:.6f} {y_norm:.6f}")

                with open(output_path, 'w') as f:
                    for kp in yolo_kps:
                        f.write(f"{kp}\n")

                processed_count += 1

        print(f"\n📊 Stanford Dogs Conversion Stats:")
        print(f"Total Images: {stats['total_images']}")
        print(f"Converted: {stats['converted_images']}")
        print(f"Processed: {processed_count}, Skipped: {skipped_count}")


def convert_ap10k_keypoints_to_yolo(json_path, output_dir, use_schema_ldre=True):
    """Convert AP-10K COCO-style keypoint annotations to YOLO format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_id_to_info = {img['id']: img for img in data['images']}

    stats = {
        'total_annotations': len(data['annotations']),
        'converted_annotations': 0,
        'total_keypoints': 0
    }

    # SchemaLDRE indices: nose(3), left_eye(1), right_eye(2), left_ear(14), right_ear(17)
    schema_indices = [3, 1, 2, 14, 17]

    for ann in data['annotations']:
        image_id = ann['image_id']
        img_info = image_id_to_info.get(image_id)
        if not img_info: continue

        img_w, img_h = img_info['width'], img_info['height']
        output_path = Path(output_dir) / f"{Path(img_info['file_name']).stem}.txt"

        keypoints = ann['keypoints']
        yolo_keypoints = []

        if use_schema_ldre and HAS_SCHEMA_LDRE:
            schema_kps = [0.0] * 10
            valid_cnt = 0
            for i, kp_idx in enumerate(schema_indices):
                if kp_idx * 3 < len(keypoints):
                    x, y, vis = float(keypoints[kp_idx * 3]), float(keypoints[kp_idx * 3 + 1]), float(
                        keypoints[kp_idx * 3 + 2])
                    if vis > 0 and x > 0 and y > 0:
                        schema_kps[i * 2] = x / img_w
                        schema_kps[i * 2 + 1] = y / img_h
                        valid_cnt += 1

            if valid_cnt > 0:
                yolo_keypoints = schema_kps

        if yolo_keypoints:
            # AP10K is typically treated as a single class or by ID. Setting to 0 here.
            line = ["0"] + [f"{coord:.6f}" for coord in yolo_keypoints]
            with open(output_path, 'w') as f:
                f.write(" ".join(line) + "\n")
            stats['converted_annotations'] += 1

    print(f"\n📊 AP-10K Keypoint Conversion Stats:")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"Converted: {stats['converted_annotations']}")


def convert_animal_pose_to_yolo(json_path, output_dir, use_schema_ldre=True):
    """Convert Animal Pose dataset annotations to YOLO format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Mapping: img_id -> filename
    images = {int(img_id): file_name for img_id, file_name in data['images'].items()} if 'images' in data else {}

    stats = {
        'total_annotations': len(data['annotations']),
        'converted_annotations': 0,
        'schema_ldre_annotations': 0
    }

    # Attempt to read image dimensions
    img_dimensions = {}
    if 'images' in data:
        for img_id, img_name in data['images'].items():
            img_path = Path("data/Self_collected_Images") / img_name
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_dimensions[int(img_id)] = (img.shape[1], img.shape[0])
                except:
                    pass

    for ann in data['annotations']:
        image_id = ann['image_id']
        img_name = images.get(image_id, f"{image_id}.jpg")
        img_w, img_h = img_dimensions.get(image_id, (300, 225))

        category_id = ann.get('category_id', 0)
        class_id = 0 if category_id == 2 else 1  # Cat=0, Dog=1

        output_path = Path(output_dir) / f"{Path(img_name).stem}.txt"
        keypoints = ann['keypoints']
        yolo_keypoints = []

        if use_schema_ldre:
            schema_ldre_keypoints = [0.0] * 10
            # Animal Pose: 0:L_Eye, 1:R_Eye, 2:Nose, 3:L_Ear, 4:R_Ear
            # SchemaLDRE: Nose, L_Eye, R_Eye, L_Ear, R_Ear
            schema_indices = [2, 0, 1, 3, 4]

            valid_cnt = 0
            for i, kp_idx in enumerate(schema_indices):
                if isinstance(keypoints, list) and len(keypoints) > kp_idx:
                    kp = keypoints[kp_idx]
                    if isinstance(kp, list) and len(kp) >= 3:
                        x, y, vis = float(kp[0]), float(kp[1]), float(kp[2])
                        if vis > 0 and x > 0 and y > 0:
                            schema_ldre_keypoints[i * 2] = x / img_w
                            schema_ldre_keypoints[i * 2 + 1] = y / img_h
                            valid_cnt += 1

            if valid_cnt > 0:
                yolo_keypoints = schema_ldre_keypoints

        if yolo_keypoints:
            line = [str(class_id)] + [f"{coord:.6f}" for coord in yolo_keypoints]
            with open(output_path, 'w') as f:
                f.write(" ".join(line) + "\n")
            stats['converted_annotations'] += 1
            if use_schema_ldre:
                stats['schema_ldre_annotations'] += 1

    print(f"\n📊 Animal Pose Conversion Stats:")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"Converted: {stats['converted_annotations']}")


def convert_self_collected_images_to_yolo(images_dir, output_dir):
    """Generate empty YOLO keypoint placeholders for self-collected images"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats = {'total_images': 0, 'converted_images': 0}
    print(f"Processing self-collected images in: {images_dir}")

    for subdir, class_id in [("cat", "0"), ("dog", "1")]:
        target_dir = Path(images_dir) / subdir
        if target_dir.exists():
            for img_file in target_dir.glob("*.jpeg"):  # Extension might need adjustment
                output_path = Path(output_dir) / f"{img_file.stem}.txt"

                # Create zero-filled keypoints
                empty_keypoints = [0.0] * 10
                line = [class_id] + [f"{coord:.6f}" for coord in empty_keypoints]
                with open(output_path, 'w') as f:
                    f.write(" ".join(line) + "\n")

                stats['total_images'] += 1
                stats['converted_images'] += 1

    print(f"\n📊 Self-Collected Images Stats:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Converted: {stats['converted_images']}")


def visualize_yolo_keypoints(img_path, label_path, output_path=None, use_schema_ldre=False):
    """Visualization utility"""
    img = cv2.imread(str(img_path))
    if img is None: return
    h, w = img.shape[:2]

    if not Path(label_path).exists(): return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue

        cls_id = parts[0]
        keypoints = [float(x) for x in parts[1:]]

        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i] * w), int(keypoints[i + 1] * h)
            if x > 0 and y > 0:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    if output_path:
        cv2.imwrite(str(output_path), img)


def main():
    print("🚀 Multi-Dataset Annotation Converter - YOLO Format")
    print("=" * 50)

    # 1. AP10K (BBox)
    ap10k_path = Path("data/ap-10k/annotations")
    if ap10k_path.exists():
        print("\n>>> Processing AP10K (BBox)...")
        for json_file in ap10k_path.glob("ap10k*.json"):
            categories = check_ap10k_categories(str(json_file))
            if categories:
                convert_ap10k_cat_and_dog_to_yolo(str(json_file), "data/ap-10k/yolo", categories)

    # 2. AP10K (Keypoints)
    if ap10k_path.exists():
        print("\n>>> Processing AP10K (Keypoints)...")
        for json_file in ap10k_path.glob("ap10k*.json"):
            convert_ap10k_keypoints_to_yolo(str(json_file), "data/ap-10k/yolo_keypoints")

    # 3. Stanford Dogs
    print("\n>>> Processing Stanford Dogs...")
    stanford_anns = find_stanford_dogs_annotations()
    if stanford_anns:
        for json_path in stanford_anns:
            convert_stanford_dogs_pose_to_yolo(
                str(json_path),
                "data/stanford_dogs/yolo_keypoints",
                images_dir="data/stanford_dogs/Images"
            )
    else:
        print("⚠️ Stanford Dogs annotations not found")

    # 4. Animal Pose
    print("\n>>> Processing Animal Pose (Self-collected)...")
    animal_pose_json = "data/Self_collected_Images/keypoints.json"
    if os.path.exists(animal_pose_json):
        convert_animal_pose_to_yolo(animal_pose_json, "data/self_collected/yolo_keypoints")
    else:
        print("⚠️ Animal Pose annotations not found")

    # 5. Self-collected Images (Empty Placeholders)
    print("\n>>> Processing Self-collected Images (Placeholders)...")
    self_imgs_dir = "data/Self_collected_Images"
    if os.path.exists(self_imgs_dir):
        convert_self_collected_images_to_yolo(self_imgs_dir, "data/Self_collected_Images/yolo_keypoints")

    print("\n✅ All tasks completed!")


if __name__ == '__main__':
    main()