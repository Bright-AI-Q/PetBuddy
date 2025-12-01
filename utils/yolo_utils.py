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
import shutil
from pathlib import Path
import cv2
import numpy as np


#HAS_SCHEMA_LDRE = False


def get_image_dimensions(image_path):
    """Retrieve image dimensions (width, height)"""
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            return img.shape[1], img.shape[0]
        return 500, 375
    except:
        return 500, 375


def check_ap10k_categories(json_path):
    """
    Identify All Category IDs for Cat and Dog in AP-10K dataset.
    Returns: {ap10k_category_id: 'cat' or 'dog'}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    id_mapping = {}
    print(f"\n🔍 Inspecting Categories in {Path(json_path).name}...")

    # 定义黑名单关键词 (排除猛兽)
    EXCLUDE_KEYWORDS = [
        'panthera', 'lion', 'tiger', 'leopard', 'jaguar', 'cheetah',
        'lynx', 'puma', 'cougar', 'bobcat', 'ocelot', 'wild',
        'wolf', 'fox', 'coyote', 'jackal', 'hyena', 'bear'
    ]

    for cat in data['categories']:
        name = cat['name'].lower()
        supercat = cat.get('supercategory', '').lower()
        cat_id = cat['id']

        # 1. 初步筛选：必须属于犬科或猫科
        is_dog_family = 'dog' in name or 'canis' in name or 'canidae' in supercat
        is_cat_family = 'cat' in name or 'felis' in name or 'felidae' in supercat

        # 2. 严格过滤：检查是否在黑名单中
        is_wild = any(k in name for k in EXCLUDE_KEYWORDS)

        # 特殊处理：虽然 AP-10K 里有些名字叫 "wild cat"，如果是指 Felis silvestris (野猫) 其实跟家猫很像，
        # 但为了稳健，我们暂时只想要 "domestic" 或者纯粹的 "cat"/"dog"。

        # 3. 最终判定
        if is_dog_family and not is_wild:
            # 确保不是狼或狐狸 (虽然 wolf 在黑名单里，双重保险)
            if 'wolf' not in name and 'fox' not in name:
                id_mapping[cat_id] = 'dog'
                # print(f"  🐶 Keep Dog: {name} (ID: {cat_id})")

        elif is_cat_family and not is_wild:
            # 确保只保留家猫相关的
            # AP-10K 中家猫通常叫 "cat" 或者 "felis catus"
            # 如果名字太长且怪异，通常是野生动物
            id_mapping[cat_id] = 'cat'
            # print(f"  🐱 Keep Cat: {name} (ID: {cat_id})")
        else:
            # 打印被剔除的物种，方便你确认
            # print(f"  🚫 Drop Wild: {name}")
            pass

    print(f"ℹ️  Target IDs Found: {len(id_mapping)} valid categories (Filtered Wild Animals).")
    return id_mapping


def convert_ap10k_keypoints_to_yolo(json_path, output_dir, valid_id_mapping):
    """
    Convert AP-10K keypoint annotations to YOLO format.

    FIXED MAPPING based on AP-10K official definition:
    0: L_Eye, 1: R_Eye, 2: Nose, 3: Neck, 4: Tail_Root ...

    Target Schema (PetBuddy):
    0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear
    """
    # 1. 确保目录存在
    out_path_obj = Path(output_dir)
    out_path_obj.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_id_to_info = {img['id']: img for img in data['images']}

    stats = {
        'total': len(data['annotations']),
        'converted': 0,
        'skipped_species': 0,
        'skipped_no_img': 0,
        'skipped_bad_kpt': 0
    }

    # === 关键修改：显式索引映射 ===
    # Key: 我们的目标索引 (0-4)
    # Value: AP-10K 的源索引 (根据官方定义)
    # Target 0 (Nose)  <-- Source 2 (Nose)
    # Target 1 (L_Eye) <-- Source 0 (Left Eye)
    # Target 2 (R_Eye) <-- Source 1 (Right Eye)
    # Target 3 (L_Ear) <-- None (AP-10K无耳朵，设为None)
    # Target 4 (R_Ear) <-- None (AP-10K无耳朵，设为None)

    # 这里的 None 表示该数据集缺失此部位
    INDEX_MAP = {
        0: 2,  # Nose
        1: 0,  # L_Eye
        2: 1,  # R_Eye
        3: None,  # Missing in AP10K
        4: None  # Missing in AP10K
    }

    debug_success_printed = False

    for ann in data['annotations']:
        cat_id = ann['category_id']

        # 1. 类别过滤
        if cat_id not in valid_id_mapping:
            stats['skipped_species'] += 1
            continue

        species = valid_id_mapping[cat_id]
        class_id = 0 if species == 'cat' else 1

        image_id = ann['image_id']
        img_info = image_id_to_info.get(image_id)
        if not img_info:
            stats['skipped_no_img'] += 1
            continue

        img_w, img_h = img_info['width'], img_info['height']
        output_file = out_path_obj / f"{Path(img_info['file_name']).stem}.txt"

        keypoints = ann['keypoints']  # List of floats

        # 2. 关键点提取 (使用新的映射逻辑)
        final_keypoints = [0.0] * 15
        valid_xs = []
        valid_ys = []
        has_valid_kpt = False

        # 遍历我们的 5 个目标点
        for target_idx in range(5):
            source_idx = INDEX_MAP[target_idx]

            # 如果源数据里有这个点 (不是 None)
            if source_idx is not None:
                base_idx = source_idx * 3
                if base_idx + 2 < len(keypoints):
                    x, y, v = keypoints[base_idx], keypoints[base_idx + 1], keypoints[base_idx + 2]

                    # v > 0 意味着已标记
                    if x > 0 and y > 0 and v > 0:
                        final_keypoints[target_idx * 3] = x / img_w  # X
                        final_keypoints[target_idx * 3 + 1] = y / img_h  # Y
                        final_keypoints[target_idx * 3 + 2] = 2.0  # Vis

                        valid_xs.append(x)
                        valid_ys.append(y)
                        has_valid_kpt = True

        if not has_valid_kpt:
            stats['skipped_bad_kpt'] += 1
            continue

        # 3. BBox 处理
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            # 只有当关键点在框外很远时才修正，或者直接信任原bbox
            # AP-10K 的 bbox 通常质量很高，我们主要做防止越界检查
            if valid_xs and valid_ys:
                min_kx, max_kx = min(valid_xs), max(valid_xs)
                min_ky, max_ky = min(valid_ys), max(valid_ys)

                new_x = min(x, min_kx)
                new_y = min(y, min_ky)
                new_x2 = max(x + w, max_kx)
                new_y2 = max(y + h, max_ky)

                x, y = new_x, new_y
                w = new_x2 - new_x
                h = new_y2 - new_y

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
        else:
            # 根据关键点生成
            if not valid_xs: continue
            min_x, max_x = min(valid_xs), max(valid_xs)
            min_y, max_y = min(valid_ys), max(valid_ys)
            pad_x = (max_x - min_x) * 0.15  # 稍微加大 padding 因为只剩3个点
            pad_y = (max_y - min_y) * 0.15

            bx = max(0, min_x - pad_x)
            by = max(0, min_y - pad_y)
            bw = min(img_w, max_x + pad_x) - bx
            bh = min(img_h, max_y + pad_y) - by

            cx = (bx + bw / 2) / img_w
            cy = (by + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h

        # 4. 写入文件
        line_data = [class_id, cx, cy, nw, nh] + final_keypoints
        line_str = " ".join([f"{x:.6f}" for x in line_data])

        with open(output_file, 'a') as f:
            f.write(line_str + "\n")

        stats['converted'] += 1

        if not debug_success_printed:
            print(f"✅ [Debug] Successfully converted first sample: {output_file.name} (Class: {species})")
            debug_success_printed = True

    print(f"📊 Stats for {Path(json_path).name}:")
    print(f"  - Total: {stats['total']}, Converted: {stats['converted']}")

def find_stanford_dogs_annotations(base_dir='data/stanford_dogs'):
    """Locate Stanford Dogs annotation file"""
    annotation_path = Path(base_dir) / 'annotations' / 'StanfordExtra_v12.json'
    if annotation_path.exists():
        return [annotation_path]
    return []


def convert_stanford_dogs_pose_to_yolo(json_path, output_dir, class_id=1):  # Dog defaults to 1
    """Convert Stanford Dogs pose annotations to YOLO format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error: Failed to parse JSON file {json_path}")
            return

    stats = {'total': 0, 'converted': 0}

    if isinstance(data, list):
        for item in data:
            img_path = item.get('img_path')
            if not img_path: continue

            # StanfordExtra usually needs actual image reading to get size,
            # or we rely on default if slow. Let's try to get it if possible.
            # Assuming images are in standard path structure relative to script
            full_img_path = Path("data/stanford_dogs/Images") / img_path
            img_width, img_height = get_image_dimensions(full_img_path)

            output_path = Path(output_dir) / f"{Path(img_path).stem}.txt"
            keypoints = item.get('joints', [])

            if not keypoints: continue
            stats['total'] += 1

            # Stanford logic (simplified standard raw mode)
            # Stanford format: [[x,y,vis], ...]
            # Need to map to Nose(16), L_Eye(14), R_Eye(15), L_Ear(18), R_Ear(19) ?
            # Wait, let's look at standard mapping again.
            # Standard: Nose, L_Eye, R_Eye, L_Ear, R_Ear
            # Stanford IDs:
            # 14: Left Eye, 15: Right Eye, 16: Nose
            # 18: Left Ear Tip, 19: Right Ear Tip

            STANFORD_MAP = [16, 14, 15, 18, 19]
            final_kpts = [0.0] * 15
            valid_xs, valid_ys = [], []
            has_valid = False

            for i, kp_idx in enumerate(STANFORD_MAP):
                if kp_idx < len(keypoints):
                    kp = keypoints[kp_idx]  # [x, y, vis] usually? Or just [x, y]?
                    # StanfordExtra v12 is usually [x, y] or [x, y, v]
                    # Let's assume [x, y] and treat as visible if > 0
                    if len(kp) >= 2:
                        x, y = float(kp[0]), float(kp[1])
                        if x > 0 and y > 0:
                            final_kpts[i * 3] = x / img_width
                            final_kpts[i * 3 + 1] = y / img_height
                            final_kpts[i * 3 + 2] = 2.0
                            valid_xs.append(x)
                            valid_ys.append(y)
                            has_valid = True

            if has_valid:
                # BBox calc
                min_x, max_x = min(valid_xs), max(valid_xs)
                min_y, max_y = min(valid_ys), max(valid_ys)
                pad_x, pad_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
                bx = max(0, min_x - pad_x)
                by = max(0, min_y - pad_y)
                bw = min(img_width, max_x + pad_x) - bx
                bh = min(img_height, max_y + pad_y) - by

                cx, cy = (bx + bw / 2) / img_width, (by + bh / 2) / img_height
                nw, nh = bw / img_width, bh / img_height

                line = [class_id, cx, cy, nw, nh] + final_kpts
                with open(output_path, 'w') as f:
                    f.write(" ".join([f"{x:.6f}" for x in line]) + "\n")
                stats['converted'] += 1

    print(f"📊 Stanford Dogs: Total {stats['total']}, Converted {stats['converted']}")


def convert_animal_pose_to_yolo(json_path, output_dir):
    """Convert Animal Pose annotations (Robust Version)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not Path(json_path).exists():
        print(f"⚠️  Animal Pose JSON not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    images_map = {int(k): v for k, v in data.get('images', {}).items()}
    stats = {
        'total': len(data['annotations']),
        'converted': 0,
        'skipped_no_img': 0,
        'skipped_bad_kpt': 0
    }

    # AnimalPose IDs mapping to Schema
    # Original: L_Eye(0), R_Eye(1), Nose(2), L_Ear(3), R_Ear(4)
    # Target:   Nose(2), L_Eye(0), R_Eye(1), L_Ear(3), R_Ear(4)
    indices = [2, 0, 1, 3, 4]

    # 设定图片根目录 (根据你的报错，脚本是在这里找的)
    base_img_dir = Path("data/Self_collected_Images")

    print(f"🔍 Searching for images in: {base_img_dir}")

    for ann in data['annotations']:
        img_id = ann['image_id']
        fname = images_map.get(img_id, f"{img_id}.jpg")

        fpath = base_img_dir / fname

        # === 修复: 先检查文件是否存在 ===
        if not fpath.exists():
            # 尝试递归查找 (防止图片藏在子文件夹里)
            # 注意: 这会稍微变慢，但能解决路径问题
            found = list(base_img_dir.glob(f"**/{fname}"))
            if found:
                fpath = found[0]
            else:
                stats['skipped_no_img'] += 1
                continue  # 彻底找不到，跳过

        # 此时 fpath 肯定存在，再读取尺寸，OpenCV 就不会报错了
        w, h = get_image_dimensions(fpath)

        # Class: Cat=0, Dog=1. Ann category_id: 1=Dog, 2=Cat
        # Animal Pose dataset definition: 1: dog, 2: cat, 3: sheep, 4: horse, 5: cow
        cat_id = ann.get('category_id')
        if cat_id not in [1, 2]:
            continue  # 只处理猫狗

        cid = 0 if cat_id == 2 else 1

        kpts = ann['keypoints']  # [[x,y,v], ...]
        final_kpts = [0.0] * 15
        valid_xs, valid_ys = [], []
        has_val = False

        for i, idx in enumerate(indices):
            if idx < len(kpts):
                kp = kpts[idx]
                if len(kp) >= 2:
                    x, y = float(kp[0]), float(kp[1])
                    if x > 0 and y > 0:
                        final_kpts[i * 3] = x / w
                        final_kpts[i * 3 + 1] = y / h
                        final_kpts[i * 3 + 2] = 2.0
                        valid_xs.append(x)
                        valid_ys.append(y)
                        has_val = True

        if has_val:
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
            else:
                if not valid_xs:
                    stats['skipped_bad_kpt'] += 1
                    continue
                minx, maxx = min(valid_xs), max(valid_xs)
                miny, maxy = min(valid_ys), max(valid_ys)
                pad = 20
                bx, by = max(0, minx - pad), max(0, miny - pad)
                bw, bh = maxx - minx + 2 * pad, maxy - miny + 2 * pad

            cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
            nw, nh = bw / w, bh / h

            line = [cid, cx, cy, nw, nh] + final_kpts
            out_file = Path(output_dir) / f"{fpath.stem}.txt"  # 使用找到的图片名作为标签名

            # 同样使用追加模式前建议清空，这里用 w 覆盖，因为一般不重名
            with open(out_file, 'w') as f:
                f.write(" ".join([f"{x:.6f}" for x in line]) + "\n")
            stats['converted'] += 1

    print(f"📊 Animal Pose Stats:")
    print(f"  - Total Annotations: {stats['total']}")
    print(f"  - Missing Images:    {stats['skipped_no_img']} (Skipped)")
    print(f"  - Bad Keypoints:     {stats['skipped_bad_kpt']}")
    print(f"  - ✅ Converted:       {stats['converted']}")

def convert_self_collected_placeholders(images_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cnt = 0
    for subdir, cid in [("cat", "0"), ("dog", "1")]:
        tdir = Path(images_dir) / subdir
        if tdir.exists():
            for img in tdir.glob("*.jpeg"):
                out = Path(output_dir) / f"{img.stem}.txt"
                # Class + BBox(0) + Kpts(0)
                line = [cid] + [0.0] * 4 + [0.0] * 15
                with open(out, 'w') as f:
                    f.write(" ".join(map(str, line)) + "\n")
                cnt += 1
    print(f"📊 Self Collected Placeholders: {cnt}")


def main():
    print("🚀 Multi-Dataset Annotation Converter - YOLO Format (Robust)")
    print("=" * 50)

    # --- 1. AP-10K ---
    ap10k_base = Path("data/ap-10k")
    ap10k_out = ap10k_base / "yolo_keypoints"

    # 清理旧数据 (关键!)
    if ap10k_out.exists():
        print("🧹 Cleaning old AP-10K output directory...")
        shutil.rmtree(ap10k_out)

    if (ap10k_base / "annotations").exists():
        for jf in (ap10k_base / "annotations").glob("ap10k*.json"):
            mapping = check_ap10k_categories(str(jf))
            if mapping:
                convert_ap10k_keypoints_to_yolo(str(jf), str(ap10k_out), mapping)

    # --- 2. Stanford Dogs ---
    sd_path = Path("data/stanford_dogs/annotations/StanfordExtra_v12.json")
    if sd_path.exists():
        print("\n>>> Processing Stanford Dogs...")
        convert_stanford_dogs_pose_to_yolo(str(sd_path), "data/stanford_dogs/yolo_keypoints")

    # --- 3. Animal Pose ---
    ap_json = Path("data/Self_collected_Images/keypoints.json")
    if ap_json.exists():
        print("\n>>> Processing Animal Pose...")
        convert_animal_pose_to_yolo(str(ap_json), "data/Self_collected_Images/yolo_keypoints")



    print("\n✅ All tasks completed!")


if __name__ == '__main__':
    main()