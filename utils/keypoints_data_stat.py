import cv2
import numpy as np
from pathlib import Path
import os
from collections import defaultdict

# configs
DATASETS_CONFIG = {
    "AP-10K": {
        "label_dir": "data/ap-10k/yolo_keypoints",
        "image_root": "data/ap-10k/data",
        "exts": [".jpg", ".jpeg"]
    },
    "Stanford Dogs": {
        "label_dir": "data/stanford_dogs/yolo_keypoints",
        "image_root": "data/stanford_dogs/Images",
        "exts": [".jpg", ".jpeg"]
    },
    "Animal Pose": {
        "label_dir": "data/Self_collected_Images/yolo_keypoints",
        "image_root": "data/Self_collected_Images",
        "exts": [".jpg", ".jpeg", ".png"]
    }
}
OUTPUT_IMG = "data/quality_analysis_result.jpg"


# ===========================================

def build_image_map(root_dir, exts):
    """
    Recursively traverse the image directory and establish a mapping of {filename (without suffix): full path}
    """
    print(f"   📂 indexing images: {root_dir} ...")
    img_map = {}
    root = Path(root_dir)
    if not root.exists():
        return img_map

    for ext in exts:
        # 递归查找所有图片
        for p in root.rglob(f"*{ext}"):
            img_map[p.stem] = p

    return img_map


def analyze_dataset(name, config):
    """
    Analyze the label quality of a single dataset and return statistical information and paths to the best samples

    """
    print(f"\n📊 analysize dataset {name}")
    label_dir = Path(config["label_dir"])

    if not label_dir.exists():
        print(f"   ❌ labels doesn't exit: {label_dir}")
        return None

    # 1. build image map
    img_map = build_image_map(config["image_root"], config["exts"])
    if not img_map:
        print("   ❌ No images found, skipping。")
        return None

    # 2. 2. Statistical Data Container
    # Format: {point count (0-5): [ (img_path, label_path), ... ] }

    quality_bins = defaultdict(list)
    total_files = 0

    # 3. iterate over all label files
    for label_file in label_dir.glob("*.txt"):
        stem = label_file.stem

        # find
        if stem not in img_map:
            continue

        img_path = img_map[stem]
        total_files += 1

        # read label file
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()


            if not lines: continue

            parts = lines[0].strip().split()
            if len(parts) < 6: continue

            kpts = parts[5:]
            valid_cnt = 0


            for i in range(0, len(kpts), 3):
                if i + 2 < len(kpts):

                    v = float(kpts[i + 2])
                    x = float(kpts[i])
                    y = float(kpts[i + 1])
                    if v > 0 and (x > 0 or y > 0):
                        valid_cnt += 1

            # categorize
            quality_bins[valid_cnt].append((img_path, label_file))

        except Exception as e:
            continue

    return {
        "total": total_files,
        "bins": quality_bins
    }


def draw_sample(name, img_path, label_path, kpt_count, target_height=500):

    img = cv2.imread(str(img_path))
    if img is None: return None

    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        line = f.readline().strip()  # 只画第一只

    parts = line.split()


    cx, cy, bw, bh = map(float, parts[1:5])
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


    kpts = parts[5:]
    for i in range(0, len(kpts), 3):
        px, py, v = map(float, kpts[i:i + 3])
        if v > 0:
            cx_k, cy_k = int(px * w), int(py * h)
            cv2.circle(img, (cx_k, cy_k), 5, (0, 0, 255), -1)


    cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)


    cv2.putText(img, name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    color_score = (0, 255, 255) if kpt_count == 5 else (0, 165, 255)
    cv2.putText(img, f"Pts: {kpt_count}/5", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_score, 2)


    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_height))


    return cv2.copyMakeBorder(resized, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=(50, 50, 50))


def main():
    print("🚀 Starting automatic data quality analysis...")
    print("=" * 60)
    print(f"{'Dataset':<15} | {'Total':<8} | {'5 Pts':<8} | {'4 Pts':<8} | {'3 Pts':<8} | {'<3 Pts':<8}")
    print("-" * 65)

    vis_images = []

    for name, config in DATASETS_CONFIG.items():
        result = analyze_dataset(name, config)

        if not result:
            print(f"{name:<15} | ❌ (Missing Path or Files)")
            continue

        bins = result['bins']
        total = result['total']

        c5 = len(bins[5])
        c4 = len(bins[4])
        c3 = len(bins[3])
        c_low = sum(len(bins[k]) for k in bins if k < 3)

        # 1. Print statistics row
        print(f"{name:<15} | {total:<8} | {c5:<8} | {c4:<8} | {c3:<8} | {c_low:<8}")

        # 2. Select best sample for visualization
        # Prioritize 5 points, then 4 points, etc.
        best_sample = None
        best_score = 0

        for score in range(5, -1, -1):
            if bins[score]:
                # Take middle sample for diversity
                idx = len(bins[score]) // 2
                best_sample = bins[score][idx]
                best_score = score
                break

        if best_sample:
            img_p, lbl_p = best_sample
            vis_img = draw_sample(name, img_p, lbl_p, best_score)
            if vis_img is not None:
                vis_images.append(vis_img)
        else:
            print(f"   ⚠️ {name} has no valid samples to visualize")

    print("=" * 60)

    # 3. Combine images
    if vis_images:
        print(f"\n🧩 Combining {len(vis_images)} best samples...")
        try:
            combined = np.hstack(vis_images)
            Path(OUTPUT_IMG).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(OUTPUT_IMG, combined)
            print(f"🎉 Success! Quality report saved to: {OUTPUT_IMG}")
            print(f"   Open the image to view 'perfect' samples from each dataset.")
        except Exception as e:
            print(f"❌ Failed to combine images (possible size mismatch): {e}")
    else:
        print("❌ No visualization images were generated.")


if __name__ == "__main__":
    main()