import cv2
import numpy as np
from pathlib import Path
import json
import os

def check_raw_json_values():
    print("🕵️‍♂️ Detecting raw JSON values...\n")

    # 1. Check Stanford Dogs (specific image)
    target_img_stanford = "n02085620-Chihuahua/n02085620_10074.jpg"
    json_stanford = "data/stanford_dogs/annotations/StanfordExtra_v12.json"

    if os.path.exists(json_stanford):
        print(f"--- Checking Stanford Dogs: {target_img_stanford} ---")
        with open(json_stanford, 'r') as f:
            data = json.load(f)

        found = False
        for item in data:
            if item.get('img_path') == target_img_stanford:
                found = True
                joints = item.get('joints', [])
                # Index mapping: 16:Nose, 14:L_Eye, 15:R_Eye, 18:L_Ear, 19:R_Ear
                indices = {'Nose(16)': 16, 'L_Eye(14)': 14, 'R_Eye(15)': 15, 'L_Ear(18)': 18, 'R_Ear(19)': 19}

                print(f"  Original joints array length: {len(joints)}")
                for name, idx in indices.items():
                    if idx < len(joints):
                        val = joints[idx]
                        print(f"  📍 {name}: {val}")
                    else:
                        print(f"  ❌ {name}: Index out of range")
                break
        if not found:
            print("  ❌ Image not found in JSON")
    else:
        print("  ❌ Stanford JSON file not found")

    print("\n" + "=" * 30 + "\n")

    # 2. Check AP-10K (specific image)
    # Note: AP-10K image names in JSON usually don't include path, just filename
    target_img_ap10k = "000000008803.jpg"
    # We need to scan all JSON files to find the one containing this image
    ap10k_dir = "data/ap-10k/annotations"

    print(f"--- Checking AP-10K: {target_img_ap10k} ---")
    if os.path.exists(ap10k_dir):
        found_ap = False
        import glob
        for json_file in glob.glob(os.path.join(ap10k_dir, "*.json")):
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Find Image ID
            img_id = None
            for img in data['images']:
                if img['file_name'] == target_img_ap10k:
                    img_id = img['id']
                    print(f"  📂 Found image in {os.path.basename(json_file)}, ID: {img_id}")
                    break

            if img_id is not None:
                # Find Annotation
                for ann in data['annotations']:
                    if ann['image_id'] == img_id:
                        kpts = ann['keypoints']
                        # COCO: 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear
                        indices = {'Nose(0)': 0, 'L_Eye(1)': 1, 'R_Eye(2)': 2, 'L_Ear(3)': 3, 'R_Ear(4)': 4}

                        print(f"  Original keypoints array length: {len(kpts)}")
                        for name, idx in indices.items():
                            base = idx * 3
                            x, y, v = kpts[base], kpts[base + 1], kpts[base + 2]
                            status = "visible" if v == 2 else "occluded" if v == 1 else "not labeled(0)"
                            print(f"  📍 {name}: x={x}, y={y}, v={v} ({status})")
                        found_ap = True
                break  # Stop scanning files once found

        if not found_ap:
            print("  ❌ No annotation found for this image")
    else:
        print("  ❌ AP-10K annotation directory not found")


def process_single_image(dataset_name, img_path, label_path, target_height=500):
    """
    Process single image and its label, draw visualization, and resize to target height.
    """
    print(f"   🔹 Processing: {dataset_name} ...")

    # 1. Check files
    if not Path(img_path).exists():
        print(f"      ❌ Image missing: {img_path}")
        return None
    if not Path(label_path).exists():
        print(f"      ❌ Label missing: {label_path}")
        # We could still draw the image without label, but here we return None
        return None

    # 2. Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print("      ❌ OpenCV failed to read image")
        return None

    h_orig, w_orig = img.shape[:2]

    # 3. Read and draw labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    COLOR_BOX = (0, 255, 0)  # Green box
    COLOR_KPT = (0, 0, 255)  # Red keypoints
    COLOR_TEXT = (255, 255, 0)  # Cyan text

    has_draw = False

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue

        # Parse Class ID (compatible with float string)
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue

        # Parse BBox (cx, cy, w, h)
        cx, cy, bw, bh = map(float, parts[1:5])

        x1 = int((cx - bw / 2) * w_orig)
        y1 = int((cy - bh / 2) * h_orig)
        x2 = int((cx + bw / 2) * w_orig)
        y2 = int((cy + bh / 2) * h_orig)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BOX, 2)
        label_text = f"ID:{cls_id}"
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BOX, 2)
        has_draw = True

        # Parse keypoints (if exists)
        if len(parts) > 5:
            kpts = parts[5:]
            num_kpts = len(kpts) // 3
            for j in range(num_kpts):
                idx = j * 3
                px = float(kpts[idx])
                py = float(kpts[idx + 1])
                vis = float(kpts[idx + 2])

                if vis > 0 and (px > 0 or py > 0):
                    cx_k = int(px * w_orig)
                    cy_k = int(py * h_orig)
                    cv2.circle(img, (cx_k, cy_k), 5, COLOR_KPT, -1)
                    # Optional: draw point index
                    # cv2.putText(img, str(j), (cx_k+5, cy_k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 4. Add dataset name label
    cv2.putText(img, dataset_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)  # Black outline
    cv2.putText(img, dataset_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2)

    # 5. Resize image to target height (for easier combination)
    scale = target_height / h_orig
    new_w = int(w_orig * scale)
    resized_img = cv2.resize(img, (new_w, target_height))

    # 6. Add black border as separator
    resized_img = cv2.copyMakeBorder(resized_img, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=(50, 50, 50))

    if not has_draw:
        print("      ⚠️ Warning: Label file has content but no valid data was parsed")

    return resized_img


def main():


    print("🚀 Starting batch diagnosis and stitching...")

    # ================= ⚙️ Configuration =================
    # Format: "Dataset Name": ("Image Path", "Label Path")
    # Please modify paths according to your actual files

    tasks = {
        "AP-10K": (
            "data/ap-10k/data/000000008803.jpg",
            "data/ap-10k/yolo_keypoints/000000008803.txt"
        ),
        "Stanford Dogs": (
            # Note: Make sure this image exists in your Images directory
            "data/stanford_dogs/Images/n02085620-Chihuahua/n02085620_199.jpg",
            "data/stanford_dogs/yolo_keypoints/n02085620_199.txt"
        ),
        "Animal Pose": (
            # Note: Make sure this image exists in data/Self_collected_Images
            "data/Self_collected_Images/cat/ca1.jpeg",
            "data/Self_collected_Images/yolo_keypoints/ca1.txt"
        )

    }

    OUTPUT_FILE = "data/debug_combined_result.jpg"
    # ===============================================

    check_raw_json_values()
    processed_images = []

    for name, (img_p, lbl_p) in tasks.items():
        # Skip if placeholder or file doesn't exist (no error raised)
        res = process_single_image(name, img_p, lbl_p)
        if res is not None:
            processed_images.append(res)

    if not processed_images:
        print("\n❌ No images processed successfully, please check path configuration!")
        return

    # Stitch images (horizontal)
    print("\n🧩 Stitching images...")
    try:
        combined = np.hstack(processed_images)

        # Ensure output directory exists
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(OUTPUT_FILE, combined)

        print(f"🎉 Success! Combined diagnostic image saved to: {OUTPUT_FILE}")
        print(f"   Image dimensions: {combined.shape[1]}x{combined.shape[0]}")
    except Exception as e:
        print(f"❌ Image stitching failed: {e}")


if __name__ == "__main__":
    main()