#!/usr/bin/env python3
"""
Dataset Merger for PetBuddy
---------------------------------
Author: Bright Wang
Project: PetBuddy

Purpose:
1. Merge multiple pet datasets into unified YOLO/COCO format
2. Support Oxford-IIIT Pets, Stanford Dogs, COCO, and CC0 datasets
3. Generate weak annotations using YOLOv8 pre-trained model
4. Provide automatic train/val/test split with deduplication

Datasets:
- Oxford-IIIT Pets: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/
- COCO: https://cocodataset.org/
- CC0: https://cc0textures.com/

Output:
- Merged dataset in YOLO format
- images/train, images/val, images/test directories
- labels/train, labels/val, labels/test directories
- dataset.yaml configuration file

Usage:
    python3 merge_detect_data.py
"""
import os, sys, shutil, json, csv, hashlib, argparse
from pathlib import Path
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from ultralytics import YOLO

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def yolo_to_coco_class(yolo_class):
    """Convert YOLOv8 class ID to standard COCO class ID

    Args:
        yolo_class: YOLOv8 class ID (15=cat, 16=dog)

    Returns:
        Standard COCO class ID (17=cat, 18=dog)
    """
    if yolo_class == 15:  # YOLOv8 cat
        return 17  # COCO cat
    elif yolo_class == 16:  # YOLOv8 dog
        return 18  # COCO dog
    else:
        return yolo_class  # Return original if not cat/dog

def coco2yolo(anns, img_w, img_h):
    """Convert COCO format annotations to YOLO format

    Args:
        anns: List of COCO annotation dictionaries
        img_w: Image width for normalization
        img_h: Image height for normalization

    Returns:
        List of YOLO format annotation strings
    """
    yolo_lines = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        xc = (x + w / 2) / img_w
        yc = (y + h / 2) / img_h
        w, h = w / img_w, h / img_h
        # Correct COCO to YOLO category mapping: 17=cat -> 0, 18=dog -> 1
        if ann["category_id"] == 17:  # cat
            cls = 0
        elif ann["category_id"] == 18:  # dog
            cls = 1
        else:
            continue  # Skip non-cat/dog annotations
        yolo_lines.append(f"{cls} {xc} {yc} {w} {h}\n")
    return yolo_lines

def process_coco(coco_root, tmp_imgs, tmp_lbls, fmt):
    """Extract COCO cats/dogs + convert to YOLO"""
    # Use the passed coco_root path to ensure correct relative path resolution
    annotation_path = coco_root / "annotations/instances_train2017.json"
    if not annotation_path.exists():
        # If relative path doesn't exist, try path relative to current working directory
        annotation_path = Path("data/coco/annotations/instances_train2017.json")
        if not annotation_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found. Tried paths:\n"
                                  f"1. {coco_root / 'annotations/instances_train2017.json'}\n"
                                  f"2. {Path('data/coco/annotations/instances_train2017.json').absolute()}")
    coco = COCO(str(annotation_path))
    catIds = coco.getCatIds(catNms=["cat", "dog"])

    # Get images containing only cats, only dogs, and both cats and dogs
    cat_only_imgIds = coco.getImgIds(catIds=[catIds[0]])  # Cats only
    dog_only_imgIds = coco.getImgIds(catIds=[catIds[1]])  # Dogs only
    both_imgIds = list(set(coco.getImgIds(catIds=catIds)) - set(cat_only_imgIds) - set(dog_only_imgIds))  # Both cats and dogs

    print(f"\n🔹 Processing COCO dataset")
    total = len(cat_only_imgIds) + len(dog_only_imgIds) + len(both_imgIds)

    # Counters for statistics
    single_cat_count = 0
    single_dog_count = 0
    multi_pet_count = 0

    # Process cat-only images
    for i, img_id in enumerate(tqdm(cat_only_imgIds, desc="COCO cats")):
        img_info = coco.loadImgs(img_id)[0]
        src_file = coco_root / "train2017" / img_info["file_name"]
        dst_file = tmp_imgs / f"coco_{img_info['file_name']}"
        ensure_dir(dst_file.parent)
        shutil.copy(src_file, dst_file)
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[catIds[0]], iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(anns) == 1:
            single_cat_count += 1
        elif len(anns) > 1:
            multi_pet_count += 1

        if fmt == "yolo":
            lbl = coco2yolo(anns, img_info["width"], img_info["height"])
            with open(dst_file.with_suffix(".txt"), "w") as f:
                f.writelines(lbl)

    # Process dog-only images
    for i, img_id in enumerate(tqdm(dog_only_imgIds, desc="COCO dogs")):
        img_info = coco.loadImgs(img_id)[0]
        src_file = coco_root / "train2017" / img_info["file_name"]
        dst_file = tmp_imgs / f"coco_{img_info['file_name']}"
        ensure_dir(dst_file.parent)
        shutil.copy(src_file, dst_file)
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[catIds[1]], iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(anns) == 1:
            single_dog_count += 1
        elif len(anns) > 1:
            multi_pet_count += 1

        if fmt == "yolo":
            lbl = coco2yolo(anns, img_info["width"], img_info["height"])
            with open(dst_file.with_suffix(".txt"), "w") as f:
                f.writelines(lbl)

    # Process images containing both cats and dogs
    for i, img_id in enumerate(tqdm(both_imgIds, desc="COCO cats & dogs")):
        if i % 10 == 0 or i == total-1:
            print(f"   Progress: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
        img_info = coco.loadImgs(img_id)[0]
        src_file = coco_root / "train2017" / img_info["file_name"]
        dst_file = tmp_imgs / f"coco_{img_info['file_name']}"
        ensure_dir(dst_file.parent)
        shutil.copy(src_file, dst_file)
        # Labels
        annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # These images all contain multiple pets
        multi_pet_count += 1

        if fmt == "yolo":
            lbl = coco2yolo(anns, img_info["width"], img_info["height"])
            with open(dst_file.with_suffix(".txt"), "w") as f:
                f.writelines(lbl)
        else:
            # Leave empty: will convert to COCO format later
            pass

    # Add multi-pet count from cat-only and dog-only images that have multiple pets
    # (This is already included in the loops above through the elif len(anns) > 1 conditions)

    # Print detailed statistics
    print(f"   Single cat images: {single_cat_count}")
    print(f"   Single dog images: {single_dog_count}")
    print(f"   Multiple cats and dogs images: {multi_pet_count}")
    print(f"   Total processed images: {single_cat_count + single_dog_count + multi_pet_count}")
def process_folder(src_imgs, src_lbls, tmp_imgs, tmp_lbls, prefix, fmt):
    """Process folder (Oxford/Stanford/CC0) copy + label conversion"""
    total = len(list(src_imgs.rglob("*.jpg")))
    print(f"\n🔹 Processing {prefix} dataset ({total} images)")
    for i, img_path in enumerate(tqdm(list(src_imgs.rglob("*.jpg")), desc=prefix)):
        if i % 100 == 0 or i == total-1:
            print(f"   Progress: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
        dst_img = tmp_imgs / f"{prefix}_{img_path.name}"
        ensure_dir(dst_img.parent)
        shutil.copy(img_path, dst_img)
        # Labels: copy directly for YOLO; leave empty for others
        if fmt == "yolo" and src_lbls:
            # Preserve the relative path structure from src_imgs to src_lbls
            relative_path = img_path.relative_to(src_imgs)
            src_txt = src_lbls / relative_path.with_suffix(".txt")
            if src_txt.exists():
                dst_txt = dst_img.with_suffix(".txt")
                ensure_dir(dst_txt.parent)
                shutil.copy(src_txt, dst_txt)
def generate_weak_labels(img_dir, out_json, conf=0.30):
    """Generate COCO format pseudo-labels for existing image directory using YOLOv8 pre-trained model (cats and dogs only)"""
    yolo = YOLO("yolov8n.pt")          # COCO pre-trained, contains cat/dog classes 15 16
    images, annotations = [], []
    img_paths = list(img_dir.glob("*.jpg"))
    ann_id = 1

    for img_id, img_path in enumerate(tqdm(img_paths, desc="Generating weak labels"), start=1):
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"⚠️ Warning: Failed to read image {img_path}, skipping")
            continue
        h, w, _ = im.shape
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        results = yolo(im, conf=conf)[0]
        for box in results.boxes:
            cls = int(box.cls)          # COCO: 15=cat 16=dog
            if cls not in [15, 16]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Convert YOLOv8 class IDs to standard COCO class IDs (15→17, 16→18)
            coco_cls = 17 if cls == 15 else 18
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": coco_cls,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            ann_id += 1

    json.dump({"images": images, "annotations": annotations,
               "categories": [{"id": 17, "name": "cat"}, {"id": 18, "name": "dog"}]},
              open(out_json, "w"), indent=2)
    print("Weak label saved →", out_json, "Images:", len(images), "Boxes:", len(annotations))

def generate_whole_image_labels(img_dir, out_json, class_id=0):
    """Generate whole image bounding boxes for classification datasets (Oxford/Stanford)"""
    images, annotations = [], []
    img_paths = list(img_dir.rglob("*.jpg"))
    ann_id = 1

    # Counters for statistics
    cat_count = 0
    dog_count = 0

    for img_id, img_path in enumerate(tqdm(img_paths, desc="Generating whole image labels"), start=1):
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"Warning: Failed to read image {img_path}, skipping...")
            continue
        h, w, _ = im.shape
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        # Determine class based on filename (Oxford/Stanford specific)
        img_stem = img_path.stem
        if "oxford" in str(img_dir).lower():
            # Oxford: filename format "breed_number.jpg", e.g., "basset_hound_112.jpg"
            breed = "_".join(img_stem.split('_')[:-1])

            # Oxford-IIIT Pets dataset: 12 cat breeds, 25 dog breeds
            cat_breeds = {
                'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
                'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
                'Siamese', 'Sphynx'
            }

            if breed in cat_breeds:
                detected_class = 0  # cat
            else:
                detected_class = 1  # dog

        elif "stanford" in str(img_dir).lower():
            # Stanford: filename format "nXXXXX_number.jpg", e.g., "n02097658_26.jpg"
            # n02097658 = silky_terrier (dog breed), so all Stanford images are dogs
            detected_class = 1  # All Stanford dogs images are dogs
        else:
            # Fallback to provided class_id if dataset not recognized
            detected_class = class_id

        # Update statistics
        if detected_class == 0:
            cat_count += 1
        else:
            dog_count += 1

        # Create whole image bounding box
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": detected_class,  # 0 for cat, 1 for dog
            "bbox": [0, 0, w, h],         # Whole image bounding box
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1

        # Also create YOLO format .txt file for each image
        # Preserve the relative path structure from img_dir
        relative_path = img_path.relative_to(img_dir)
        yolo_txt_path = img_dir.parent / "annotations" / "yolo" / relative_path.with_suffix(".txt")
        yolo_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yolo_txt_path, 'w') as f:
            # YOLO format: class_id center_x center_y width height
            # For whole image bbox: center at (0.5, 0.5), size (1.0, 1.0)
            f.write(f"{detected_class} 0.5 0.5 1.0 1.0\n")

    json.dump({"images": images, "annotations": annotations,
               "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]},
              open(out_json, "w"), indent=2)
    print("Whole image labels saved →", out_json, "Images:", len(images), "Boxes:", len(annotations))
    print(f"✅ Class distribution: {cat_count} cats, {dog_count} dogs")
    print(f"✅ Also created YOLO format labels in {img_dir.parent / 'annotations' / 'yolo'}")

    json.dump({"images": images, "annotations": annotations,
               "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]},
              open(out_json, "w"), indent=2)
    print("Whole image labels saved →", out_json, "Images:", len(images), "Boxes:", len(annotations))
    print(f"✅ Also created YOLO format labels in {img_dir.parent / 'annotations' / 'yolo'}")

def process_weak(weak_json, tmp_imgs, tmp_lbls, prefix, fmt):
    """Weak label JSON (COCO format) → copy images + convert to YOLO"""
    with open(weak_json) as f:
        data = json.load(f)
    # Create id→filename mapping
    im_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

    # Get the original image directory from the weak_json path
    # weak_json is expected to be in the same directory as the images
    image_dir = weak_json.parent

    for ann in tqdm(data["annotations"], desc="Weak"):
        img_id = ann["image_id"]
        img_name = im_id_to_name[img_id]
        src_img = image_dir / img_name   # Original image path
        dst_img = tmp_imgs / f"{prefix}_{img_name}"
        ensure_dir(dst_img.parent)
        if not dst_img.exists():
            shutil.copy(src_img, dst_img)
        if fmt == "yolo":
            # Find the corresponding image info for width/height
            img_info = next((img for img in data["images"] if img["id"] == img_id), None)
            if img_info:
                # Convert YOLOv8 class IDs to standard COCO class IDs using utility function
                ann_copy = ann.copy()
                ann_copy["category_id"] = yolo_to_coco_class(ann_copy["category_id"])
                lbl = coco2yolo([ann_copy], img_info["width"], img_info["height"])
                with open(dst_img.with_suffix(".txt"), "w") as f:
                    f.writelines(lbl)

def split_set(images, labels, split_ratios):
    """Return (train, val, test) filename lists"""
    ratio = [int(r) for r in split_ratios.split('/')]
    train_r, val_r, test_r = ratio[0], ratio[1], ratio[2]
    total = train_r + val_r + test_r
    files = [p.name for p in images.glob("*.jpg")]
    train, tmp = train_test_split(files, test_size=(val_r + test_r) / total, random_state=42)
    val, test = train_test_split(tmp, test_size=test_r / (val_r + test_r), random_state=42)
    return train, val, test

def validate_yolo_labels(tmp_imgs):
    """Validate YOLO format labels and ensure consistency"""
    valid_count = 0
    invalid_count = 0

    # Find all label files recursively
    txt_paths = list(tmp_imgs.rglob("*.txt"))

    for txt_path in tqdm(txt_paths, desc="Validating labels"):
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # Validate each annotation line
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid annotation format: {line}")

                cls_id, x_center, y_center, width, height = parts
                cls_id = int(cls_id)
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)

                # Validate class ID
                if cls_id not in [0, 1]:
                    raise ValueError(f"Invalid class ID: {cls_id}")

                # Validate normalized coordinates
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 <= width <= 1 and 0 <= height <= 1):
                    raise ValueError(f"Invalid normalized coordinates: {x_center}, {y_center}, {width}, {height}")

            valid_count += 1

        except Exception as e:
            print(f"Warning: Invalid label file {txt_path}: {e}")
            invalid_count += 1
            # Remove invalid label file
            txt_path.unlink()

    print(f"Label validation: {valid_count} valid, {invalid_count} invalid labels")
    return valid_count, invalid_count

def build_yolo_dataset(tmp_imgs, tmp_lbls, train_files, val_files, test_files, out_dir):
    """Split temporary files into train/val/test + generate dataset.yaml"""
    # First validate all labels
    valid_count, invalid_count = validate_yolo_labels(tmp_imgs)

    if invalid_count > 0:
        print(f"⚠️  Removed {invalid_count} invalid label files")

    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        for fname in files:
            img_src = tmp_imgs / fname
            if img_src.exists():
                # Preserve directory structure from tmp_imgs
                img_dst = out_dir / "images" / split / fname
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(img_src, img_dst)

            lbl_src = tmp_imgs / fname.replace(".jpg", ".txt")
            if lbl_src.exists():
                # Preserve directory structure from tmp_imgs
                lbl_dst = out_dir / "labels" / split / fname.replace(".jpg", ".txt")
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(lbl_src, lbl_dst)

    # YOLOv8 configuration
    yaml_content = f"""path: {out_dir.absolute()}
train: images/train
val: images/val
test: images/test
nc: 2
names: ['cat', 'dog']
"""
    (out_dir / "dataset.yaml").write_text(yaml_content)

    # Print dataset statistics
    print(f"\n📊 Final Dataset Statistics:")
    for split in ["train", "val", "test"]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_count = len(list(img_dir.glob("*.jpg")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))
        print(f"   {split.capitalize()}: {img_count} images, {lbl_count} labels")

def main():
    config = {
        "oxford": Path("../data/oxford_pets"),      # Oxford-IIIT Pets dataset
        "stanford": Path("../data/stanford_dogs"),  # Stanford Dogs dataset
        "coco": Path("../data/coco"),               # COCO2017 dataset
        "weak": Path("../data/cc0_api/weak_anno.json"),  # Wpateak label JSON (COCO format) - Not available
        "cc0": Path("../data/cc0_api"),             # CC0 API data
        "fmt": "yolo",                           # Output format: "yolo" or "coco"
        "split": "85/5/10",                      # Train/val/test ratio
        "out": Path("../data/merged_pet_dataset")      # Output directory
    }

    # Check if output directory already exists and contains files
    if config["out"].exists() and any(config["out"].rglob("*")):
        print(f"\n⚠️ Warning: Output directory {config['out']} already exists and is not empty.")
        print("This indicates the merge process may have already been completed.")
        response = input("Do you want to proceed anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("Merge process aborted.")
            return

    # Check if Oxford and Stanford have YOLO annotations, if not generate whole image labels
    if config["oxford"] and not (config["oxford"] / "annotations" / "yolo").exists():
        print("⚠️ Oxford Pets YOLO annotations not found, will generate whole image labels")
        generate_whole_image_labels(config["oxford"] / "images", config["oxford"] / "whole_anno.json", class_id=0)
        config["weak_oxford"] = config["oxford"] / "whole_anno.json"
        # Create YOLO annotations directory for Oxford
        oxford_yolo_dir = config["oxford"] / "annotations" / "yolo"
        oxford_yolo_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created YOLO annotations directory: {oxford_yolo_dir}")

    if config["stanford"] and not (config["stanford"]  / "annotations" / "yolo").exists():
        print("⚠️ Stanford Dogs YOLO annotations not found, will generate whole image labels")
        generate_whole_image_labels(config["stanford"] / "Images", config["stanford"] / "whole_anno.json", class_id=1)
        config["weak_stanford"] = config["stanford"] / "whole_anno.json"
        # Create YOLO annotations directory for Stanford
        stanford_yolo_dir = config["stanford"] / "annotations" / "yolo"
        stanford_yolo_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created YOLO annotations directory: {stanford_yolo_dir}")

    tmp_imgs = Path("/tmp/merge_imgs"); tmp_imgs.mkdir(exist_ok=True)
    tmp_lbls = Path("/tmp/merge_lbls"); tmp_lbls.mkdir(exist_ok=True)

    # 1. Process each source: copy + convert
    if config["oxford"]:
        process_folder(config["oxford"] / "images", config["oxford"] / "annotations" / "yolo", tmp_imgs, tmp_lbls, "oxford", config["fmt"])
    if config["stanford"]:
        process_folder(config["stanford"] / "Images", config["stanford"] / "annotations" / "yolo", tmp_imgs, tmp_lbls, "stanford", config["fmt"])
    if config["coco"]:
        process_coco(config["coco"], tmp_imgs, tmp_lbls, config["fmt"])

    if config["weak"]:
        # Check if weak_anno.json exists, if not generate it
        weak_json_path = config["weak"]
        if not weak_json_path.exists():
            print(f"?? Weak annotation file not found at {weak_json_path}, generating...")
            generate_weak_labels(config["cc0"], weak_json_path, conf=0.30)
        # Process weak annotations for CC0 dataset
        process_weak(config["weak"], tmp_imgs, tmp_lbls, "cc0", config["fmt"])
    elif config["cc0"]:
        # Fallback: if no weak annotations, process CC0 dataset without labels
        print("⚠️ Warning: No weak annotations found for CC0 dataset, processing without labels")
        process_folder(config["cc0"], None, tmp_imgs, tmp_lbls, "cc0", config["fmt"])

    # 2. Hash-based deduplication (file level)
    print("Deduplicating...")
    seen = set()
    for p in tqdm(list(tmp_imgs.rglob("*.jpg"))):
        h = hash_file(p)
        if h in seen:
            p.unlink(missing_ok=True)
            p.with_suffix(".txt").unlink(missing_ok=True)
        else:
            seen.add(h)

    # 3. Split dataset
    train_files, val_files, test_files = split_set(tmp_imgs, tmp_lbls, config["split"])

    # 4. Output
    print("\n" + "="*50)
    print("🏁 Final Dataset Creation")
    print("="*50 + "\n")
    print("Building YOLO dataset structure...")
    build_yolo_dataset(tmp_imgs, tmp_lbls, train_files, val_files, test_files, config["out"])
    print("\n✅ All done! Final dataset saved to:", config["out"].absolute())
    print(f"Total images processed: {len(train_files)+len(val_files)+len(test_files)}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

if __name__ == "__main__":
    main()