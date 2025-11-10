#!/usr/bin/env python3
"""
Dataset Merger for PetBuddy
---------------------------------
Author: Bright Wang
Project: PetBuddy project

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
        cls = 0 if ann["category_id"] == 15 else 1  # 15=cat 16=dog
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
    print(f"   Multiple pet images: {multi_pet_count}")
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
            src_txt = (src_lbls / img_path.stem).with_suffix(".txt")
            if src_txt.exists():
                dst_txt = dst_img.with_suffix(".txt")
                shutil.copy(src_txt, dst_txt)

def generate_weak_labels(img_dir, out_json, conf=0.30):
    """Generate COCO format pseudo-labels for existing image directory using YOLOv8 pre-trained model (cats and dogs only)"""
    yolo = YOLO("yolov8n.pt")          # COCO pre-trained, contains cat/dog classes 15 16
    images, annotations = [], []
    img_paths = list(img_dir.glob("*.jpg"))
    ann_id = 1

    for img_id, img_path in enumerate(tqdm(img_paths, desc="Generating weak labels"), start=1):
        im = cv2.imread(str(img_path))
        h, w, _ = im.shape
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        results = yolo(im, conf=conf)[0]
        for box in results.boxes:
            cls = int(box.cls)          # COCO: 15=cat 16=dog
            if cls not in [15, 16]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 15 if cls == 15 else 16,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            ann_id += 1

    json.dump({"images": images, "annotations": annotations,
               "categories": [{"id": 15, "name": "cat"}, {"id": 16, "name": "dog"}]},
              open(out_json, "w"), indent=2)
    print("Weak label saved →", out_json, "Images:", len(images), "Boxes:", len(annotations))

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
                lbl = coco2yolo([ann], img_info["width"], img_info["height"])
                with open(dst_img.with_suffix(".txt"), "a") as f:
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

def build_yolo_dataset(tmp_imgs, tmp_lbls, train_files, val_files, test_files, out_dir):
    """Split temporary files into train/val/test + generate dataset.yaml"""
    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for fname in files:
            shutil.move(tmp_imgs / fname, out_dir / "images" / split / fname)
            lbl_src = tmp_imgs / fname.replace(".jpg", ".txt")
            if lbl_src.exists():
                shutil.move(lbl_src, out_dir / "labels" / split / fname.replace(".jpg", ".txt"))
    # YOLOv8 configuration
    yaml_content = f"""path: {out_dir.absolute()}
train: images/train
val: images/val
test: images/test
nc: 2
names: ['cat', 'dog']
"""
    (out_dir / "dataset.yaml").write_text(yaml_content)

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

    tmp_imgs = Path("/tmp/merge_imgs"); tmp_imgs.mkdir(exist_ok=True)
    tmp_lbls = Path("/tmp/merge_lbls"); tmp_lbls.mkdir(exist_ok=True)

    # 1. Process each source: copy + convert
    if config["oxford"]:
        process_folder(config["oxford"] / "images", config["oxford"] / "annotations" / "yolo", tmp_imgs, tmp_lbls, "oxford", config["fmt"])
    if config["stanford"]:
        process_folder(config["stanford"] / "Images", config["stanford"] / "YOLO", tmp_imgs, tmp_lbls, "stanford", config["fmt"])
    if config["coco"]:
        process_coco(config["coco"], tmp_imgs, tmp_lbls, config["fmt"])
    if config["weak"]:
        # Check if weak_anno.json exists, if not generate it
        weak_json_path = config["weak"]
        if not weak_json_path.exists():
            print(f"🔹 Weak annotation file not found at {weak_json_path}, generating...")
            generate_weak_labels(config["cc0"], weak_json_path, conf=0.30)
        process_weak(config["weak"], tmp_imgs, tmp_lbls, "weak", config["fmt"])
    if config["cc0"]:
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