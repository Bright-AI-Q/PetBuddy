#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: merge_classification_data.py
====================================
Breed Classification Dataset Merger
Purpose:
- Merge Oxford-IIIT Pet and Stanford Dogs datasets
- Generate unified folder structure with sequential numbering
- Create YOLOv8-compatible dataset.yaml
"""
import json
import shutil
from pathlib import Path

OXFORD_ROOT   = Path("../data/oxford_pets")
STANFORD_ROOT = Path("../data/stanford_dogs")
OUTPUT_ROOT   = Path("../data/merged_cls_dataset")

# Breed counter
breed_counter = {}

def get_cat_breeds():
    """Extract cat breed list from YOLO label files (class_id=0)"""
    cat_breeds = set()
    for txt_file in (OXFORD_ROOT / "annotations/yolo").rglob("*.txt"):
        with open(txt_file) as f:
            first_line = f.readline().strip()
            if first_line.startswith('0 '):  # class_id=0 indicates cat
                breed = txt_file.stem.split('_')[0].lower()
                cat_breeds.add(breed)
    return cat_breeds

# Dynamically get cat breeds
CAT_BREEDS = get_cat_breeds()

def get_breed_number(breed_name: str) -> str:
    """Assign unique breed number (starting from 0001)"""
    breed_lower = breed_name.lower()
    if breed_lower not in breed_counter:
        breed_counter[breed_lower] = len(breed_counter) + 1
    return f"{breed_counter[breed_lower]:04d}"

def oxford_to_pet_id(oxford_name: str) -> str:
    """Return format: pets_[number]_[breedname]"""
    breed_num = get_breed_number(oxford_name)
    return f"pets_{breed_num}_{oxford_name.lower()}"

def build_oxford_cls(out_dir: Path):
    """Oxford → pets_[number]_[breedname] format folders"""
    src_imgs = OXFORD_ROOT / "images"
    images = list(src_imgs.rglob("*.jpg"))
    total = len(images)
    print(f"🔸 Processing {total} Oxford images...")

    for i, img in enumerate(images, 1):
        breed = "_".join(img.stem.split('_')[:-1])
        dir_name = oxford_to_pet_id(breed)
        dst_dir = out_dir / dir_name
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst_dir / img.name)
            if i % 100 == 0 or i == total:
                print(f"  Processed {i}/{total} ({i/total:.1%})")
        except Exception as e:
            print(f"⚠️ Failed to process {img}: {str(e)}")

def build_stanford_cls(out_dir: Path):
    """Stanford → pets_[number]_[breedname] format folders"""
    src_imgs = STANFORD_ROOT / "Images"
    images = list(src_imgs.rglob("*.jpg"))
    total = len(images)
    print(f"🔸 Processing {total} Stanford images...")

    for i, img in enumerate(images, 1):
        breed_name = img.parent.name.split('-')[1]  # Extract breed name from directory
        dir_name = oxford_to_pet_id(breed_name)  # Use unified naming function
        dst_dir = out_dir / dir_name
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst_dir / img.name)
            if i % 100 == 0 or i == total:
                print(f"  Processed {i}/{total} ({i/total:.1%})")
        except Exception as e:
            print(f"⚠️ Failed to process {img}: {str(e)}")

def build_dataset_yaml(out_dir: Path):
    """Generate YOLOv8-cls dataset.yaml"""
    breeds = sorted([d.name for d in out_dir.iterdir() if d.is_dir()])
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(f"path: {out_dir.absolute()}\n")
        f.write("train: .\n")
        f.write("val: .\n")
        f.write(f"nc: {len(breeds)}\n")
        f.write("names: [\n" + ",\n".join(f'  "{b}"' for b in breeds) + "\n]\n")

def ensure_dir(path: Path):
    """Ensure output directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def validate_input_dirs():
    """Validate input directories exist"""
    if not OXFORD_ROOT.exists():
        raise FileNotFoundError(f"Oxford dataset not found at {OXFORD_ROOT}")
    if not STANFORD_ROOT.exists():
        raise FileNotFoundError(f"Stanford dataset not found at {STANFORD_ROOT}")

def main():
    try:
        validate_input_dirs()
        ensure_dir(OUTPUT_ROOT)
        print("🔹 Merging Oxford + Stanford → unified breed folders...")
        build_oxford_cls(OUTPUT_ROOT)
        build_stanford_cls(OUTPUT_ROOT)
        build_dataset_yaml(OUTPUT_ROOT)
        print("✅ Done ->", OUTPUT_ROOT.absolute())
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()