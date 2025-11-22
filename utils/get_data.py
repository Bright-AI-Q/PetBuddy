#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Deng Jinyang
File: get_data.py
====================================
Get dataset for PetBuddy project
Purpose:
- This script downloads the final pet_cls_training dataset required for training.
- After this script, you can directly call tools/experiments/optuna_train.py

Previous process (time-consuming) to get the same dataset:
- 1. Inside utils folder
    1. data_downloader.py
    2. scarpe_cc0_api.py
    3. merge_classification_data.py
    4. merge_detection_data.py
- 2. Inside main folder
    1. data_utils.py
"""

import os
import zipfile
from pathlib import Path
import gdown

DATA_URL = "https://drive.google.com/uc?id=1W9MyCf2cpgGKUiDNS_1jIZvQXIaMdHGu"

def main():
    # Get parent directory (PetBuddy root)
    parent_dir = Path(__file__).parent.parent.absolute()
    tmp_folder = Path(parent_dir / "tmp"); tmp_folder.mkdir(parents=True, exist_ok=True)

    # Download the file from google drive
    print("⬇️ Downloading data...")
    output = tmp_folder / "data.zip"
    gdown.download(DATA_URL, str(output))

    # Extract to parent/data folder
    print("📂 Extracting data...")
    data_folder = parent_dir / "data"
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    # Delete the zip file
    os.remove(output)
    print("✅ Finished setting up.")

if __name__ == "__main__":
    main()