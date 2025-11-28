"""
Purpose:
This merges:
1. dataset_cat_train.jsonl into dataset_train.jsonl
2. dataset_cat_val.jsonl files dataset_val.jsonl

**Why a 2-step process**:
The jsonl are synthetic data and may sometimes be malformed.
Please manually verify and correct the generated dataset as needed.

This script assumes that the cat datasets have been verified and corrected.
"""

import os

# Paths
cat_train = os.path.join(os.path.dirname(__file__), "dataset_cat_train.jsonl")
cat_val = os.path.join(os.path.dirname(__file__), "dataset_cat_val.jsonl")
train = os.path.join(os.path.dirname(__file__), "dataset_train.jsonl")
val = os.path.join(os.path.dirname(__file__), "dataset_val.jsonl")

def merge_jsonl(src, dest):
    with open(dest, "a", encoding="utf-8") as fout:
        with open(src, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)

if __name__ == "__main__":
    # Merge cat_train into train
    if os.path.exists(cat_train) and os.path.exists(train):
        merge_jsonl(cat_train, train)
        os.remove(cat_train)
        print("✅ Merged cat training data.")
    # Merge cat_val into val
    if os.path.exists(cat_val) and os.path.exists(val):
        merge_jsonl(cat_val, val)
        os.remove(cat_val)
        print("✅ Merged cat validation data.")

