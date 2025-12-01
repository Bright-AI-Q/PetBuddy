#!/usr/bin/env python3
"""
Project: PetBuddy
File: tools/experiments/analysis/visualize_confusion.py
Purpose: Generate Confusion Matrix.
"""

import torch
import argparse
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import timm

# Path setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from models.petnet import PetNet
from utils.data_loader import build_dataloader


def load_config(config_path):
    with open(config_path, 'r') as f: return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default='confusion_matrix.pdf')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    data_config = config['data']

    # Load Data
    val_loader = build_dataloader(
        root_dir=data_config.get('root_dir', 'data/pet_cls_training'),
        batch_size=64, split="val", shuffle=False,
        img_size=data_config.get('input_size', 224)[0]
    )

    # Load Model (Same logic as above, simplified here)
    num_classes = val_loader.dataset.num_classes
    model_config = config['model']
    model_type = model_config.get('type', 'petnet')
    use_baseline = config.get('use_baseline', False)

    if use_baseline:
        model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=num_classes)
    elif model_type == 'petnet':
        petnet_params = {k: v for k, v in model_config.items() if k in ['stage_repeats', 'attn_cfg', 'selfkd_cfg']}
        model = PetNet(num_classes=num_classes, **petnet_params)
    else:
        model = timm.create_model(model_type, pretrained=False, num_classes=num_classes)

    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.to(device).eval()

    # Inference
    all_preds = []
    all_labels = []

    print("⏳ Running inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            logits = model(batch['images'].to(device))
            if isinstance(logits, tuple): logits = logits[0]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

        # Compute Matrix
        cm = confusion_matrix(all_labels, all_preds)


        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"]
        })

        plt.figure(figsize=(10, 8))

        # 核心修改：xticklabels=False, yticklabels=False
        ax = sns.heatmap(cm,
                         annot=False,
                         cmap='Blues',
                         xticklabels=False,
                         yticklabels=False,
                         cbar_kws={'label': 'Sample Count'})

        plt.title(f"Confusion Matrix (144 Classes)", fontsize=16, pad=20)
        plt.xlabel("Predicted Classes", fontsize=14)
        plt.ylabel("True Classes", fontsize=14)

        plt.tight_layout()

        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion Matrix saved to {args.output}")

        # Additionally generate a CSV listing the Top-10 confusion pairs for the paper's Analysis section
        # Find the maximum values on the off-diagonal

        np.fill_diagonal(cm, 0)
        indices = np.dstack(np.unravel_index(np.argsort(cm.ravel())[-10:], cm.shape))[0]

        print("\n🔍 Top 10 Confused Pairs (Write these in your qualitative analysis):")
        idx_to_name = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
        for idx in indices[::-1]:  # 降序
            true_cls, pred_cls = idx
            cnt = cm[true_cls, pred_cls]
            print(f"   - True: {idx_to_name[true_cls]} -> Pred: {idx_to_name[pred_cls]} (Count: {cnt})")


if __name__ == "__main__":
    main()