#!/usr/bin/env python3
"""
Project: PetBuddy
File: tools/experiments/analysis/visualize_tsne.py
Purpose: Generate t-SNE feature embedding visualization.
"""

import torch
import argparse
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm
import timm

# Path setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from models.petnet import PetNet
from utils.data_loader import build_dataloader


def load_config(config_path):
    with open(config_path, 'r') as f: return yaml.safe_load(f)


def get_features(model, inputs):
    """Extract features before the classifier head"""
    # For PetNet (Custom)
    if hasattr(model, 'stage3') and hasattr(model, 'head'):
        # PetNet Forward logic until head
        x = model.stem(inputs)
        x = model.stage1(x)
        x = model.stage2(x)
        x = model.stage3(x)
        # Global Average Pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    # For TIMM models (MobileNet, EfficientNet, etc.)
    elif hasattr(model, 'forward_features'):
        x = model.forward_features(inputs)
        x = model.forward_head(x, pre_logits=True)  # Get features before final linear
        return x

    # For torchvision models
    else:
        # Fallback: This is tricky, usually requires hooks.
        # For simplicity, we assume PetNet or TIMM structure as per your project.
        raise NotImplementedError("Model structure not supported for feature extraction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True, help='Path to best.pt')
    parser.add_argument('--output', default='tsne_plot.pdf')
    parser.add_argument('--max-samples', type=int, default=1000, help='Limit points to plot')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Config & Data
    config = load_config(args.config)
    data_config = config['data']

    # Load Validation Set
    val_loader = build_dataloader(
        root_dir=data_config.get('root_dir', 'data/pet_cls_training'),
        batch_size=32,
        split="val",
        shuffle=True,  # Shuffle to get random samples if limited
        img_size=data_config.get('input_size', 224)[0]
    )

    # 2. Load Model
    num_classes = val_loader.dataset.num_classes
    model_config = config['model']
    model_type = model_config.get('type', 'petnet')
    use_baseline = config.get('use_baseline', False)

    print(f"⏳ Loading model: {model_type} (Baseline: {use_baseline})")

    if use_baseline:
        model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=num_classes)
    elif model_type == 'petnet':
        petnet_params = {k: v for k, v in model_config.items() if k in ['stage_repeats', 'attn_cfg', 'selfkd_cfg']}
        model = PetNet(num_classes=num_classes, **petnet_params)
    elif model_type in ['efficientnet_b0', 'mobileone_s0']:
        model = timm.create_model(model_type, pretrained=False, num_classes=num_classes)

    # Load weights
    ckpt = torch.load(args.model, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. Extract Features
    features_list = []
    labels_list = []

    print("⏳ Extracting features...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            imgs = batch['images'].to(device)
            lbls = batch['labels']

            feats = get_features(model, imgs)

            features_list.append(feats.cpu().numpy())
            labels_list.append(lbls.numpy())

            if len(features_list) * 32 >= args.max_samples:
                break

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    # 4. Run t-SNE
    print(f"⏳ Running t-SNE on {X.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)

    # 5. Plot (IEEE Style)
    plt.style.use('seaborn-v0_8-paper')
    # force use Times New Roman match  LaTeX
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 10
    })

    plt.figure(figsize=(8, 8))


    scatter = sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=y,
        palette="tab10",
        s=60,
        alpha=0.7,
        edgecolor="w",
        legend=False
    )

    # 修改标题，去掉下划线
    clean_title = Path(args.config).stem.replace('_', ' ').title().replace('Petnet', 'PetNet')
    plt.title(f"t-SNE Feature Visualization: {clean_title}", fontsize=16, pad=15)

    # 去掉坐标轴刻度，只保留框线（t-SNE的坐标数值没有物理意义）
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()

    # 保存
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.savefig(args.output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE plot saved to {args.output}")


if __name__ == "__main__":
    main()