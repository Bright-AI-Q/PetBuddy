#!/usr/bin/env python3
"""
Project: PetBuddy
File: tools/experiments/analysis/visualize_sota_bar.py
Purpose: Generate Bar Chart comparing SOTA models vs Ours.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='runs', help='Root runs directory')
    parser.add_argument('--output', default='report/images/sota_comparison.pdf')
    args = parser.parse_args()

    # Define models to look for
    targets = {
        "petnet_mobilenet_baseline": "MobileNetV2",
        "efficientnet_b0": "EfficientNet-B0",
        "mobileone_s0": "MobileOne-S0",
        "petnet_fine_tune": "Ours (PetNet)"
    }

    data = []
    runs_dir = Path(args.runs_dir)

    for folder_name, display_name in targets.items():
        log_file = runs_dir / folder_name / "training_log.csv"
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                best_acc = df['Val_Acc'].max()
                data.append({"Model": display_name, "Accuracy": best_acc})
            except:
                print(f"⚠️ Could not read log for {folder_name}")
        else:
            # 这是一个容错逻辑，防止因为没有训练某个模型导致脚本报错
            # 如果找不到日志，填一个假数据或者跳过，这里选择跳过
            print(f"⚠️ Log not found for {folder_name}")

    if not data:
        print("❌ No data found.")
        return

    df_plot = pd.DataFrame(data)

    # === 全局字体设置 (防止 Times New Roman 报错) ===
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "axes.labelsize": 14,
        "font.size": 12,
        "lines.linewidth": 2.5
    })

    plt.figure(figsize=(7, 5))

    # Custom color palette: Ours highlighted
    clrs = ['grey' if x != "Ours (PetNet)" else '#d62728' for x in df_plot['Model']]


    ax = sns.barplot(
        x='Model',
        y='Accuracy',
        data=df_plot,
        hue='Model',
        palette=clrs,
        legend=False  # 3. urn off the legend (since bar charts do not require an additional legend)


    )

    # Add numbers on top
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f%%', padding=3, fontsize=11)

    plt.ylim(min(df_plot['Accuracy']) - 5, max(df_plot['Accuracy']) + 3)  # 稍微增加一点顶部空间
    plt.title("Comparison with State-of-the-Art Lightweight Models", fontsize=14, pad=15)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.savefig(args.output.replace('.pdf', '.png'), dpi=300)
    print(f"✅ SOTA comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()