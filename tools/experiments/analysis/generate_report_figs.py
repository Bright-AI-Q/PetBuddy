#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/experiments/analysis/generate_report_figs.py
Location: tools/experiments/analysis/
====================================
Report Figures Generator (GenerateReportFigs)

Purpose:
- Generate comparative visualization plots for pet recognition experiments
- Support performance comparison across different model configurations
- Enable academic paper-quality figure generation with IEEE styling

Key Features:
1. Multi-Experiment Comparison: Visualize multiple model configurations simultaneously
2. IEEE Paper Styling: Apply academic publication standard formatting
3. Flexible Data Loading: Support both explicit config log paths and auto-detection
4. Vector Graphics Output: Generate PDF files suitable for academic publications
5. Performance Metrics: Compare validation accuracy across different experiments
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_experiment(config_paths, output_dir="report/images"):
    """
    Args:
        config_paths: List containing paths to multiple config files (for comparison)
        output_dir: Image output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set IEEE paper plotting style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "figure.dpi": 300
    })
    # Prepare data container
    data_list = []

    for cfg_path in config_paths:
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            print(f"⚠️ Config not found: {cfg_path}")
            continue

        config = load_config(cfg_path)

        # 1. Get log directory from config
        log_dir = Path(config.get('logging', {}).get('log_dir', ''))
        if str(log_dir) == '':
            # If not specified in config, try to infer default path
            log_dir = Path(f"runs/{cfg_path.stem}")

        csv_file = log_dir / "training_log.csv"

        # 2. Read CSV
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # Add label to data (using config file name, or define your own mapping)
            # Simple processing: remove 'petnet_' prefix for legend name
            label_name = cfg_path.stem.replace('petnet_', '').replace('mobilenet_', 'Baseline: ').replace('_', ' ').title()
            df['Experiment'] = label_name
            data_list.append(df)
            print(f"✅ Loaded data for: {label_name}")
        else:
            print(f"❌ Log not found for {cfg_path.stem} at {csv_file}")
            print("   (Please ensure you ran the modified train.py)")

    if not data_list:
        print("No data loaded. Exiting.")
        return

    # Merge data
    all_data = pd.concat(data_list)

    # --- Plot 1: Validation Accuracy ---
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=all_data, x='Epoch', y='Val_Acc', hue='Experiment', palette='tab10')
    plt.title('Comparison of Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = Path(output_dir) / "comparison_accuracy.pdf"  # PDF vector graphics for papers
    plt.savefig(save_path)
    plt.savefig(save_path.with_suffix('.png'))  # Also save PNG for easy viewing
    print(f"📊 Accuracy plot saved to {save_path}")
    plt.close()

    # --- Plot 2: Train vs Val Loss (Optional) ---
    # Can plot more complex graphs here, e.g., only showing SOTA model loss

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Allow passing multiple config files for comparison
    parser.add_argument('--configs', nargs='+', default=[
        'configs/petnet_mobilenet_baseline.yaml',
        'configs/petnet_base.yaml',
        'configs/petnet_att.yaml',
        'configs/petnet_att_skd.yaml',
        'configs/petnet_att_skd_ldr.yaml',
        'configs/petnet_att_skd_enldr.yaml',
        'configs/petnet_att_skd_mup.yaml',
        'configs/petnet_att_skd_mup_ldr.yaml',
        'configs/petnet_att_skd_mup_enldr.yaml',
        'configs/petnet_fine_tune.yaml'
    ], help='List of config files to compare')

    args = parser.parse_args()
    plot_experiment(args.configs)