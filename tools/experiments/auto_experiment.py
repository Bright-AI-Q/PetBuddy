#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/experiments/auto_experiment.py
Location: tools/experiments/
====================================
Auto Experiment Pipeline (AutoExperiment)

Purpose:
- Provide fully automated research pipeline for pet recognition experiments
- Support batch training, grouped plotting, Grad-CAM generation, and LaTeX table creation
- Enable comprehensive ablation studies and performance comparisons

Key Features:
1. Batch Training: Automatically train multiple model configurations
2. Smart GPU Detection: Auto-select between single-GPU and multi-GPU training
3. Grouped Visualization: Generate comparative plots for different experiment groups
4. Grad-CAM Analysis: Create heatmap visualizations for model interpretability
5. LaTeX Reporting: Automatically generate academic paper-quality tables
6. Configurable Groups: Flexible experiment grouping for ablation studies
"""

import argparse
import subprocess
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

import os
import shutil

# Must be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import torch

# --- 📍 Path location system (Core modification) ---
# 1. Get current script directory: tools/experiments/
CURRENT_DIR = Path(__file__).resolve().parent
# 2. Get project root directory: PetBuddy/ (向上两级)
PROJECT_ROOT = CURRENT_DIR.parents[1]
# 3. Analysis script directory: tools/experiments/analysis/
ANALYSIS_DIR = CURRENT_DIR / "analysis"

# Add analysis directory to search path，以便导入绘图模块
sys.path.append(str(ANALYSIS_DIR))

try:
    from generate_report_figs import plot_experiment
except ImportError:
    print(f"❌ Error: Could not import generate_report_figs from {ANALYSIS_DIR}")
    sys.exit(1)

# ================= 🧪 Experiment configuration section =================

# Experiment list (Config Path, Display Name)
# Note：Config paths are relative to PROJECT_ROOT
EXPERIMENTS = [
    # Group A: Architecture baseline
    ("configs/petnet_mobilenet_baseline.yaml", "Baseline (MobileNetV2)"),  # 0
    ("configs/petnet_base.yaml", "PetNet (Basic Structure)"),  # 1

    # Group B: Module ablation
    ("configs/petnet_att.yaml", "+ Attention"),  # 2
    ("configs/petnet_att_skd.yaml", "+ Attn + SelfKD"),  # 3

    # Group C: Data augmentation (LDRE variants)
    ("configs/petnet_att_skd_ldr.yaml", "+ Standard LDRE"),  # 4
    ("configs/petnet_att_skd_enldr.yaml", "+ Enhanced LDRE"),  # 5

    # Group D: Combination techniques (MixUp interaction)
    ("configs/petnet_att_skd_mup.yaml", "+ MixUp Only"),  # 6
    ("configs/petnet_att_skd_mup_ldr.yaml", "+ MixUp + Std LDRE"),  # 7
    ("configs/petnet_att_skd_mup_enldr.yaml", "Ours (MixUp + Enh LDRE)"),  # 8

    # Group E: Final fine-tuning
    ("configs/petnet_fine_tune.yaml", "Ours (Fine-tuned)")  # 9
]

# Plot grouping strategy
PLOT_GROUPS = {
    "fig_architecture": {
        "title": "Impact of Architecture Modules",
        "indices": [0, 1, 2, 3]
    },
    "fig_ldre_variants": {
        "title": "Standard vs. Enhanced LDRE",
        "indices": [3, 4, 5]
    },
    "fig_synergy": {
        "title": "Synergy between MixUp and LDRE",
        "indices": [3, 6, 8]
    },
    "fig_finetune": {
        "title": "Performance Boost from Fine-tuning",
        "indices": [8, 9]
    }
}

# Grad-CAM test image (Relative to PROJECT_ROOT)
SAMPLE_IMAGE = "Abyssinian_15.png"


# ===================================================

def clean_runs_directory():
    """
    Clean up all files in runs/ directory
    """
    runs_dir = PROJECT_ROOT / "runs"

    if not runs_dir.exists():
        print(f"ℹ️  directory: {runs_dir} not exist，Cleanup skipped")
        return

    print("\n" + "!" * 60)
    print(f"⚠️  Warning: You used the --force parameter。")
    print(f"⚠️  This will permanently delete '{runs_dir}' all logs, weights, and checkpoints in！")
    print("!" * 60)

    while True:
        response = input(f"❓ Confirm to empty {runs_dir} ? [y/N]: ").strip().lower()
        if response == 'y':
            try:
                print(f"🧹 deleting... {runs_dir} ...")
                shutil.rmtree(runs_dir)
                print("✅ Deletion complete, environment reset。")

                runs_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"❌ Deletion failed: {e}")
                sys.exit(1)
            break
        elif response == 'n' or response == '':
            print("🚫 Operation cancelled, program exiting。")
            sys.exit(0)
        else:
            print("Invalid input, please enter y 或 n。")


def run_training(force_rerun=False):
    """Stage 1: Batch Training"""


    if force_rerun:
        clean_runs_directory()


    print("\n" + "=" * 50)
    print("🚀 Stage 1: Batch Training")
    print("=" * 50)

    # Detect GPU count and select training script
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPUs")

    if num_gpus > 1:
        train_script = PROJECT_ROOT / "tools" / "train.py"  # Note：新版 train.py 已集成 DDP
        print("✅ Using multi-GPU distributed training (torchrun)")
    else:
        train_script = PROJECT_ROOT / "tools" / "train.py"
        print("✅ Using single-GPU training (python)")

    for config_rel_path, exp_name in EXPERIMENTS:
        # Construct absolute path
        config_path = PROJECT_ROOT / config_rel_path

        if not config_path.exists():
            print(f"⚠️  Config missing: {config_path}")
            continue

        config_name = config_path.stem
        log_file = PROJECT_ROOT / "runs" / config_name / "training_log.csv"


        if log_file.exists():
            print(f"⏩ [{exp_name}] Already exists, skipping。")
            continue

        print(f"▶️  Training: {exp_name} ...")
        try:
            # Key: cwd=PROJECT_ROOT ensures training script can find configs/ and data/
            if num_gpus > 1:
                # Multi-GPU training requires torchrun to start
                subprocess.run(
                    ["torchrun", "--nproc_per_node", str(num_gpus),
                     str(train_script), "--config", str(config_path)],
                    check=True,
                    cwd=PROJECT_ROOT
                )
            else:
                # Single-GPU training
                subprocess.run(
                    ["python", str(train_script), "--config", str(config_path)],
                    check=True,
                    cwd=PROJECT_ROOT
                )
        except subprocess.CalledProcessError:
            print(f"❌ Error executing {config_name}")
            continue

        print("🧹 Cleaning up GPU memory...")
        gc.collect()  # Clear Python garbage objects
        torch.cuda.empty_cache()  # Clear PyTorch cache allocator


def run_visualization():
    """Stage 2: Generating Grouped Plots"""
    print("\n" + "=" * 50)
    print("🎨 Stage 2: Generating Grouped Plots")
    print("=" * 50)

    output_dir = PROJECT_ROOT / "report" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, group_info in PLOT_GROUPS.items():
        print(f"📊 Generating {filename}...")
        try:

            selected_configs = [PROJECT_ROOT / EXPERIMENTS[i][0] for i in group_info["indices"]]
            _custom_plot(selected_configs, group_info["title"], output_dir / f"{filename}.pdf")

        except IndexError:
            print(f"⚠️  Index out of range for group {filename}, skipping.")


def _custom_plot(config_paths, title, save_path):
    """内部辅助绘图函数"""
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-paper')

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2
    })

    has_data = False
    for cfg_path in config_paths:
        cfg_path = Path(cfg_path)

        try:

            display_name = next((name for path, name in EXPERIMENTS if Path(path).stem == cfg_path.stem), cfg_path.stem)
        except StopIteration:
            display_name = cfg_path.stem

        config_name = cfg_path.stem
        log_file = PROJECT_ROOT / "runs" / config_name / "training_log.csv"

        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                df['Val_Acc_Smooth'] = df['Val_Acc'].rolling(window=3, min_periods=1).mean()
                best_val = df['Val_Acc'].max()
                plt.plot(df['Epoch'], df['Val_Acc_Smooth'], label=f"{display_name} ({best_val:.2f}%)")
                has_data = True
            except Exception as e:
                print(f"Error reading {log_file}: {e}")

    if has_data:
        plt.title(title, fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Validation Accuracy (%)", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix(".png"))
        print(f"   -> Saved to {save_path}")
        plt.close()
    else:
        print("   -> No data found for this group.")


def run_gradcam():
    """Stage 3: Generating Grad-CAM"""
    print("\n" + "=" * 50)
    print("🔥 Stage 3: Generating Grad-CAM")
    print("=" * 50)

    img_path = PROJECT_ROOT / SAMPLE_IMAGE
    if not img_path.exists():
        print(f"⚠️  Image not found: {img_path}")
        return

    key_indices = [0, 3, 8, 9]  # Baseline, Base+SelfKD, Full, Finetune


    vis_script = ANALYSIS_DIR / "visualize_gradcam.py"

    for i in key_indices:
        if i >= len(EXPERIMENTS): continue

        config_rel, exp_name = EXPERIMENTS[i]
        config_path = PROJECT_ROOT / config_rel
        config_name = config_path.stem
        model_path = PROJECT_ROOT / "runs" / config_name / "best.pt"

        if model_path.exists():
            print(f"🖼️  Processing {exp_name}...")
            output_path = PROJECT_ROOT / "report" / "images" / f"gradcam_{config_name}.jpg"

            cmd = [
                "python", str(vis_script),
                "--image", str(img_path),
                "--model", str(model_path),
                "--config", str(config_path),
                "--output", str(output_path),
                "--img-size", "256",
                "--concat"
            ]

            subprocess.run(cmd, cwd=PROJECT_ROOT)
        else:
            print(f"   ⚠️  Skipping {exp_name}: Model not found.")


def generate_latex_table():
    """Stage 4: Generating LaTeX Table"""
    print("\n" + "=" * 50)
    print("📝 Stage 4: Generating LaTeX Table")
    print("=" * 50)

    results = []
    baseline_acc = 0.0

    for i, (config_rel, exp_name) in enumerate(EXPERIMENTS):
        config_name = Path(config_rel).stem
        log_file = PROJECT_ROOT / "runs" / config_name / "training_log.csv"

        best_acc = 0.0
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                best_acc = df['Val_Acc'].max()
            except:
                pass

        if i == 0:
            baseline_acc = best_acc
            gain = "-"
        else:
            diff = best_acc - baseline_acc
            gain = f"+{diff:.2f}\\%" if baseline_acc > 0 else "N/A"

        results.append({"name": exp_name, "acc": best_acc, "gain": gain})

    latex = r"""
\begin{table}[h]
    \centering
    \caption{Comprehensive Ablation Study.}
    \label{tab:ablation_full}
    \resizebox{\linewidth}{!}{
    \begin{tabular}{lcc}
        \toprule
        \textbf{Method / Configuration} & \textbf{Accuracy (\%)} & \textbf{Gain} \\
        \midrule
        \textit{Baseline} & & \\
"""

    for i, res in enumerate(results):
        name, acc, gain = res['name'], res['acc'], res['gain']

        if i == 1: latex += r"        \midrule" + "\n        \textit{Architecture Modules} & & \\\\\n"
        if i == 4: latex += r"        \midrule" + "\n        \textit{Data Augmentation (LDRE)} & & \\\\\n"
        if i == 6: latex += r"        \midrule" + "\n        \textit{System Synergy (MixUp + LDRE)} & & \\\\\n"
        if i == 9: latex += r"        \midrule" + "\n"

        if i == len(results) - 1:
            row = f"        \\textbf{{{name}}} & \\textbf{{{acc:.2f}}} & \\textbf{{{gain}}} \\\\"
        else:
            row = f"        {name} & {acc:.2f} & {gain} \\\\"
        latex += row + "\n"

    latex += r"""        \bottomrule
    \end{tabular}
    }
\end{table}
"""

    out_path = PROJECT_ROOT / "report" / "ablation_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"✅ LaTeX table saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='Skip training, only generate report')
    parser.add_argument('--force', action='store_true', help='Force re-training (DELETE ALL LOGS)')
    args = parser.parse_args()

    # 1. Training
    if not args.skip_train:
        run_training(force_rerun=args.force)

    # 2. plot
    run_visualization()

    # 3. Grad-CAM
    run_gradcam()

    # 4. write-table
    generate_latex_table()

    print(f"\n🎉 Pipeline Finished! Check '{PROJECT_ROOT}/report/' folder.")