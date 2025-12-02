#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/experiments/auto_experiment.py
Location: tools/experiments/
====================================
Auto Experiment Pipeline (AutoExperiment) - Updated for Curriculum Learning

Purpose:
- Provide fully automated research pipeline for pet recognition experiments
- Support batch training, grouped plotting, Grad-CAM generation, and LaTeX table creation
- Enable comprehensive ablation studies and performance comparisons

Key Features:
1. Batch Training: Automatically train multiple model configurations
2. Smart GPU Detection: Auto-select between single-GPU and multi-GPU training
3. Grouped Visualization: Generate comparative plots (including Curriculum Stitching)
4. Grad-CAM Analysis: Create heatmap visualizations
5. LaTeX Reporting: Automatically generate academic paper-quality tables
6. Advanced Analysis: t-SNE, SOTA Bar Chart, Confusion Matrix
"""

import argparse
import subprocess
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import gc
import torch

# Must be set before importing torch logic if OOM occurs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 📍 Path location system ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
ANALYSIS_DIR = CURRENT_DIR / "analysis"

# Add analysis directory to search path
sys.path.append(str(ANALYSIS_DIR))

# ================= 🧪 Experiment Configuration Section =================

# Experiment list (Config Path, Display Name)
# Config paths are relative to PROJECT_ROOT
EXPERIMENTS = [
    # --- Group A: SOTA Baselines ---
    ("configs/petnet_mobilenet_baseline.yaml", "MobileNetV2 (Baseline)"),  # 0
    ("configs/efficientnet_b0.yaml", "EfficientNet-B0"),  # 1
    ("configs/mobileone_s0.yaml", "MobileOne-S0"),  # 2

    # --- Group B: Module Ablation (Architecture) ---
    ("configs/petnet_att.yaml", "+ Attention"),  # 3
    ("configs/petnet_att_skd.yaml", "+ Attn + SelfKD"),  # 4

    # --- Group C: LDRE Innovation Validation ---
    ("configs/petnet_att_skd_ldr.yaml", "+ Standard LDRE"),  # 5
    ("configs/petnet_att_skd_enldr.yaml", "+ Enhanced LDRE (Ours)"),  # 6

    # --- Group D: Synergy with MixUp ---
    ("configs/petnet_att_skd_mup.yaml", "+ MixUp Only"),  # 7
    ("configs/petnet_att_skd_mup_ldr.yaml", "+ MixUp + Std LDRE"),  # 8

    # --- Group E: Final Proposed Method (Curriculum Learning) ---
    ("configs/petnet_base.yaml", "Ours (Stage 1: MixUp)"),  # 9
    ("configs/petnet_fine_tune.yaml", "Ours (Stage 2: LDRE Fine-tune)")  # 10
]

# Plot grouping strategy
PLOT_GROUPS = {
    "fig_sota_comparison": {
        "title": "Comparison with SOTA Lightweight Models",
        "indices": [0, 1, 2, 10],  # Compare against Fine-tuned model (Best)
        "is_continuous": False
    },
    "fig_module_ablation": {
        "title": "Impact of Architecture Modules",
        "indices": [0, 3, 4],
        "is_continuous": False
    },
    "fig_ldre_innovation": {
        "title": "Validation of Enhanced LDRE (Innovation)",
        "indices": [4, 5, 6],
        "is_continuous": False
    },
    "fig_synergy": {
        "title": "Synergy: MixUp vs LDRE Variants",
        "indices": [7, 8, 9],
        "is_continuous": False
    },
    # ✨ NEW: Curriculum Learning Jump Plot
    "fig_curriculum_jump": {
        "title": "Curriculum Learning Effect: MixUp -> LDRE Fine-tuning",
        "indices": [9, 10],  # MixUp Base -> Fine-tuned
        "is_continuous": True  # Enable stitching logic
    }
}

# Grad-CAM test image (Relative to PROJECT_ROOT)
SAMPLE_IMAGE = "Abyssinian_15.png"


# =======================================================================

def clean_runs_directory():
    """Clean up all files in runs/ directory"""
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return

    print("\n" + "!" * 60)
    print(f"⚠️  Warning: --force flag detected.")
    print(f"⚠️  Deleting ALL logs and checkpoints in '{runs_dir}'!")
    print("!" * 60)

    while True:
        response = input(f"❓ Confirm delete? [y/N]: ").strip().lower()
        if response == 'y':
            try:
                shutil.rmtree(runs_dir)
                print("✅ Cleaned.")
                runs_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"❌ Deletion failed: {e}")
                sys.exit(1)
            break
        elif response == 'n' or response == '':
            sys.exit(0)


def run_training(force_rerun=False):
    """Stage 1: Batch Training"""
    if force_rerun:
        clean_runs_directory()

    print("\n" + "=" * 50)
    print("🚀 Stage 1: Batch Training")
    print("=" * 50)

    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPUs")
    train_script = PROJECT_ROOT / "tools" / "train.py"

    for config_rel_path, exp_name in EXPERIMENTS:
        config_path = PROJECT_ROOT / config_rel_path

        if not config_path.exists():
            print(f"⚠️  Config missing: {config_path}, skipping.")
            continue

        config_name = config_path.stem
        log_file = PROJECT_ROOT / "runs" / config_name / "training_log.csv"
        if log_file.exists():
            print(f"⏩ [{exp_name}] Logs found, skipping training.")
            continue

        print(f"▶️  Training: {exp_name} ({config_name}) ...")

        try:
            if num_gpus > 1:
                cmd = ["torchrun", "--nproc_per_node", str(num_gpus), str(train_script), "--config", str(config_path)]
            else:
                cmd = ["python", str(train_script), "--config", str(config_path)]

            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        except subprocess.CalledProcessError:
            print(f"❌ Training failed for {config_name}")
            continue

        gc.collect()
        torch.cuda.empty_cache()


def run_visualization():
    """Stage 2: Generating Grouped Plots"""
    print("\n" + "=" * 50)
    print("🎨 Stage 2: Generating Comparative Plots")
    print("=" * 50)

    output_dir = PROJECT_ROOT / "report" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, group_info in PLOT_GROUPS.items():
        print(f"📊 Generating {filename}...")
        try:
            selected_configs = []
            for i in group_info["indices"]:
                if i < len(EXPERIMENTS):
                    selected_configs.append(PROJECT_ROOT / EXPERIMENTS[i][0])

            if selected_configs:
                _custom_plot(
                    selected_configs,
                    group_info["title"],
                    output_dir / f"{filename}.pdf",
                    is_continuous=group_info.get("is_continuous", False)
                )
            else:
                print("   ⚠️ No valid configs for this group.")

        except IndexError:
            print(f"   ⚠️ Index error in group {filename}")


def _custom_plot(config_paths, title, save_path, is_continuous=False):
    """Internal helper to plot comparison graphs"""

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],  # times new roman for paper
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,  # Bolden the lines
        "axes.grid": True,
        "grid.linestyle": '--',
        "grid.alpha": 0.5
    })


    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12, "font.size": 12,
        "legend.fontsize": 10, "lines.linewidth": 2
    })

    # === ✨ Logic for Curriculum Learning Stitching ===
    if is_continuous and len(config_paths) == 2:
        path1, path2 = config_paths

        try:
            df1 = pd.read_csv(PROJECT_ROOT / "runs" / path1.stem / "training_log.csv")
            df2 = pd.read_csv(PROJECT_ROOT / "runs" / path2.stem / "training_log.csv")

            # Smooth
            df1['Val_Acc_Smooth'] = df1['Val_Acc'].rolling(window=5, min_periods=1).mean()
            df2['Val_Acc_Smooth'] = df2['Val_Acc'].rolling(window=5, min_periods=1).mean()

            # Plot Stage 1
            plt.plot(df1['Epoch'], df1['Val_Acc_Smooth'], label="Stage 1: MixUp (Generalization)", color='#1f77b4',
                     alpha=0.8)

            # Plot Stage 2 (Offset Epochs)
            offset = df1['Epoch'].max()
            df2['Epoch_Abs'] = df2['Epoch'] + offset
            plt.plot(df2['Epoch_Abs'], df2['Val_Acc_Smooth'], label="Stage 2: LDRE (Refinement)", color='#d62728',
                     linewidth=3)

            # Vertical Line
            plt.axvline(x=offset, color='black', linestyle='--', alpha=0.5, label="Fine-tune Start")

            # Annotate Jump
            jump_start = df1['Val_Acc_Smooth'].iloc[-1]
            jump_end = df2['Val_Acc_Smooth'].max()
            boost = jump_end - jump_start

            if boost > 0:
                plt.annotate(f"+{boost:.2f}% Boost",
                             xy=(offset + 1, jump_end),
                             xytext=(offset + 4, jump_end - 0.5),
                             arrowprops=dict(facecolor='black', shrink=0.05),
                             fontsize=12, fontweight='bold', color='#d62728')

            plt.title(title, fontsize=14)
            plt.xlabel("Total Epochs")
            plt.ylabel("Validation Accuracy (%)")
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.savefig(save_path.with_suffix(".png"))
            print(f"   ✅ Saved Stitched Plot to {save_path.name}")
            plt.close()
            return

        except Exception as e:
            print(f"   ⚠️ Error plotting continuous graph: {e}. Falling back to standard plot.")

    # === Standard Plot Logic ===
    has_data = False
    for cfg_path in config_paths:
        display_name = cfg_path.stem
        for path, name in EXPERIMENTS:
            if Path(path).stem == cfg_path.stem:
                display_name = name
                break

        log_file = PROJECT_ROOT / "runs" / cfg_path.stem / "training_log.csv"

        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                df['Val_Acc_Smooth'] = df['Val_Acc'].rolling(window=3, min_periods=1).mean()
                best_val = df['Val_Acc'].max()
                plt.plot(df['Epoch'], df['Val_Acc_Smooth'], label=f"{display_name} (Max: {best_val:.2f}%)")
                has_data = True
            except Exception as e:
                print(f"   Error reading {log_file.name}: {e}")

    if has_data:
        plt.title(title, fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy (%)")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix(".png"))
        print(f"   ✅ Saved to {save_path.name}")
        plt.close()
    else:
        print("   ❌ No data available to plot.")


def run_gradcam():
    """Stage 3: Generating Grad-CAM"""
    print("\n" + "=" * 50)
    print("🔥 Stage 3: Generating Grad-CAM")
    print("=" * 50)

    img_path = PROJECT_ROOT / SAMPLE_IMAGE
    if not img_path.exists():
        test_dir = PROJECT_ROOT / "data" / "pet_cls_training" / "test"
        if test_dir.exists():
            cats = list(test_dir.glob("**/*.jpg"))
            if cats: img_path = cats[0]

    if not img_path or not img_path.exists():
        print(f"⚠️  Image for Grad-CAM not found. Skipping.")
        return

    print(f"   Using image: {img_path.name}")
    vis_script = ANALYSIS_DIR / "visualize_gradcam.py"

    # Indices: MobV2, EffNet, MobOne, Enhanced LDRE, Final Fine-tuned
    target_indices = [0, 1, 2, 6, 10]

    for i in target_indices:
        if i >= len(EXPERIMENTS): continue

        config_rel, exp_name = EXPERIMENTS[i]
        config_path = PROJECT_ROOT / config_rel
        config_name = config_path.stem
        # Use fine-tuned model path if applicable
        model_path = PROJECT_ROOT / "runs" / config_name / "best.pt"

        if model_path.exists():
            print(f"   🖼️  Processing {exp_name}...")
            output_path = PROJECT_ROOT / "report" / "images" / f"gradcam_{config_name}.jpg"

            cmd = [
                "python", str(vis_script),
                "--image", str(img_path),
                "--model", str(model_path),
                "--config", str(config_path),
                "--output", str(output_path),
                "--img-size", "224",
                "--concat"
            ]
            subprocess.run(cmd, cwd=PROJECT_ROOT)
        else:
            print(f"   ⚠️  Model not found: {config_name}")


def generate_latex_table():
    """Stage 4: Generating LaTeX Table"""
    print("\n" + "=" * 50)
    print("📝 Stage 4: Generating LaTeX Table")
    print("=" * 50)

    results = []
    baseline_acc = 0.0

    # 1. 读取基准线准确率
    base_log = PROJECT_ROOT / "runs" / Path(EXPERIMENTS[0][0]).stem / "training_log.csv"
    if base_log.exists():
        try:
            baseline_acc = pd.read_csv(base_log)['Val_Acc'].max()
        except:
            pass

    # 2. 读取所有实验结果
    for config_rel, exp_name in EXPERIMENTS:
        config_name = Path(config_rel).stem
        log_file = PROJECT_ROOT / "runs" / config_name / "training_log.csv"

        best_acc = 0.0
        if log_file.exists():
            try:
                best_acc = pd.read_csv(log_file)['Val_Acc'].max()
            except:
                pass

        if config_name == Path(EXPERIMENTS[0][0]).stem:
            gain = "-"
        else:
            diff = best_acc - baseline_acc

            gain = f"{diff:+.2f}\\%" if baseline_acc > 0 else "N/A"

        results.append({"name": exp_name, "acc": best_acc, "gain": gain})

    #3. Generate LaTeX String

    latex = r"""
\begin{table}[h]
    \centering
    \caption{Performance Comparison with SOTA and Ablation Studies.}
    \label{tab:main_results}
    \resizebox{\linewidth}{!}{
    \begin{tabular}{lcc}
        \toprule
        \textbf{Method / Configuration} & \textbf{Accuracy (\%)} & \textbf{Gain} \\
        \midrule
        \multicolumn{3}{l}{\textit{State-of-the-Art Baselines}} \\
"""

    for i, res in enumerate(results):
        # Fix the escaping of \multicolumn (to prevent Python SyntaxWarning)
        if i == 3: latex += r"        \midrule" + "\n        \\multicolumn{3}{l}{\\textit{PetNet Module Ablation}} \\\\\n"
        if i == 5: latex += r"        \midrule" + "\n        \\multicolumn{3}{l}{\\textit{LDRE Innovation (Std vs. Enh)}} \\\\\n"
        if i == 7: latex += r"        \midrule" + "\n        \\multicolumn{3}{l}{\\textit{Synergy with MixUp}} \\\\\n"
        if i == 9: latex += r"        \midrule" + "\n        \\multicolumn{3}{l}{\\textit{Curriculum Learning (Final)}} \\\\\n"

        name, acc, gain = res['name'], res['acc'], res['gain']

        # bolden (Index 10)
        if i == 10:
            row = f"        \\textbf{{{name}}} & \\textbf{{{acc:.2f}}} & \\textbf{{{gain}}} \\\\"
        else:
            row = f"        {name} & {acc:.2f} & {gain} \\\\"

        latex += row + "\n"

    latex += r"""        \bottomrule
    \end{tabular}
    }
\end{table}
"""

    out_path = PROJECT_ROOT / "report" / "results_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"✅ LaTeX table saved to {out_path}")

def run_advanced_analysis():
    """Stage 5: Advanced Analysis (t-SNE, Confusion Matrix, SOTA Bar)"""
    print("\n" + "=" * 50)
    print("🔬 Stage 5: Advanced Analysis & Reporting")
    print("=" * 50)

    # 1. SOTA Comparison Bar Chart
    print("📊 Generating SOTA Bar Chart...")
    subprocess.run([
        "python", str(ANALYSIS_DIR / "visualize_sota_bar.py"),
        "--runs-dir", str(PROJECT_ROOT / "runs"),
        "--output", str(PROJECT_ROOT / "report/images/sota_comparison.pdf")
    ], cwd=PROJECT_ROOT)

    # 2. t-SNE & Confusion Matrix (For the Best Model: Ours Fine-tuned)
    # ✨ Changed target to 'petnet_fine_tune.yaml'
    target_config = "configs/petnet_fine_tune.yaml"
    target_name = Path(target_config).stem
    best_model = PROJECT_ROOT / "runs" / target_name / "best.pt"

    if best_model.exists():
        print(f"🔮 Generating t-SNE for {target_name}...")
        subprocess.run([
            "python", str(ANALYSIS_DIR / "visualize_tsne.py"),
            "--config", str(PROJECT_ROOT / target_config),
            "--model", str(best_model),
            "--output", str(PROJECT_ROOT / f"report/images/tsne_{target_name}.pdf")
        ], cwd=PROJECT_ROOT)

        print(f"🌀 Generating Confusion Matrix for {target_name}...")
        subprocess.run([
            "python", str(ANALYSIS_DIR / "visualize_confusion.py"),
            "--config", str(PROJECT_ROOT / target_config),
            "--model", str(best_model),
            "--output", str(PROJECT_ROOT / f"report/images/confusion_{target_name}.pdf")
        ], cwd=PROJECT_ROOT)
    else:
        print(f"⚠️ Best model for Ours ({target_name}) not found, skipping deep analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='Skip training, only generate report')
    parser.add_argument('--force', action='store_true', help='Force re-training (DELETE ALL LOGS)')
    args = parser.parse_args()

    # 1. Training
    if not args.skip_train:
        run_training(force_rerun=args.force)

    # 2. Visualization (Plots)
    run_visualization()

    # 3. Grad-CAM
    run_gradcam()

    # 4. LaTeX Table
    generate_latex_table()

    # 5. Advanced Analysis
    run_advanced_analysis()

    print(f"\n🎉 Pipeline Finished! Check '{PROJECT_ROOT}/report/' folder.")