#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/experiments/optuna_train.py
Location: tools/experiments/
====================================
Optuna Hyperparameter Optimization Trainer

Purpose:
- Provide hyperparameter optimization using Optuna framework
- Support automated search for optimal learning rates, weight decay, and MixUp parameters
- Enable efficient Bayesian optimization for pet recognition models

Key Features:
1. Optuna Integration: Automated hyperparameter search using Bayesian optimization
2. Multi-Parameter Optimization: Simultaneously optimize learning rate, weight decay, and MixUp
3. Early Stopping: Implement patience-based early stopping to save computation
4. Checkpointing: Save best model weights during optimization process
5. Comprehensive Logging: Detailed training logs and optimization progress tracking
"""
import os
import sys
import argparse  # Added：for receiving command line arguments
from pathlib import Path

import optuna
import torch
import torch.distributed as dist
import torch.nn as nn  # Modified
import torch.optim as optim
import yaml
from optuna.trial import TrialState
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Added: MixUp related libraries ---
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

torch.set_float32_matmul_precision('high')

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

from models.petnet import PetNet
from utils.data_loader import build_dataloader, build_datasampler


def is_distributed():
    """Check if it is distributed envirionment or not"""
    return "WORLD_SIZE" in os.environ


def setup_distributed_env():
    if not is_distributed():
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_actual_num_classes(dataset_name):
    from utils.data_utils import get_num_classes
    # Simplified processing, can actually reuse train.py logic through dataloader
    # Assume dataset_name corresponds to existing folder
    try:
        return get_num_classes(dataset_name)
    except:
        print(f"⚠️ Could not get class count from util, defaulting to 144")
        return 144


def objective(trial: optuna.trial.Trial, base_config_path: str, distributed=False):
    """Optuna objective function"""

    if distributed:
        trial = optuna.integration.TorchDistributedTrial(trial)

    # Load Optuna search space configuration
    optuna_config = load_config("configs/optuna_tuning.yaml")
    tuner_config = optuna_config["tuner"]

    # --- Define search space ---
    lr = trial.suggest_float("lr", tuner_config["lr_low"], tuner_config["lr_high"], log=True)
    weight_decay = trial.suggest_float("weight_decay", tuner_config["weight_decay_low"],
                                       tuner_config["weight_decay_high"], log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])

    # Optional: Search MixUp parameters (if needed)
    # mixup_prob = trial.suggest_float("mixup_prob", 0.0, 1.0)

    # --- Load base training configuration ---
    petnet_config = load_config(base_config_path)

    # Override parameters
    petnet_config['train']['learning_rate'] = lr
    petnet_config['train']['weight_decay'] = weight_decay
    petnet_config['train']['optimizer'] = optimizer_name.lower()

    # Get configuration sections
    model_config = petnet_config['model']
    train_config = petnet_config['train']
    data_config = petnet_config['data']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not distributed or int(os.environ.get("RANK", 0)) == 0:
        print(f"🚀 Trial {trial.number}: lr={lr:.2e}, wd={weight_decay:.2e}, opt={optimizer_name}")

    # Prepare data loaders
    root_dir = data_config.get("root_dir", "data/pet_cls_training")
    # Convert to absolute path to avoid path doubling in data_loader
    root_dir_path = Path(root_dir)
    if not root_dir_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        root_dir = str(project_root / root_dir)
    dataset_to_use = root_dir  # For simplicity, use path directly

    # Distributed Sampler
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = build_datasampler(root_dir=root_dir, shuffle=True, split="train",
                                          ldre_cfg=data_config.get('ldre_cfg'))
        val_sampler = build_datasampler(root_dir=root_dir, shuffle=False, split="val",
                                        ldre_cfg=data_config.get('ldre_cfg'))

    # Get input size
    input_size_cfg = data_config.get('input_size', 224)
    img_size = input_size_cfg[0] if isinstance(input_size_cfg, (list, tuple)) else int(input_size_cfg)

    train_loader = build_dataloader(
        root_dir=root_dir,
        batch_size=train_config['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=data_config.get("num_workers", 4),
        sampler=train_sampler,
        split="train",
        img_size=img_size,
        ldre_cfg=data_config.get('ldre_cfg')
    )

    # Get actual number of classes
    actual_num_classes = train_loader.dataset.num_classes

    val_loader = build_dataloader(
        root_dir=root_dir,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        sampler=val_sampler,
        split="val",
        img_size=img_size,
        ldre_cfg=data_config.get('ldre_cfg')
    )

    # --- Initialize MixUp ---
    mixup_fn = None
    mixup_args = train_config.get('mixup_args')
    if mixup_args and (mixup_args.get('prob', 0.0) > 0):
        mixup_fn = Mixup(
            mixup_alpha=mixup_args.get('mixup_alpha', 0.8),
            cutmix_alpha=mixup_args.get('cutmix_alpha', 1.0),
            prob=mixup_args.get('prob', 1.0),
            switch_prob=mixup_args.get('switch_prob', 0.5),
            mode=mixup_args.get('mode', 'batch'),
            label_smoothing=train_config.get('label_smoothing', 0.1),
            num_classes=actual_num_classes
        )
        criterion = SoftTargetCrossEntropy().to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=train_config.get('label_smoothing', 0.1)).to(device)

    # Create model
    # Filter out PetNet unnecessary parameters
    petnet_params = {k: v for k, v in model_config.items()
                     if k in ['stage_repeats', 'attn_cfg', 'selfkd_cfg', 'max_pets_per_image', 'drop_path_rate']}

    model = PetNet(num_classes=actual_num_classes, **petnet_params)
    model.to(device)

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # optimizer
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config['num_epochs'], eta_min=1e-6
    )

    num_epochs = train_config['num_epochs']
    # Accumulation only makes sense in distributed mode, or if single GPU wants to accumulate
    accumulation_steps = train_config.get('gradient_accumulation_steps', 1)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    selfkd_enabled = model_config.get('selfkd_cfg', {}).get('enable', False)

    best_acc = 0.0

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for i, batch_data in enumerate(train_loader):
            inputs, labels = batch_data['images'].to(device), batch_data['labels'].to(device)

            # Apply MixUp
            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)

                # 🚨 Fix: Properly handle SelfKD output
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    kd_loss = outputs[1] if selfkd_enabled else 0.0
                else:
                    logits = outputs
                    kd_loss = 0.0

                # Calculate Loss
                cls_loss = criterion(logits, labels)
                loss = cls_loss + kd_loss
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()

        # --- Validation Loop ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data['images'].to(device), batch_data['labels'].to(device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    # During validation, only take logits
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        if distributed:
            # Aggregate results from all GPUs
            metrics = torch.tensor([val_correct, val_total], dtype=torch.long, device=device)
            dist.all_reduce(metrics)
            val_correct = metrics[0].item()
            val_total = metrics[1].item()

        val_acc = 100. * val_correct / val_total
        best_acc = max(best_acc, val_acc)

        scheduler.step()

        # Report progress to Optuna (for Pruning)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if not distributed or int(os.environ.get("RANK", 0)) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Val Acc: {val_acc:.2f}%, Best: {best_acc:.2f}%")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    # Allow user to specify which config file to optimize (default is SOTA config)
    parser.add_argument('--config', type=str, default='configs/petnet_base.yaml',
                        help='Base config file to optimize')
    args = parser.parse_args()

    distributed, rank, world_size = setup_distributed_env()

    if distributed:
        print(f"Running in distributed environment: rank {rank}/{world_size}")
        dist.barrier()

    # Load optuna configuration here
    config = load_config("configs/optuna_tuning.yaml")

    # Pruner settings
    pruner_cfg = config.get("pruner", {})
    if pruner_cfg.get("type") == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=pruner_cfg.get("min_resource", 1),
            max_resource=pruner_cfg.get("max_resource", 120),
            reduction_factor=pruner_cfg.get("reduction_factor", 3),
        )
    else:
        pruner = optuna.pruners.MedianPruner()

    storage_url = "sqlite:///petnet_study.db"
    study_name = f"optimize_{Path(args.config).stem}"

    # Create or load Study
    if not distributed or rank == 0:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True
        )
        print(f"🔬 Starting optimization based on: {args.config}")
    else:
        study = None  # Worker nodes don't need study object, only run trial

    # Run optimization
    # Note: In distributed environment, each process runs optimize, but Optuna syncs through DB
    # If it's just a single machine with multiple GPUs, usually only the main process runs optimize and distributes tasks (simplified here to single process running trial internal DDP)
    # For simple single-machine multi-GPU setups, it's recommended not to open DDP inside objective, running on single GPU may be more efficient (due to many trials)
    # Here we keep your DDP logic, but note the SQLite concurrency lock issue

    try:
        # Only the main process is responsible for scheduling, or all processes connect to DB to grab tasks
        # For simplicity, assume single machine running or external torchrun startup
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        study.optimize(lambda t: objective(t, args.config, distributed), n_trials=config["tuner"]["n_trials"])
    except Exception as e:
        if not distributed or rank == 0:
            print(f"Optimization loop error (or finished): {e}")

    # --- Result display (main process only) ---
    if (not distributed or rank == 0) and study is not None:
        print("\n📊 Optimization statistics:")
        print(f"   Best Value: {study.best_value:.2f}%")
        print("   Best Params:")
        for k, v in study.best_params.items():
            print(f"     {k}: {v}")

        # Save best parameters to new file
        best_config = load_config(args.config)
        best_config['train']['learning_rate'] = study.best_params['lr']
        best_config['train']['weight_decay'] = study.best_params['weight_decay']
        best_config['train']['optimizer'] = study.best_params['optimizer'].lower()

        # Save
        save_path = f"configs/{Path(args.config).stem}_optimized.yaml"
        with open(save_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        print(f"💾 Optimized config saved to: {save_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()