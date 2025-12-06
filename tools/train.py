"""
Project: PetBuddy
Author: Bright Wang
File: tools/train.py
Location: tools/
====================================
Main Training Script for PetBuddy Pet Recognition Models
 (Single GPU & DDP)


Purpose:
- Provide comprehensive training pipeline for pet recognition models
- Support configuration-based training with YAML configuration files
- Enable advanced techniques like MixUp, label smoothing, and curriculum learning

Key Features:
1. Configuration-Driven: Training parameters fully configurable through YAML files
2. Advanced Data Augmentation: Support for MixUp, CutMix, and other augmentation techniques
3. Model Flexibility: Compatible with various model architectures including custom PetNet
4. Training Monitoring: TensorBoard integration with comprehensive logging
5. Checkpoint Management: Automatic model checkpointing and best model saving
6. Performance Optimization: Support for mixed precision training and gradient accumulation
7. Curriculum Learning: Progressive training difficulty adjustment
8. Comprehensive Metrics: Training and validation accuracy/loss tracking

Usage:
1. Single GPU:
   python tools/train.py --config configs/petnet_base.yaml

2. Multi-GPU (DDP):
   torchrun --nproc_per_node=4 tools/train.py --config configs/petnet_base.yaml
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import yaml
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
import torch.nn.functional as F
import os
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import csv
import random
import timm
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models


# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.petnet import PetNet
from utils.data_loader import build_dataloader, build_datasampler


# ==============================================================================
# 1. Distributed Helpers
# ==============================================================================

def init_distributed_mode():
    """
    Auto-detect and initialize DDP environment.
    Returns: (is_ddp, rank, local_rank, world_size, device)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # DDP mode for multi gpu training
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()  # Sync all processes before proceeding

        print(f"🟢 [DDP] Process {rank}/{world_size} initialized on GPU {local_rank}")
        return True, rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")
    else:
        # Single GPU mode for single gpu training
        print("🟡 [Single] Running in Single-GPU mode")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_tensor(tensor, world_size):
    """Average a tensor across all GPUs."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ==============================================================================
# 2. Basic Utilities
# ==============================================================================

def set_seed(seed=42, rank=0):
    """Set seed for reproducibility, offsetting by rank for DDP dataloaders."""
    # In DDP, we want same weight init but different data augmentations if random
    # But DistributedSampler handles data shuffling indices.
    # Usually we set same seed for model init.
    seed = seed + rank  # Optional: distinct seeds per rank for some randomness layers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if rank == 0:
        print(f"🔒 Random Seed set to: {seed}")


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_accuracy(logits, labels):
    _, predicted = logits.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    return correct, total


# ==============================================================================
# 3. Training & Validation Functions
# ==============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device, model_config,
                    rank=0, is_ddp=False, mixup_fn=None):
    model.train()
    total_loss = 0.0
    mixup_mode = False

    # SelfKD Logic
    selfkd_cfg = model_config.get('selfkd_cfg', {})
    if selfkd_cfg is None: selfkd_cfg = {}
    selfkd_enabled = selfkd_cfg.get('enable', False)

    for i, batch_data in enumerate(train_loader):
        images = batch_data['images'].to(device, non_blocking=True)
        labels = batch_data['labels'].to(device, non_blocking=True)

        #    Apply MixUp/CutMix if enabled
        if mixup_fn is not None:
            mixup_mode = True
            images, labels = mixup_fn(images, labels)


        optimizer.zero_grad()
        output = model(images)

        #   Handle Tuple output (logits, kd_loss) or just logits
        if isinstance(output, tuple):
            logits, kd_loss = output
            if not selfkd_enabled: kd_loss = 0.0
        else:
            logits = output
            kd_loss = 0.0

        cls_loss = criterion(logits, labels)
        loss = cls_loss + kd_loss

        loss.backward()
        optimizer.step()

        # Track loss (Approximate for DDP to avoid syncing every batch)
        total_loss += loss.item()

        if is_main_process(rank) and i % 100 == 0:
            if mixup_mode:
                print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            else:
                correct, total = calculate_accuracy(logits, labels)
                acc = 100. * correct / total if total > 0 else 0.0
                print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")


    # Average loss across batches
    avg_loss = total_loss / len(train_loader)

    # Sync loss across GPUs for logging accuracy (Optional but good for monitoring)
    if is_ddp:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor)
        avg_loss = loss_tensor.item() / dist.get_world_size()

    return avg_loss


def validate_one_epoch(model, val_loader, criterion, device, is_ddp=False, tta_scales=[1.0]):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in val_loader:
            images = batch_data['images'].to(device, non_blocking=True)
            labels = batch_data['labels'].to(device, non_blocking=True)

                # TTA or Standard Forward
            if len(tta_scales) > 1:
                logits_list = []
                for scale in tta_scales:
                    if scale == 1.0:
                        inp = images
                    else:
                        inp = F.interpolate(images, scale_factor=scale, mode='bilinear', align_corners=False)
                    logits_list.append(model(inp) if not isinstance(model(inp), tuple) else model(inp)[0])
                    # Optional: Add flip TTA here
                logits = torch.stack(logits_list).mean(dim=0)
            else:
                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

            # Calc Loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calc Accuracy
            c, t = calculate_accuracy(logits, labels)
            total_correct += c
            total_samples += t

    # Aggregate metrics in DDP  (Optional but good for monitoring)
    if is_ddp:
        # Pack into a tensor for all_reduce: [loss, correct, samples]
        metrics = torch.tensor([total_loss, total_correct, total_samples], device=device)
        dist.all_reduce(metrics)

        # Loss is sum of means per GPU, so divide by world_size * num_batches is roughly avg
        # Or better: just average the "total_loss" part by world_size, then divide by loader length
        # NOTE: Since loaders have same length (roughly), simple average is fine.

        final_loss = metrics[0].item() / len(val_loader) / dist.get_world_size()
        final_correct = metrics[1].item()
        final_samples = metrics[2].item()
    else:
        final_loss = total_loss / len(val_loader)
        final_correct = total_correct
        final_samples = total_samples

    acc = 100. * final_correct / final_samples if final_samples > 0 else 0.0
    return final_loss, acc


def get_scheduler(optimizer, train_config):
    """Cosine Annealing with Warmup"""
    num_epochs = train_config['num_epochs']
    warmup_epochs = train_config.get('warmup_epochs', 5)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=warmup_epochs,
        cycle_limit=1,
        t_in_epochs=True,
    )
    return scheduler


# ==============================================================================
# 4. Main Training Pipeline & Logging
# ==============================================================================

def train_with_config(config_path: str):
    # --- 1. Init Env ---
    is_ddp, rank, local_rank, world_size, device = init_distributed_mode()

    config = load_config(config_path)
    config_name = Path(config_path).stem

    # Set Seed (Important: Base seed from config)
    seed = config.get('train', {}).get('seed', 42)
    set_seed(seed, rank)

    # Logging Setup (Only on Rank 0)
    log_dir = None
    csv_path = None
    tb_writer = None

    if is_main_process(rank):
        log_dir_root = config.get('logging', {}).get('log_dir', f"runs/{config_name}")
        if log_dir_root is None: log_dir_root = f"runs/{config_name}"
        log_dir = Path(log_dir_root)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Logs will be saved to: {log_dir}")

        if config.get('logging', {}).get('tensorboard', True):
            tb_writer = SummaryWriter(log_dir=log_dir)

        csv_path = log_dir / "training_log.csv"
        with open(csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'LR'])

    # --- 2. Data Loaders ---
    model_config = config['model']
    train_config = config['train']
    data_config = config['data']

    dataset_to_use = data_config.get('root_dir', "data/pet_cls_training")
    input_size_cfg = data_config.get('input_size', 224)
    img_size = input_size_cfg[0] if isinstance(input_size_cfg, list) else int(input_size_cfg)

    # Create Samplers for DDP
    train_sampler = build_datasampler(dataset_to_use, "train", True,
                                      ldre_cfg=data_config.get('ldre_cfg')) if is_ddp else None
    val_sampler = build_datasampler(dataset_to_use, "val", False,
                                    ldre_cfg=data_config.get('ldre_cfg')) if is_ddp else None

    # Note: When using DDP, batch_size in dataloader is "per GPU"
    train_loader = build_dataloader(
        root_dir=dataset_to_use, batch_size=train_config['batch_size'], split="train",
        img_size=img_size, sampler=train_sampler, ldre_cfg=data_config.get('ldre_cfg')
    )
    val_loader = build_dataloader(
        root_dir=dataset_to_use, batch_size=train_config['batch_size'], split="val",
        img_size=img_size, sampler=val_sampler, ldre_cfg=data_config.get('ldre_cfg')
    )

    actual_num_classes = train_loader.dataset.num_classes
    if is_main_process(rank):
        print(f"🔍 Detected classes: {actual_num_classes}")

    # --- 3. Model Init ---
    use_baseline = config.get('use_baseline', False)

    if use_baseline:
        if is_main_process(rank): print("🏛️  MODE: Baseline (MobileNetV2)")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, actual_num_classes)
    else:
        model_type = model_config.get('type', 'petnet')
        if is_main_process(rank): print(f"🚀 MODE: Custom {model_type}")

        if model_type == 'petnet':
            # Pass actual_num_classes via config
            petnet_params = {k: v for k, v in model_config.items()
                             if k in ['stage_repeats', 'attn_cfg', 'selfkd_cfg', 'max_pets_per_image']}
            model = PetNet(num_classes=actual_num_classes, **petnet_params)

            if 'pretrained_weights_path' in model_config:
                model.load_pretrained_weights(model_config['pretrained_weights_path'])

        elif model_type in ['efficientnet_b0', 'mobileone_s0']:
            model = timm.create_model(model_type, pretrained=True, num_classes=actual_num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Wrap DDP  (Optional but good for monitoring)
    if is_ddp:
        # Use find_unused_parameters=True only if necessary (e.g. self-kd skips some branches)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- 4. Optimizer & Loss ---
    # MixUp Setup
    mixup_fn = None
    criterion = None
    mixup_args = train_config.get('mixup_args')

    if mixup_args and (mixup_args.get('prob', 0.0) > 0):
        if is_main_process(rank): print("✨ MixUp/CutMix Enabled")
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

    # Optimizer
    if train_config['optimizer'].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'],
                                weight_decay=train_config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=0.9,
                              weight_decay=train_config['weight_decay'])

    scheduler = get_scheduler(optimizer, train_config)


    if train_config.get('resume_from_checkpoint'):
        checkpoint_path = Path(train_config['resume_from_checkpoint'])
        if checkpoint_path.exists():
            print(f"🔁 Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            # Optional: Load optimizer state. Usually skipped for fine-tuning to reset LR.
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Checkpoint loaded successfully.")
        else:
            print(f"⚠️ Checkpoint file not found: {checkpoint_path}. Starting from scratch.")

    # --- 5. Training Loop ---
    best_acc = 0.0
    patience = train_config.get('patience', 10)
    min_delta = train_config.get('min_delta', 0.1)
    epochs_no_improve = 0
    val_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(train_config['num_epochs']):
        # Important for DDP shuffling   (DDP Sampler is not shuffled)
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            model_config, rank, is_ddp, mixup_fn
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, val_criterion, device, is_ddp
        )

        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']

        # Log & Save (Only Rank 0)
        if is_main_process(rank):
            print(f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

            # TensorBoard
            if tb_writer:
                tb_writer.add_scalar('Train/Loss', train_loss, epoch)
                tb_writer.add_scalar('Val/Accuracy', val_acc, epoch)
                tb_writer.add_scalar('Val/Loss', val_loss, epoch)
                tb_writer.add_scalar('Train/LR', current_lr, epoch)

            # CSV
            with open(csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch + 1, train_loss, val_loss, val_acc, current_lr])

            # Save Checkpoint & Early Stopping
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }

            if val_acc > best_acc + min_delta:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(save_dict, log_dir / "best.pt")
                print(f"🔥 New Best Model Saved: {best_acc:.2f}%")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch + 1}")
                    break

            # Regular Checkpoint
            if config.get('logging', {}).get('save_checkpoints', True):
                freq = config.get('logging', {}).get('checkpoint_freq', 10)
                if (epoch + 1) % freq == 0:
                    torch.save(save_dict, log_dir / f"checkpoint_ep{epoch + 1}.pt")

    if is_ddp:
        cleanup_distributed()

    if is_main_process(rank):
        print(f"✅ Training Finished. Best Acc: {best_acc:.2f}%")
        if tb_writer: tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/petnet_base.yaml')
    args = parser.parse_args()

    train_with_config(args.config)