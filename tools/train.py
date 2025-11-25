"""
Project: PetBuddy
Author: Bright Wang
File: tools/train.py
Location: tools/
====================================
Main Training Script for PetBuddy Pet Recognition Models

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
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import csv
import os
import random
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.petnet import PetNet
from utils.data_loader import build_dataloader


def set_seed(seed=42):
    """
    Set fixed seed for reproducibility.
    """
    # 1. Python random
    random.seed(seed)
    # 2. Numpy
    np.random.seed(seed)
    # 3. Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # 4. Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 5. Deterministic algorithms
    # Set to True for exact reproducibility (paper); False for speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔒 Random Seed set to: {seed}")


# ...

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
    return 100. * correct / total



def train_one_epoch(model, train_loader, optimizer, criterion, device, model_config, writer=None, epoch=0, mixup_fn=None):

    model.train()
    total_loss = 0.0
    mixup_mode = False

    # Safely retrieve SelfKD config
    selfkd_cfg = model_config.get('selfkd_cfg', {})
    if selfkd_cfg is None: selfkd_cfg = {}
    selfkd_enabled = selfkd_cfg.get('enable', False)


    for i, batch_data in enumerate(train_loader):
        images = batch_data['images'].to(device)
        labels = batch_data['labels'].to(device)

        # --- Key Upgrade: Apply MixUp/CutMix here ---
        # mixup_fn handles image blending and label smoothing automatically
        if mixup_fn is not None:
            mixup_mode = True
            images, labels = mixup_fn(images, labels)


        optimizer.zero_grad()
        output = model(images)

        if isinstance(output, tuple):
            logits, kd_loss = output
            if not selfkd_enabled:
                kd_loss = 0.0
        else:
            logits, kd_loss = output, 0.0

        # --- Key Upgrade: Criterion now handles soft targets ---
        cls_loss = criterion(logits, labels)
        loss = cls_loss + kd_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i == 0 and writer is not None:
            # Denormalize images for visualization
            img_vis = images[0].cpu().clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_vis = img_vis * std + mean

            # Log to TensorBoard
            writer.add_image('Augmented_Input', img_vis, global_step=epoch)



        if i % 100 == 0:

            writer.add_scalar('Train/Batch_Loss', loss.item(), epoch * len(train_loader) + i)
            # Note: Batch accuracy is noisy with MixUp; rely on Val Acc.
            if mixup_mode:
                print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            else:
                acc = calculate_accuracy(logits, labels)
                print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")


    return total_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, device, tta_scales=[1.0, 1.15, 0.9]):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in val_loader:
            images = batch_data['images'].to(device)
            labels = batch_data['labels'].to(device)

            all_logits = []

            for scale in tta_scales:
                if scale == 1.0:
                    scaled_imgs = images
                else:
                    scaled_size = int(images.shape[-1] * scale)
                    scaled_imgs = F.interpolate(images, size=scaled_size, mode='bilinear', align_corners=False)

                # # 1. Standard prediction
                logits_orig = model(scaled_imgs)
                # # 2. Flipped prediction
                logits_flip = model(torch.flip(scaled_imgs, dims=[3]))
                all_logits.append((logits_orig + logits_flip) / 2.0)

            # # 3. Average logits
            final_logits = torch.stack(all_logits).mean(dim=0)

            loss = criterion(final_logits, labels)
            total_loss += loss.item()
            _, predicted = final_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# Scheduler definition
def get_scheduler(optimizer, train_config, steps_per_epoch):
    """Combine Warmup and Cosine Annealing"""
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

def train_with_config(config_path: str = "configs/petnet_base.yaml"):


    config = load_config(config_path)
    config_name = Path(config_path).stem


    # --- 🌟 Insert: Set Random Seed ---
    # Prioritize config seed, default to 42.
    seed = config.get('train', {}).get('seed', 42)
    set_seed(seed)

    log_dir_root = config.get('logging', {}).get('log_dir', f"runs/{config_name}")
    # Handle None log_dir; use default path.
    if log_dir_root is None:
        log_dir_root = f"runs/{config_name}"
    log_dir = Path(log_dir_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Logs will be saved to: {log_dir}")

    # 1. TensorBoard Writer
    use_tb = config.get('logging', {}).get('tensorboard', True)
    tb_writer = SummaryWriter(log_dir=log_dir) if use_tb else None

    # 2. CSV File Path
    csv_path = log_dir / "training_log.csv"
    # Initialize CSV header (overwrite mode).
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Define logged columns.
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'LR'])

    # Check for Baseline mode (default: False).
    use_baseline = config.get('use_baseline', False)



    print(f"--- Starting Training for: {config_name} ---")

    if use_baseline:
        print("🏛️  MODE: Baseline (Official MobileNetV2)")
    else:
        print("🚀  MODE: Custom PetNet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = config['model']
    train_config = config['train']
    data_config = config['data']

    # --- 1. Build Dataloader (Reliable way to get class count) ---
    dataset_to_use = "pet_cls_training"
    print(f"📁 Using dataset: {dataset_to_use}")

    input_size_cfg = data_config.get('input_size', 224)
    if isinstance(input_size_cfg, list) or isinstance(input_size_cfg, tuple):
        img_size = input_size_cfg[0]  # Take first dimension, e.g., 256
    else:
        img_size = int(input_size_cfg)
    print(f"📏 Training with image size: {img_size}x{img_size}")

    train_loader = build_dataloader(
        root_dir=dataset_to_use,
        batch_size=train_config['batch_size'],
        split="train",
        img_size=img_size,
        ldre_cfg=data_config.get('ldre_cfg')
    )

    val_loader = build_dataloader(
        root_dir=dataset_to_use,
        batch_size=train_config['batch_size'],
        split="val",
        shuffle=False,  # No shuffle for validation
        ldre_cfg=data_config.get('ldre_cfg')
    )

    # --- 2. Update model config with actual class count ---
    actual_num_classes = train_loader.dataset.num_classes
    print(f"🔍 Detected {actual_num_classes} classes from the dataset.")
    model_config['num_classes'] = actual_num_classes

    # --- 3. Create Model ---
    # Filter parameters for PetNet constructor

    if use_baseline:
        # 🏛️ Baseline Mode: Load official MobileNetV2
        print("⬇️  Loading ImageNet pretrained weights for MobileNetV2...")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Modify classifier head for dataset classes.
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, actual_num_classes)

        model = model.to(device)
    else:
        petnet_params = {k: v for k, v in model_config.items()
                        if k in ['num_classes', 'stage_repeats', 'attn_cfg', 'selfkd_cfg', 'max_pets_per_image']}

        model = PetNet(**petnet_params).to(device)
        print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters.")

        if 'pretrained_weights_path' in model_config and model_config['pretrained_weights_path']:
            model.load_pretrained_weights(model_config['pretrained_weights_path'])

    mixup_fn = None
    criterion = None

    # Load MixUp parameters from config.
    mixup_args = train_config.get('mixup_args')
    if mixup_args and mixup_args.get('mixup_alpha', 0.) > 0. or mixup_args.get('cutmix_alpha', 0.) > 0.:
        print("✨ Enabling MixUp/CutMix")
        mixup_fn = Mixup(
            mixup_alpha=mixup_args.get('mixup_alpha', 0.8),
            cutmix_alpha=mixup_args.get('cutmix_alpha', 1.0),
            prob=mixup_args.get('prob', 1.0),
            switch_prob=mixup_args.get('switch_prob', 0.5),
            mode=mixup_args.get('mode', 'batch'),
            label_smoothing=train_config.get('label_smoothing', 0.1),
            num_classes=actual_num_classes
        )
        # Use SoftTargetCrossEntropy if MixUp/CutMix is enabled.
        print("🔥 Using SoftTargetCrossEntropy loss")
        criterion = SoftTargetCrossEntropy().to(device)
    else:
        print("✅ Using standard CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(label_smoothing=train_config.get('label_smoothing', 0.1)).to(device)

    # --- 4. Setup Loss and Optimizer ---
    # criterion = nn.CrossEntropyLoss(label_smoothing=train_config.get('label_smoothing', 0.0))

    if train_config['optimizer'].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'],
                                weight_decay=train_config['weight_decay'])
    else:  # Fallback to SGD
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=0.9,
                              weight_decay=train_config['weight_decay'])

    scheduler = get_scheduler(optimizer, train_config, len(train_loader))


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

    # --- 5. Training Loop & Early Stopping ---
    best_acc = 0.0
    epochs_no_improve = 0
    patience = train_config.get('patience', 10)
    min_delta = train_config.get('min_delta', 0.1)

    val_criterion = nn.CrossEntropyLoss().to(device)


    for epoch in range(train_config['num_epochs']):



        # Print trainable parameter stats.
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🎯 Trainable parameters: {trainable_params / 1e6:.2f}M")


        print(f"\n--- Epoch {epoch + 1}/{train_config['num_epochs']} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_config,
                                     tb_writer,epoch,mixup_fn)
        val_loss, val_acc = validate_one_epoch(model, val_loader, val_criterion, device)

        scheduler.step(epoch + 1)


        print(f"Summary: Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")


        print(f"Epoch Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR updated to: {current_lr:.6f}")

        # --- 🌟 Write Logs ---

        # A. Write TensorBoard (for monitor instance)
        if tb_writer:
            tb_writer.add_scalar('Train/Loss', train_loss, epoch)
            tb_writer.add_scalar('Val/Loss', val_loss, epoch)
            tb_writer.add_scalar('Val/Accuracy', val_acc, epoch)
            tb_writer.add_scalar('Train/LR', current_lr, epoch)

        # B. Write CSV (for report charts)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc, current_lr])

        # Early Stopping and Model Saving logic
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            epochs_no_improve = 0
            print(f"🚀 New best accuracy! Saving model to runs/{config_name}/best.pt")

            save_dir = Path(f"runs/{config_name}")
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, save_dir / "best.pt")
        else:
            epochs_no_improve += 1
            print(f"📉 No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("⏹️ Early stopping triggered!")
                break

    print(f"\n🎉 Training finished! Best validation accuracy: {best_acc:.2f}%")

    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train PetNet model.')
    parser.add_argument('--config', type=str, default='configs/petnet_base.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    train_with_config(args.config)