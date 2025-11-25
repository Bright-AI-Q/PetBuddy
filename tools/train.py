"""
Training script with configuration file support
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import yaml
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

sys.path.append(str(Path(__file__).parent.parent))

from models.petnet import PetNet
from utils.data_loader import build_dataloader
import copy

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file doesn't exit: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def train_with_config(config_path: str = "configs/petnet_base.yaml"):
    """Train model using configuration file"""
    # load the config file
    config = load_config(config_path)
    config_name = Path(config_path).stem
    print(f"Loading config file: {config_path}")
    print(f"Config name: {config_name}")
    print(f"Config content: {config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model_config = config['model']
    train_config = config['train']


    dataset_to_use = "pet_cls_training" if (Path('data') / 'pet_cls_training').exists() else "merged_cls_dataset"
    print(f"📁 Using dataset: {dataset_to_use}")

    # Get number of classes from dataset config - more reliable method
    def get_actual_num_classes(dataset_name):
        """More reliably get actual number of classes in dataset"""
        from utils.data_utils import get_num_classes
        try:
            return get_num_classes(dataset_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Failed to get class count from dataset.yaml: {e}")
            print("Falling back to directory counting method...")



    actual_num_classes = get_actual_num_classes(dataset_to_use)
    print(f"🔍 Detected {actual_num_classes} classes in dataset, updating model config")
    model_config['num_classes'] = actual_num_classes

    # 创建模型
    model = PetNet(
        num_classes=model_config['num_classes'],
        stage_repeats=model_config['stage_repeats'],
        model_cfg=model_config['model_cfg'],
        attn_cfg=model_config['attn_cfg'],
        selfkd_cfg=model_config['selfkd_cfg']
    )
    model.to(device)

    # Create teacher model for knowledge distillation (if SelfKD enabled)
    teacher_model = None
    if model_config['selfkd_cfg']['enable']:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()  # Teacher model always in eval mode
        # Set teacher model parameters to not require gradients
        for param in teacher_model.parameters():
            param.requires_grad = False
    label_smoothing = train_config.get('label_smoothing', 0.0)
    print(f"🛠️ Using Label Smoothing with epsilon = {label_smoothing}")

    # Setup loss function and optimizer
    if augment_cfg['enable']:
        # Mixup / CutMix → labels are soft
        criterion = SoftTargetCrossEntropy()
    else:
        # Normal hard labels
        label_smoothing = train_config.get("label_smoothing", 0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if train_config['optimizer'] == "adam":
        optimizer = optim.Adam(model.parameters(),
                             lr=train_config['learning_rate'],
                             weight_decay=train_config['weight_decay'])
    elif train_config['optimizer'] == "adamw":
        optimizer = optim.AdamW(model.parameters(),
                              lr=train_config['learning_rate'],
                              weight_decay=train_config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(),
                            lr=train_config['learning_rate'],
                            weight_decay=train_config['weight_decay'],
                            momentum=0.9)
        # --- ADD SCHEDULER DEFINITION HERE ---
    print("💡 Using CosineAnnealingLR scheduler.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['num_epochs'], eta_min=1e-6)
    # Build data loader - using dataset path from config
    dataset_config = config['data']
    print(f"🔄 Building training data loader, dataset path: {dataset_config['dataset']}")
    train_loader = build_dataloader(
        root_dir=dataset_config['dataset'],
        batch_size=train_config['batch_size'],
        shuffle=True,
        transform_type="yolo_to_cls",
        split="train",
        ldre_cfg=dataset_config.get('ldre_cfg')
    )
    print(f"✅ Training data loader built, total samples: {len(train_loader.dataset)}")

    val_loader = build_dataloader(
        root_dir=dataset_config['dataset'],
        batch_size=train_config['batch_size'],
        shuffle=False,
        transform_type="yolo_to_cls",
        split="val",
        ldre_cfg=dataset_config.get('ldre_cfg')
    )

    # 训练循环与Early Stopping
    best_acc = 0.0
    num_epochs = train_config['num_epochs']
    patience = train_config.get('patience', 10)
    min_delta = train_config.get('min_delta', 0.001)

    # Early Stopping
    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False

    # Mixup augmentation
    augment_cfg = train_config['augmentation']
    if augment_cfg["enable"]:
        mixup_fn = Mixup(
            mixup_alpha=augment_cfg['mixup'],
            cutmix_alpha=augment_cfg['cutmix'],
            prob=augment_cfg['mixup_prob'],
            switch_prob=augment_cfg['switch_prob'],
            mode=augment_cfg['mixup_mode'],
            label_smoothing=augment_cfg['label_smoothing'],
            num_classes=actual_num_classes,
        )

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        # training phrase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        accumulation_steps = 8  # Simulate a batch size of 8 * 8 = 64
        print(f"⚡ Using Gradient Accumulation with {accumulation_steps} steps to simulate a batch size of "
              f"{train_config['batch_size'] * accumulation_steps}")

        for i, batch_data in enumerate(train_loader):
            inputs, labels = batch_data['images'], batch_data['labels']
            # Do mix-up for if augmentation is enabled    
            if augment_cfg['enable']:
                inputs, labels = mixup_fn(inputs, labels)
                
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, stage_logits = outputs
            else:
                logits = outputs
                stage_logits = None
            loss = criterion(logits, labels)


            kd_loss = 0
            if model_config['selfkd_cfg']['enable'] and stage_logits is not None and teacher_model is not None:
                kd_weights = model_config['selfkd_cfg']['w']


                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    if isinstance(teacher_outputs, tuple):
                        _, teacher_stage_logits = teacher_outputs
                    else:
                        teacher_stage_logits = None


                if teacher_stage_logits is not None:
                    for stage_idx, (student_feat, teacher_feat, w) in enumerate(zip(stage_logits, teacher_stage_logits, kd_weights)):
                        if stage_idx < len(model.selfkd_modules) and model.selfkd_modules[stage_idx] is not None:
                            # 计算当前阶段的KL散度损失
                            student_logits = model.selfkd_modules[stage_idx].pools[0](student_feat)
                            student_logits = student_logits.squeeze(-1).squeeze(-1)

                            teacher_logits = model.selfkd_modules[stage_idx].pools[0](teacher_feat)
                            teacher_logits = teacher_logits.squeeze(-1).squeeze(-1)


                            stage_kd_loss = F.kl_div(
                                F.log_softmax(student_logits / model.selfkd_modules[stage_idx].T, dim=1),
                                F.softmax(teacher_logits / model.selfkd_modules[stage_idx].T, dim=1),
                                reduction='batchmean'
                            ) * (model.selfkd_modules[stage_idx].T ** 2)

                            kd_loss += w * stage_kd_loss

                loss += kd_loss
            loss = loss / accumulation_steps
            loss.backward()


            # --- Perform optimizer step only after accumulation_steps ---
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Reset gradients for the next accumulation cycle

                # EMA update for the teacher model should also happen here
                if model_config['selfkd_cfg']['enable'] and teacher_model is not None:
                    alpha = model_config['selfkd_cfg']['alpha']
                    with torch.no_grad():
                        for param, teacher_param in zip(model.parameters(), teacher_model.parameters()):
                            teacher_param.data.mul_(alpha).add_((1 - alpha) * param.data)

            # --- Your logging and accuracy calculation logic can remain the same ---
            running_loss += loss.item() * accumulation_steps  # Un-normalize for logging
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 0:
                acc = 100. * correct / total
                print(f"[Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}] "
                      f"loss: {running_loss / 100:.3f}, acc: {acc:.2f}%")
                running_loss = 0.0


        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                inputs, labels = batch_data['images'], batch_data['labels']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Acc: {val_acc:.2f}%")
        scheduler.step()
        print(f"   LR Scheduler stepped. New LR: {scheduler.get_last_lr()[0]:.6f}")
        # Early Stopping
        if val_acc > best_acc + min_delta:

            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0


            log_dir = config['logging']['log_dir']
            if not Path(log_dir).is_absolute():

                project_root = Path(__file__).parent.parent
                save_dir = project_root / log_dir
            else:
                save_dir = Path(log_dir)

            save_dir.mkdir(parents=True, exist_ok=True)


            config_name = Path(config_path).stem  #
            config_save_dir = save_dir / config_name
            config_save_dir.mkdir(parents=True, exist_ok=True)


            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"best_{timestamp}.pt"

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'train_config': train_config,
                'val_acc': best_acc,
                'epoch': best_epoch,
                'train_timestamp': timestamp
            }, config_save_dir / model_filename)


            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'train_config': train_config,
                'val_acc': best_acc,
                'epoch': best_epoch,
                'train_timestamp': timestamp
            }, config_save_dir / "best.pt")

            print(f"✅ Best model saved to {config_save_dir}/{model_filename}, validation accuracy: {val_acc:.2f}%")
            print(f"✅ Also saved as {config_save_dir}/best.pt")
        else:

            epochs_no_improve += 1
            print(f"📉 No improvement for {epochs_no_improve}/{patience} epochs, best accuracy: {best_acc:.2f}%")


            if epochs_no_improve >= patience:
                early_stop = True
                print(f"⏹️  Early stopping triggered! Best accuracy {best_acc:.2f}% achieved at epoch {best_epoch}")

    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"⏱️  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {hours}:{minutes:02d}:{seconds:02d}")


    config_name = Path(config_path).stem
    log_dir = config['logging']['log_dir']
    if not Path(log_dir).is_absolute():
        project_root = Path(__file__).parent.parent
        log_dir = project_root / log_dir / config_name
    else:
        log_dir = Path(log_dir) / config_name

    log_dir.mkdir(parents=True, exist_ok=True)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_log_{timestamp}.txt"


    general_log_file = log_dir / "experiment_log.txt"


    log_content = f'''Experiment: {config_name}
Description: Pet classification model training
Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}:{minutes:02d}:{seconds:02d}
Command: python {__file__} --config {config_path}
------------------------------------------------------------

Configuration:
{yaml.dump(config, allow_unicode=True)}

Training Results:
- Best validation accuracy: {best_acc:.2f}%
- Best epoch: {best_epoch}
- Total training epochs: {epoch + 1}
- Early stopping: {'Yes' if early_stop else 'No'}

Model Info:
- Save path: {config_save_dir if 'config_save_dir' in locals() else 'N/A'}
- Parameter count: {sum(p.numel() for p in model.parameters())}
- Device: {device}

Dataset Info:
- Dataset: {dataset_to_use}
- Training samples: {len(train_loader.dataset) if 'train_loader' in locals() else 'N/A'}
- Validation samples: {len(val_loader.dataset) if 'val_loader' in locals() else 'N/A'}
- Number of classes: {actual_num_classes}
'''

    with open(log_file, 'w') as f:
        f.write(log_content)


    with open(general_log_file, 'w') as f:
        f.write(log_content)

    print(f"📝 Detailed experiment log saved to: {log_file}")
    print(f"📝 General experiment log saved to: {general_log_file}")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description='Train pet classification model using configuration file')
    parser.add_argument('--config', type=str, default='configs/petnet_base.yaml',
                      help='Configuration file path (default: configs/petnet_base.yaml)')

    args = parser.parse_args()
    start_time = datetime.now()
    print(f"⏱️  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    train_with_config(args.config)