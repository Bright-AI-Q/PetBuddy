"""
Optuna hyperparameter optimization training script
Modified from tools/train.py with Optuna integration
"""
import os
import sys
from pathlib import Path

import optuna
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import yaml
from optuna.trial import TrialState
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_float32_matmul_precision('high')

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

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
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_actual_num_classes(dataset_name):
    """Get the actual number of classes in the dataset reliably"""
    from utils.data_utils import get_num_classes
    actual_num_classes = get_num_classes(dataset_name)
    print(f"📊 Actual number of classes in dataset '{dataset_name}': {actual_num_classes}")
    return actual_num_classes

def objective(trial: optuna.trial.Trial, distributed=False):
    """Optuna objective function for hyperparameter optimization"""

    if distributed:
        trial = optuna.integration.TorchDistributedTrial(trial)

    optuna_config = load_config("configs/optuna_tuning.yaml")
    tuner_config = optuna_config["tuner"]

    # --- Phase 1: Core parameter optimization ---
    lr = trial.suggest_float("lr", tuner_config["lr_low"], tuner_config["lr_high"], log=True)
    weight_decay = trial.suggest_float("weight_decay", tuner_config["weight_decay_low"], tuner_config["weight_decay_high"], log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])

    # --- Load base configuration ---
    petnet_config = load_config("configs/petnet_base.yaml")

    # Override configuration values with Optuna suggested parameters
    petnet_config['train']['learning_rate'] = lr
    petnet_config['train']['weight_decay'] = weight_decay
    petnet_config['train']['optimizer'] = optimizer_name.lower()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Starting trial {trial.number}: lr={lr:.2e}, wd={weight_decay:.2e}, opt={optimizer_name}")

    # Get parameters from petnet_config
    model_config = petnet_config['model']
    train_config = petnet_config['train']
    data_config = petnet_config['data']

    # Check dataset
    dataset_to_use = "pet_cls_training" if (Path('data') / 'pet_cls_training').exists() else "merged_cls_dataset"
    actual_num_classes = get_actual_num_classes(dataset_to_use)

    # Create model
    model = PetNet(
        stage_repeats=model_config['stage_repeats'],
        num_classes=actual_num_classes,
        attn_cfg=model_config['attn_cfg'],
        selfkd_cfg=model_config['selfkd_cfg']
    )
    model.to(device)

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create optimizer
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config['num_epochs'],
        eta_min=1e-6
    )

    # Create data loaders - use correct parameters
    root_dir = data_config["root_dir"]

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = build_datasampler(
            root_dir=root_dir,
            shuffle=True,
            split="train",
            transform_type="yolo_to_cls",
            ldre_cfg=data_config.get('ldre_cfg')
        )
        val_sampler = build_datasampler(
            root_dir=root_dir,
            shuffle=False,
            split="val",
            transform_type="yolo_to_cls",
            ldre_cfg=data_config.get('ldre_cfg')
        )

    num_workers = data_config["num_workers"]
    train_loader = build_dataloader(
        root_dir=root_dir,
        batch_size=train_config['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        transform_type="yolo_to_cls",
        split="train",
        ldre_cfg=data_config.get('ldre_cfg')
    )
    val_loader = build_dataloader(
        root_dir=root_dir,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        transform_type="yolo_to_cls",
        split="val",
        ldre_cfg=data_config.get('ldre_cfg')
    )

    # Training parameters
    num_epochs = train_config['num_epochs']
    accumulation_steps = train_config['gradient_accumulation_steps']  # Gradient accumulation steps

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        correct = 0
        total = 0

        for i, batch_data in enumerate(train_loader):
            inputs, labels = batch_data['images'], batch_data['labels']
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Forward pass
                outputs = model(inputs)

                # Handle model output (could be tuple or tensor)
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Extract classification logits
                    stage_outs = outputs[1]  # Intermediate outputs for SelfKD
                else:
                    logits = outputs

                # Calculate loss
                loss = F.cross_entropy(logits, labels,
                                      label_smoothing=train_config.get('label_smoothing', 0.0))

                # Gradient accumulation
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            # Optimizer step
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Record statistics
            running_loss += loss.item() * accumulation_steps
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data['images'], batch_data['labels']
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(device_type=str(device), enabled=(str(device) == "cuda")):
                    outputs = model(inputs)
                    # Handle model output (could be tuple or tensor)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]  # Extract classification logits
                    else:
                        logits = outputs

                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        if distributed:
            val_correct = torch.tensor([val_correct], dtype=torch.long, device=device)
            val_total = torch.tensor([val_total], dtype=torch.long, device=device)
            dist.all_reduce(val_correct)
            dist.all_reduce(val_total)
            val_correct = val_correct.item()
            val_total = val_total.item()

        val_acc = 100. * val_correct / val_total
        best_acc = max(best_acc, val_acc)

        # Learning rate scheduling
        scheduler.step()

        # --- Optuna pruning ---
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.2f}%, Best: {best_acc:.2f}%")

    return best_acc

def main():
    """Main function to run Optuna optimization"""

    distributed, rank, world_size = setup_distributed_env()

    if distributed:
        print(f"Running in distributed environment: rank {rank}/{world_size}")
        dist.barrier()
    else:
        print(f"Running in single machine, default to rank 0")

    config = load_config("configs/optuna_tuning.yaml")

    config_pruner = config["pruner"]
    if config_pruner["type"] == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=config_pruner["min_resource"],
            max_resource=config_pruner["max_resource"],
            reduction_factor=config_pruner["reduction_factor"],
        )
    else:
        # Default to median pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config_pruner["n_startup_trials"],
            n_warmup_steps=config_pruner["n_warmup_steps"],
            interval_steps=config_pruner["interval_steps"],
        )

    # Run by default in single machine, otherwise main machine runs the study
    # Create study object
    study = optuna.create_study(
        study_name="petnet_optimization_phase1",
        direction="maximize",
        pruner=pruner,
        storage="sqlite:///petnet_study.db",
        load_if_exists=True
    )

    print("🔬 Starting Optuna hyperparameter optimization (Phase 1: Core parameters)")
    print("📊 Optimizing: Learning rate, weight decay, optimizer type")
    print("📈 Objective: Maximize validation accuracy")

    # Start optimization
    n_trials = config["tuner"]["n_trials"]
    study.optimize(lambda t: objective(t, distributed), n_trials=n_trials)

    if rank == 0:
        # If running on single machine, this will always run, otherwise run on master node
        # Print result statistics
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("\n📊 Optimization statistics:")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Pruned trials: {len(pruned_trials)}")
        print(f"   Completed trials: {len(complete_trials)}")

        # Print best parameters
        best_trial = study.best_trial
        print(f"\n🏆 Best trial:")
        print(f"   Validation accuracy: {best_trial.value:.2f}%")
        print(f"   Parameters:")
        for key, value in best_trial.params.items():
            print(f"     {key}: {value}")

        # Save best configuration
        best_config = load_config("configs/petnet_base.yaml")
        best_config['train']['learning_rate'] = best_trial.params['lr']
        best_config['train']['weight_decay'] = best_trial.params['weight_decay']
        best_config['train']['optimizer'] = best_trial.params['optimizer'].lower()

        best_config_path = "configs/petnet_optimized_phase1.yaml"
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)

        print(f"\n💾 Best configuration saved to: {best_config_path}")

        # --- Generate Optuna visualization charts ---
        print("\n📊 Generating Optuna visualization charts...")
        try:
            # 1. Optimization history plot
            print("📈 Generating optimization history plot...")
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_image("optuna_optimization_history.png")
            fig_history.write_image("optuna_optimization_history.svg")
            print("✅ Optimization history plot saved: optuna_optimization_history.png, optuna_optimization_history.svg")

            # 2. Parameter importance plot
            print("📊 Generating parameter importance plot...")
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image("optuna_parameter_importance.svg")
            print("✅ Parameter importance plot saved: optuna_parameter_importance.svg")

            # 3. Parallel coordinate plot
            print("📐 Generating parallel coordinate plot...")
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            fig_parallel.write_image("optuna_parallel_coordinate.svg")
            print("✅ Parallel coordinate plot saved: optuna_parallel_coordinate.svg")

            # 4. Slice plot
            print("🔍 Generating slice plot...")
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.write_image("optuna_slice_plot.svg")
            print("✅ Slice plot saved: optuna_slice_plot.svg")

            print("\n🎉 All visualization charts generated successfully!")

        except ImportError as e:
            print(f"⚠️  Visualization dependencies not installed: {e}")
            print("Please install required dependencies: pip install plotly kaleido")
        except Exception as e:
            print(f"❌ Error generating charts: {e}")

        # Print detailed trial results table
        print("\n📋 Detailed trial results:")
        print("Trial | Accuracy(%) | Learning Rate | Weight Decay | Optimizer")
        print("-" * 70)
        for i, trial in enumerate(study.trials):
            if trial.state == TrialState.COMPLETE:
                acc = trial.value
                params = trial.params
                print(f"{i:5d} | {acc:8.2f} | {params.get('lr', 'N/A'):.2e} | {params.get('weight_decay', 'N/A'):.2e} | {params.get('optimizer', 'N/A')}")

        # Create parameter table for paper
        print("\n📄 Paper parameter table:")
        print("| Hyperparameter | Optimized Value |")
        print("| :--- | :--- |")
        print(f"| Optimizer | {best_trial.params['optimizer']} |")
        print(f"| Learning Rate | {best_trial.params['lr']:.2e} |")
        print(f"| Weight Decay | {best_trial.params['weight_decay']:.2e} |")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()