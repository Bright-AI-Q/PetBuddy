from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import optuna
import gc
import os

use_subset = True

def cleanup_memory():
    """Clean up GPU memory between trials"""
    gc.collect()
    torch.cuda.empty_cache()

def objective(trial):
    """Optuna objective function to minimize"""
    
    # Clean memory before each trial
    cleanup_memory()
    
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 24, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 48])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [1, 2])
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [16, 32])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    max_length = trial.suggest_categorical("max_length", [512, 640, 768])
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": "pet_knowlege/dataset_train.jsonl",
        "validation": "pet_knowlege/dataset_val.jsonl"
    })
    if use_subset:
        train_subset = dataset["train"].shuffle(seed=42).select(range(len(dataset["train"]) // 5))
        val_subset = dataset["validation"].shuffle(seed=42).select(range(len(dataset["validation"]) // 5))
    
    def formatting_func(example):
        instruction = example["instruction"]
        output = example["output"]
        text = f"{instruction}\n{output}" if instruction else output
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens['input_ids'])
    
    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj"
        ],
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM"
    )
    
    # Training args
    output_dir = f"qwen-petnet-trial-{trial.number}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        num_train_epochs=1,  # Use 1 epoch for faster tuning
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=1000,  # Don't save during tuning
        save_total_limit=0,  # Don't keep checkpoints
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=False,  # Speed up
        report_to="none",  # Disable wandb/tensorboard
        max_steps=100
    )
    
    # Callback to report to Optuna
    class OptunaCallback(TrainerCallback):
        def __init__(self, trial):
            self.trial = trial
            
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            # Report intermediate value to Optuna
            self.trial.report(metrics["eval_loss"], state.global_step)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()
    
    if use_subset:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            peft_config=peft_config,
            formatting_func=formatting_func,
            callbacks=[OptunaCallback(trial)],
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            formatting_func=formatting_func,
            callbacks=[OptunaCallback(trial)],
        )
    
    try:
        # Train
        trainer.train()
        
        # Get final eval loss
        eval_results = trainer.evaluate()
        final_loss = eval_results["eval_loss"]
        
        # Cleanup
        del model
        del trainer
        cleanup_memory()
        
        # Remove checkpoint directory
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        return final_loss
        
    except optuna.TrialPruned:
        # Cleanup on pruned trial
        del model
        del trainer
        cleanup_memory()
        
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        raise
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        
        # Cleanup
        try:
            del model
            del trainer
        except:
            pass
        cleanup_memory()
        
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        # Return a high loss to indicate failure
        return float('inf')


# Create Optuna study
study = optuna.create_study(
    direction="minimize",
    study_name="qwen-petnet-tuning",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=30,
    ),
)

# Run optimization
print("Starting hyperparameter optimization...")
print("This will take several hours depending on n_trials")

study.optimize(
    objective,
    n_trials=20,  # Adjust based on time/budget
    timeout=None,
    n_jobs=1,  # Must be 1 for GPU
    show_progress_bar=True,
)

# Print results
print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
print(f"Best trial: {study.best_trial.number}")
print(f"Best eval loss: {study.best_value:.4f}")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Save results
import json
with open("optuna_results.json", "w") as f:
    json.dump({
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state)
            }
            for trial in study.trials
        ]
    }, f, indent=2)

print("\n✓ Results saved to optuna_results.json")

# Visualizations (optional - requires plotly)
try:
    import plotly
    
    # Optimization history
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_html("optuna_history.html")
    
    # Parameter importances
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("optuna_importances.html")
    
    # Parallel coordinate plot
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.write_html("optuna_parallel.html")
    
    print("✓ Visualizations saved to optuna_*.html")
except ImportError:
    print("Install plotly for visualizations: pip install plotly")

# Train final model with best parameters
print("\n" + "="*70)
print("Training final model with best hyperparameters...")
print("="*70)

best_params = study.best_params

# Now train the final model with more epochs
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("json", data_files={
    "train": "pet_knowlege/dataset_train.jsonl",
    "validation": "pet_knowlege/dataset_val.jsonl"
})

def formatting_func(example):
    instruction = example["instruction"]
    output = example["output"]
    text = f"{instruction}\n{output}" if instruction else output
    tokens = tokenizer(text, truncation=True, max_length=best_params["max_length"])
    return tokenizer.decode(tokens['input_ids'])

peft_config = LoraConfig(
    r=best_params["lora_r"],
    lora_alpha=best_params["lora_alpha"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj"
    ],
    lora_dropout=best_params["lora_dropout"],
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="qwen-petnet-final",
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
    gradient_checkpointing=True,
    learning_rate=best_params["learning_rate"],
    lr_scheduler_type="cosine",
    num_train_epochs=3,  # Full training
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    warmup_ratio=best_params["warmup_ratio"],
    group_by_length=True,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    formatting_func=formatting_func,
)

trainer.train()
trainer.save_model("qwen-petnet-final")

print("\n🎉 Training complete with optimized hyperparameters!")