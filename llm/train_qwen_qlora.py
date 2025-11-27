from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import torch
import os
import gc
from pathlib import Path
import yaml
import argparse
import json

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA = "pet_knowlege/dataset_train.jsonl"
VAL_DATA = "pet_knowlege/dataset_val.jsonl"
RESULTS_FILE = "optuna_semantic_bert_results.json"
OUTPUT_DIR = "qwen-petnet-final"

def cleanup_memory():
    """Clean GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
        
def main():
    parser = argparse.ArgumentParser(description='Train Qwen model with optimized hyperparameters')
    parser.add_argument('--results', type=str, default=RESULTS_FILE,
                        help='Path to Optuna results JSON file')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--train_data', type=str, default=TRAIN_DATA,
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default=VAL_DATA,
                        help='Path to validation data')
    args = parser.parse_args()
    
    best_config = load_config("configs/qwen_base.yaml")
    best_params = best_config['train']
    
    # Clean memory before starting
    print("Cleaning GPU memory...")
    cleanup_memory()
    print("✓ Memory cleaned\n")
    
    # Load model
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"✓ Model loaded: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files={
        "train": args.train_data,
        "validation": args.val_data
    })
    print(f"✓ Training samples: {len(dataset['train'])}")
    print(f"✓ Validation samples: {len(dataset['validation'])}")
    
    # Formatting function
    def formatting_func(example):
        instruction = example["instruction"]
        output = example["output"]
        text = f"{instruction}\n{output}" if instruction else output
        tokens = tokenizer(text, truncation=True, max_length=best_params["max_length"])
        return tokenizer.decode(tokens['input_ids'])
    
    # LoRA configuration
    print("\nConfiguring LoRA...")
    peft_config = LoraConfig(
        r=best_params["lora_r"],
        lora_alpha=best_params["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj"  # Full set for final training
        ],
        lora_dropout=best_params["lora_dropout"],
        task_type="CAUSAL_LM"
    )
    print(f"✓ LoRA rank: {best_params['lora_r']}")
    print(f"✓ LoRA alpha: {best_params['lora_alpha']}")
    print(f"✓ LoRA dropout: {best_params['lora_dropout']}")
    
    # Training arguments
    print("\nConfiguring training...")
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=1,  # Keep at 1 for 8GB GPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        learning_rate=best_params["learning_rate"],
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=150,
        save_steps=100,
        save_total_limit=2,  # Keep 2 checkpoints
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=best_params["warmup_ratio"],
        group_by_length=True,
        dataloader_pin_memory=False,
        metric_for_best_model="eval_loss",
    )
    print(f"✓ Learning rate: {best_params['learning_rate']:.6f}")
    print(f"✓ Gradient accumulation: {best_params['gradient_accumulation_steps']}")
    print(f"✓ Warmup ratio: {best_params['warmup_ratio']}")
    print(f"✓ Max length: {best_params['max_length']}")
    print(f"✓ Training epochs: {args.epochs}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    print("✓ Trainer initialized")
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    final_output = f"{args.output}/final"
    trainer.save_model(final_output)
    print(f"✓ Model saved to: {final_output}")
    
    # Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "best_params": best_params,
        "num_train_samples": len(dataset["train"]),
        "num_val_samples": len(dataset["validation"]),
        "num_epochs": args.epochs,
        "output_dir": final_output,
    }
    
    info_path = f"{final_output}/training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✓ Training info saved to: {info_path}")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Final model: {final_output}")
    print(f"Checkpoints: {args.output}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()