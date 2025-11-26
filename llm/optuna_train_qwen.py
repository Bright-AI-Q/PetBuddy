from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import torch
import optuna
import gc
import os
import shutil
import time
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
import numpy as np
import yaml
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA = "pet_knowlege/dataset_train.jsonl"
VAL_DATA = "pet_knowlege/dataset_val.jsonl"
N_TRIALS = 15  # Number of hyperparameter combinations to try
EVAL_SAMPLES = 20  # Number of samples to evaluate per trial
USE_SUBSET = False  # Whether to use only a subset of the training and validation set for faster optimization runs

# Load semantic model once (reuse across trials)
print("Loading semantic similarity model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Semantic model loaded")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_memory():
    """Aggressive GPU memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def generate_response(model, tokenizer, prompt, max_new_tokens=250):
    """Generate response from model for a given prompt"""
    # Get the underlying model (unwrap PEFT if needed)
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model
    
    base_model.eval()
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(base_model.device)
    attention_mask = inputs["attention_mask"].to(base_model.device)
    
    try:
        with torch.no_grad():
            # Use model's generate method directly
            outputs = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(text):].strip()
        return response
        
    except Exception as e:
        print(f"      Generation error: {e}")
        # Return empty string on failure
        return ""

def calculate_semantic_similarity(predictions, references):
    """
    Calculate average semantic similarity using Sentence-BERT
    Returns: score between 0 and 1
    """
    similarities = []
    
    for pred, ref in zip(predictions, references):
        emb_pred = semantic_model.encode(pred, convert_to_tensor=True)
        emb_ref = semantic_model.encode(ref, convert_to_tensor=True)
        similarity = util.cos_sim(emb_pred, emb_ref).item()
        similarities.append(similarity)
    
    return np.mean(similarities)

def calculate_bertscore(predictions, references):
    """
    Calculate BERTScore F1
    Returns: average F1 score between 0 and 1
    """
    P, R, F1 = bert_score(
        predictions, 
        references, 
        lang="en", 
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return F1.mean().item()

def evaluate_model(model, tokenizer, eval_dataset, num_samples=20):
    """
    Evaluate model using both semantic similarity and BERTScore
    Returns: combined score and detailed metrics
    """
    # CRITICAL: Get the actual model for generation
    # The trainer wraps the model, we need the underlying PEFT model
    if hasattr(model, 'model'):
        gen_model = model.model  # From trainer
    else:
        gen_model = model
    
    gen_model.eval()


    # OPTIONAL BUT RECOMMENDED: Enable cache for generation
    # (Gradient checkpointing usually disables this, making generation slow)
    if hasattr(gen_model, "config"):
        gen_model.config.use_cache = True
    
    # Sample random examples
    sample_indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()
    samples = eval_dataset.select(sample_indices)
    
    predictions = []
    references = []
    
    print(f"   Generating {num_samples} predictions...")
    for i, example in enumerate(samples):
        instruction = example["instruction"]
        expected = example["output"]
        
        # Generate prediction
        try:
            # Simple generation without helper function
            messages = [{"role": "user", "content": instruction}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt")
            
            # Ensure inputs are on the correct device
            input_ids = inputs["input_ids"].to(gen_model.device)
            attention_mask = inputs["attention_mask"].to(gen_model.device)
            
            with torch.no_grad():
                # Autocast handles the mixed precision automatically
                # needed for the 4-bit compute_dtype
                with torch.amp.autocast('cuda'): 
                    outputs = gen_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=250,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(text):].strip()
            
            predictions.append(generated)
            references.append(expected)
            
        except Exception as e:
            print(f"   Warning: Generation failed for example {i}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append("")
            references.append(expected)
    
    # [Rest of the function remains the same...]
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip()]
    
    if not valid_pairs:
        print("   ERROR: All generations failed!")
        return 0.0, {
            "semantic_similarity": 0.0,
            "bertscore_f1": 0.0,
            "combined_score": 0.0
        }
    
    valid_predictions, valid_references = zip(*valid_pairs)
    
    print(f"   Calculating semantic similarity on {len(valid_predictions)} valid samples...")
    semantic_score = calculate_semantic_similarity(valid_predictions, valid_references)
    
    print(f"   Calculating BERTScore...")
    bertscore_f1 = calculate_bertscore(valid_predictions, valid_references)
    
    combined_score = 0.5 * semantic_score + 0.5 * bertscore_f1
    
    metrics = {
        "semantic_similarity": semantic_score,
        "bertscore_f1": bertscore_f1,
        "combined_score": combined_score,
        "valid_samples": len(valid_predictions),
        "total_samples": num_samples
    }
    
    return combined_score, metrics

def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Callback to report to Optuna (MOVED OUTSIDE objective function)
class OptunaCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.trial.report(metrics["eval_loss"], state.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def objective(trial):
    """Optuna objective function to minimize"""
    
    # Clean memory before each trial
    cleanup_memory()
    
    # Suggest hyperparameters - REDUCED RANGES for memory safety
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 24])  # Removed 32
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32])  # Removed 48
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    batch_size = 1  # Fixed at 1 for memory safety
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [16, 32])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    max_length = trial.suggest_categorical("max_length", [512, 640])  # Removed 768
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*70}")
    print(f"  Learning Rate: {lr:.6f}")
    print(f"  LoRA Rank: {lora_r}")
    print(f"  LoRA Alpha: {lora_alpha}")
    print(f"  LoRA Dropout: {lora_dropout:.3f}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gradient Accumulation: {grad_accum}")
    print(f"  Warmup Ratio: {warmup_ratio:.3f}")
    print(f"  Max Length: {max_length}")
    print(f"{'='*70}\n")
    
    output_dir = f"qwen-petnet-trial-{trial.number}"
    
    try:
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
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("json", data_files={
            "train": TRAIN_DATA,
            "validation": VAL_DATA
        })
        
        if USE_SUBSET:
            # Use smaller subset for faster trials
            train_subset = dataset["train"].shuffle(seed=42).select(range(min(200, len(dataset["train"]) // 5)))
            val_subset = dataset["validation"].shuffle(seed=42).select(range(min(50, len(dataset["validation"]) // 5)))
        else:
            train_subset = dataset["train"]
            val_subset = dataset["validation"]
        
        def formatting_func(example):
            question = example["instruction"]
            reference_answer = example["output"]

            # Build a chat-style prompt
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reference_answer}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return text
        
        # LoRA config
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                # Removed gate_proj to save memory
            ],
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM"
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            num_train_epochs=1,  # 1 epoch for faster tuning
            logging_steps=20,
            eval_strategy="no",  # We'll evaluate manually
            save_steps=10000,  # Don't save during tuning
            save_total_limit=0,
            fp16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            dataloader_pin_memory=False,
            load_best_model_at_end=False,
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        
        print(f"Starting training...")
        # Train 
        trainer.train()
        print("✓ Training complete")
        
        # Evaluating with semantic + BertScore
        print(f"\nEvaluating model quality...") 
            
        combined_score, metrics = evaluate_model(
            trainer.model,
            tokenizer,
            val_subset,  # Use val_subset, not full dataset
            num_samples=EVAL_SAMPLES
        )
        
        print(f"\n{'='*70}")
        print(f"Trial {trial.number} Results:")
        print(f"{'='*70}")
        print(f"  Semantic Similarity: {metrics['semantic_similarity']:.4f}")
        print(f"  BERTScore F1:        {metrics['bertscore_f1']:.4f}")
        print(f"  Combined Score:      {combined_score:.4f}")
        print(f"{'='*70}\n")
        
        # Store metrics in trial user attributes for later analysis
        trial.set_user_attr("semantic_similarity", metrics['semantic_similarity'])
        trial.set_user_attr("bertscore_f1", metrics['bertscore_f1'])
        
        # Memory cleanup
        del trainer
        del model
        del tokenizer
        cleanup_memory()
        
        # Remove checkpoint directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        return combined_score
    
    except Exception as e:
        print(f"\n❌ Trial {trial.number} failed with error: {e}")
        
        try:
            del trainer
        except:
            pass
        try:
            del model
        except:
            pass
        try:
            del tokenizer
        except:
            pass
        
        cleanup_memory()
        
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except:
                pass
        
        return 0.0  # Return low score on failure


def main():
    parser = argparse.ArgumentParser()
    # Allow user to specify which config file to optimize (default is base config)
    parser.add_argument('--config', type=str, default='configs/qwen_petnet_base.yaml',
                        help='Base config file to optimize')
    args = parser.parse_args()
    
    storage_url = "sqlite:///qwen_petnet_semantic_bert_tuning.db"

    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION WITH SEMANTIC + BERTSCORE")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Evaluation samples per trial: {EVAL_SAMPLES}")
    print(f"Using subset: {USE_SUBSET}")
    print(f"Metric: 50% Semantic Similarity + 50% BERTScore F1")
    print("="*70 + "\n")

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="qwen-petnet-semantic-bert-tuning",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,  # Reduced from 3
            n_warmup_steps=20,   # Reduced from 30
        ),
        storage=storage_url,
        load_if_exists=True
    )

    # Run optimization
    print("Starting hyperparameter optimization...")
    print("This will take several hours depending on n_trials\n")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=None,
        n_jobs=1,
        show_progress_bar=True,
        catch=(Exception,),  # Catch exceptions and continue
    )

    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

    # Check if we have any successful trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        print("No trials completed successfully!")
        return

    print(f"Completed trials: {len(completed_trials)}/{len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best combined score: {study.best_value:.4f}")
    print(f"  - Semantic Similarity: {study.best_trial.user_attrs['semantic_similarity']:.4f}")
    print(f"  - BERTScore F1: {study.best_trial.user_attrs['bertscore_f1']:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results to YAML if config exists
    if os.path.exists(args.config):
        try:
            best_config = load_config(args.config)
            best_config['train']['learning_rate'] = study.best_params['learning_rate']
            best_config['train']['lora_r'] = study.best_params['lora_r']
            best_config['train']['lora_alpha'] = study.best_params['lora_alpha']
            best_config['train']['lora_dropout'] = study.best_params['lora_dropout']
            best_config['train']['gradient_accumulation_steps'] = study.best_params['gradient_accumulation_steps']
            best_config['train']['warmup_ratio'] = study.best_params['warmup_ratio']
            best_config['train']['max_length'] = study.best_params['max_length']

            # Save
            save_path = f"configs/{Path(args.config).stem}_optimized.yaml"
            os.makedirs("configs", exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(best_config, f, default_flow_style=False)
            print(f"\n💾 Optimized config saved to: {save_path}")
        except Exception as e:
            print(f"\nWarning: Could not save config: {e}")

    # Save JSON results
    import json
    with open("optuna_semantic_bert_results.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_combined_score": study.best_value,
            "best_semantic_similarity": study.best_trial.user_attrs['semantic_similarity'],
            "best_bertscore_f1": study.best_trial.user_attrs['bertscore_f1'],
            "best_params": study.best_params,
            "completed_trials": len(completed_trials),
            "total_trials": len(study.trials),
            "all_trials": [
                {
                    "number": trial.number,
                    "combined_score": trial.value if trial.value is not None else "N/A",
                    "semantic_similarity": trial.user_attrs.get('semantic_similarity', 0),
                    "bertscore_f1": trial.user_attrs.get('bertscore_f1', 0),
                    "params": trial.params,
                    "state": str(trial.state)
                }
                for trial in study.trials
            ]
        }, f, indent=2)

    print("✓ Results saved to optuna_semantic_bert_results.json")

    # Visualizations
    try:
        import plotly
        
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html("optuna_history.html")
        
        if len(completed_trials) >= 3:
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html("optuna_importances.html")
        
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html("optuna_parallel.html")
        
        print("✓ Visualizations saved to optuna_*.html")
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")

    # Train final model with best parameters
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*70)

    print("\nCleaning GPU memory before final training...")
    
    # Delete semantic model
    global semantic_model
    del semantic_model

    # Multiple cleanup passes
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    time.sleep(5)
    print("✓ Memory cleaned\n")

    best_params = study.best_params

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files={
        "train": TRAIN_DATA,
        "validation": VAL_DATA
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
            "gate_proj"  # Add back for final training
        ],
        lora_dropout=best_params["lora_dropout"],
        task_type="CAUSAL_LM",
        dtype=torch.float16
    )

    training_args = TrainingArguments(
        output_dir="qwen-petnet-final-semantic-bert",
        per_device_train_batch_size=1,  # Keep at 1 for safety
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        learning_rate=best_params["learning_rate"],
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=150,  # Stagger with save_steps
        save_steps=100,
        save_total_limit=1,  # Only keep 1 checkpoint
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

    print("Starting final training...")
    trainer.train()
    
    print("\nSaving final model...")
    trainer.save_model("qwen-petnet-final-semantic-bert")

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: qwen-petnet-final-semantic-bert/")
    print(f"Results: optuna_semantic_bert_results.json")
    print("="*70)

    
if __name__ == "__main__":
    main()
