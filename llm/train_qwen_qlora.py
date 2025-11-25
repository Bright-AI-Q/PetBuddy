from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# More aggressive quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Changed from bfloat16
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

# Formatting function with length limit
def formatting_func(example):
    inputs = example["input"]
    outputs = example["output"]
    text = f"{inputs}\n{outputs}" if inputs else outputs
    # Truncate if needed
    tokens = tokenizer(text, truncation=True, max_length=1024) # to prevent memory overflow
    return tokenizer.decode(tokens['input_ids'])

# Smaller LoRA config
peft_config = LoraConfig(
    r=32,  #Reduced from 64
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"
        # Removed MLP layers to save memory
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="qwen-petnet",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # Increased from 16
    gradient_checkpointing=True,  # Important for memory
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=5,
    save_steps=100,
    save_total_limit=1,  # Reduced from 2
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,  # Groups similar lengths together
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
trainer.save_model("qwen-petnet")
print("🎉 Training complete!")