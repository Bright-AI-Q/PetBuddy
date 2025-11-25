from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

# -------------------------
# 1️⃣ Model & Tokenizer
# -------------------------
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",  # safer for 8GB GPU
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # automatically place layers on GPU/CPU
    offload_folder="offload",  # moves unused layers to CPU to save VRAM
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# 2️⃣ Dataset
# -------------------------
dataset = load_dataset("json", data_files={"train": "pet_knowledge/dataset_train.jsonl", "validation": "pet_knowledge/dataset_val.jsonl"})

# -------------------------
# 3️⃣ LoRA Config
# -------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# -------------------------
# 4️⃣ Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="messages",  # adjust if your field is "output"
    max_seq_length=1024,            # smaller seq length saves memory
    peft_config=peft_config,
    output_dir="qwen-petnet",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # simulate larger batch
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=200,
)

# -------------------------
# 5️⃣ Train & Save
# -------------------------
trainer.train()
trainer.save_model("qwen-petnet")
print("🎉 Training complete!")

