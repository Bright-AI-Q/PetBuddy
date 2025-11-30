from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_length=2048,
    dtype=None,  # None for auto detection.
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=42,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# Source: https://docs.unsloth.ai/basics/chat-templates
chat_template = """Below are some instructions that describe some tasks. Write
responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""


def format_prompt(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for _input, _output in zip(inputs, outputs):
        text = chat_template.format(_input, _output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}


from datasets import load_dataset

train_dataset = load_dataset(
    "json", data_files="pet_knowlege/dataset_train.jsonl", split="train"
).map(format_prompt, batched=True)
val_dataset = load_dataset(
    "json", data_files="pet_knowlege/dataset_val.jsonl", split="train"
).map(format_prompt, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = True, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = "unsloth",
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = "qwen-petnet-final/unsloth",
    ),
)

trainer.train()