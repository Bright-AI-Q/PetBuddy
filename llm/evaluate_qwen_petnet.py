import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_finetuned_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "qwen-petnet")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    # 1. Build chat-style prompt
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 2. Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

    # 3. Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 4. Strip the prompt part: keep only newly generated tokens
    generated_ids = output_ids[0]
    prompt_len = inputs.input_ids.shape[-1]
    new_token_ids = generated_ids[prompt_len:]

    # 5. Decode only the assistant's answer
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return response

def calculate_perplexity(model, tokenizer, text):
    """Lower perplexity = better"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

# Load test dataset
print("Loading dataset...")
test_data = load_dataset("json", data_files={"test": "pet_knowlege/dataset_val.jsonl"})["test"]

print("Loading models...")
base_model, base_tokenizer = load_base_model()
print("Base model loaded!")

ft_model, ft_tokenizer = load_finetuned_model()
print("Fine-tuned model loaded!")

results = []
base_perplexities = []
ft_perplexities = []

# Limit to 50 examples or less
num_examples = min(50, len(test_data))
print(f"\nEvaluating on {num_examples} test examples...")

for i, example in enumerate(tqdm(test_data.select(range(num_examples)))):
    # Use "instruction" field instead of "input"
    prompt = example.get("instruction", "")
    expected_output = example.get("output", "")
    
    # Skip if prompt or output is empty
    if not prompt.strip() or not expected_output.strip():
        print(f"Skipping example {i}: empty instruction or output")
        continue
    
    try:
        # Generate responses
        base_response = generate_response(base_model, base_tokenizer, prompt)
        ft_response = generate_response(ft_model, ft_tokenizer, prompt)
        
        # Calculate perplexity on the expected output
        # Combine instruction and output for perplexity calculation
        full_text = f"{prompt}\n{expected_output}"
        base_ppl = calculate_perplexity(base_model, base_tokenizer, full_text)
        ft_ppl = calculate_perplexity(ft_model, ft_tokenizer, full_text)
        
        base_perplexities.append(base_ppl)
        ft_perplexities.append(ft_ppl)
        
        results.append({
            "prompt": prompt,
            "expected": expected_output,
            "base_response": base_response,
            "finetuned_response": ft_response,
            "base_perplexity": base_ppl,
            "finetuned_perplexity": ft_ppl
        })
    except Exception as e:
        print(f"Error on example {i}: {e}")
        continue

if not base_perplexities:
    print("No valid examples to evaluate!")
    exit()

# Calculate average perplexity
avg_base_ppl = sum(base_perplexities) / len(base_perplexities)
avg_ft_ppl = sum(ft_perplexities) / len(ft_perplexities)

print(f"\n{'='*60}")
print(f"Evaluated {len(results)} examples")
print(f"Average Base Model Perplexity: {avg_base_ppl:.2f}")
print(f"Average Fine-tuned Model Perplexity: {avg_ft_ppl:.2f}")
improvement = ((avg_base_ppl - avg_ft_ppl) / avg_base_ppl * 100)
print(f"Improvement: {improvement:.1f}%")
print(f"{'='*60}\n")

# Save detailed results
with open("evaluation_results.json", "w") as f:
    json.dump({
        "summary": {
            "avg_base_perplexity": avg_base_ppl,
            "avg_finetuned_perplexity": avg_ft_ppl,
            "improvement_percent": improvement,
            "num_examples": len(results)
        },
        "examples": results
    }, f, indent=2)

print("Detailed results saved to evaluation_results.json")

# Print a few example comparisons
print("\nSample Comparisons:")
print("="*60)
for i, result in enumerate(results[:3]):
    print(f"\n{'='*60}")
    print(f"Example {i+1}:")
    print(f"\nPrompt: {result['prompt']}")
    print(f"\nExpected Output (first 300 chars):\n{result['expected'][:300]}...")
    print(f"\n--- Base Model Response (first 300 chars) ---")
    print(result['base_response'][:300])
    print(f"\n--- Fine-tuned Model Response (first 300 chars) ---")
    print(result['finetuned_response'][:300])
    print(f"\nPerplexity - Base: {result['base_perplexity']:.2f}, Fine-tuned: {result['finetuned_perplexity']:.2f}")
    print("="*60)