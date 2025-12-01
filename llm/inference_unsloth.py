"""
Inference script for Unsloth-trained Qwen model.
Loads test questions from test_samples.jsonl and generates answers.
"""
import json
import os
import argparse
from unsloth import FastLanguageModel
import torch

def load_test_samples(test_file="pet_knowlege/test_samples.jsonl"):
    """Load test questions from JSONL file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, test_file)
    
    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append(data)
    
    return samples

def load_model(model_path, max_seq_length=2048):
    """Load the fine-tuned model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_new_tokens=512):
    """Generate an answer for the given question."""
    # Format prompt using the same template as training
    prompt = f"""Below are some instructions that describe some tasks. Write
responses that appropriately complete each request.

### Instruction:
{question}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (skip the prompt)
    prompt_length = inputs.input_ids.shape[-1]
    generated_ids = outputs[0][prompt_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return answer

def main():
    parser = argparse.ArgumentParser(description='Run inference on test samples')
    parser.add_argument('--model_path', type=str, default="qwen-petnet-final/unsloth/final",
                        help='Path to the fine-tuned model')
    parser.add_argument('--test_file', type=str, default="pet_knowlege/test_samples.jsonl",
                        help='Path to test samples JSONL file')
    parser.add_argument('--output', type=str, default="inference_output.jsonl",
                        help='Output file for results')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    args = parser.parse_args()
    
    print("Loading test samples...")
    test_samples = load_test_samples(args.test_file)
    print(f"Loaded {len(test_samples)} test samples")
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded successfully")
    
    results = []
    
    for i, sample in enumerate(test_samples, 1):
        question = sample.get("question", "")
        breed = sample.get("breed", "")
        tag = sample.get("tag", "")
        reference = sample.get("reference_answer", "")
        
        print(f"\n[{i}/{len(test_samples)}] Processing: {breed} - {question[:50]}...")
        
        # Generate answer
        answer = generate_answer(model, tokenizer, question, args.max_new_tokens)
        
        # Store result
        result = {
            "breed": breed,
            "tag": tag,
            "question": question,
            "reference_answer": reference,
            "model_response": answer
        }
        results.append(result)
        
        print(f"Generated: {answer[:100]}...")
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"Total samples processed: {len(results)}")

if __name__ == "__main__":
    main()
