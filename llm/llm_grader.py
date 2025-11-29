"""
This module runs the model based on questions from pet_knowledge/test_samples.jsonl
and then uses LLM to judge the quality of generated data samples.
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

def load_questions_and_reference_answers() -> List[Dict[str, Any]]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_samples_path = os.path.join(script_dir, "pet_knowlege", "test_samples.jsonl")

    questions_and_answers = []

    with open(test_samples_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            questions_and_answers.append(
                {
                    "breed": data.get("breed"),
                    "question": data.get("question"),
                    "reference_answer": data.get("reference_answer"),
                    "tag": data.get("tag"),
                }
            )

    return questions_and_answers


def generate_evaluation_prompt(
    breed: str, question: str, reference_answer: str, tag: str, candidate_answer: str
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a professional veterinarian and pet care expert."
                "Evaluate the AI assistant's response against 3 criteria (1-5 points each: 5=excellent, 1=poor)."
                "1. Breed Consistency: Does the response align with the given pet breed?"
                "2. Medical Safety: Is the advice safe (no hallucinations like toxic food recommendations)?"
                "3. Usefulness: Does it solve the user's problem with actionable steps?"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Background: Detected pet = {breed} {tag}\n"
                f"User Question: {question}\n"
                f"Reference Answer: {reference_answer}\n"
                f"AI Assistant Response: {candidate_answer}\n\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Please provide a total score (out of 15) and a 1-sentence reason in the following JSON format."
                "{\n"
                '  "total_score": <score 1-15>,\n'
                '  "reason": An explanation less than 200 words,\n'
                "}\n"
                "Only output the JSON object without any additional text."
            ),
        },
    ]


# Copied from evaluate_qwen_petnet.py
# Unable to import because of unable to load_finetuned_model
def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_finedtuned_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Copied from evaluate_qwen_petnet.py
# Unable to import because of unable to load_finetuned_model
def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    # Format the prompt properly for chat models
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

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

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response[len(text) :].strip()
    return response

api_key = "your_api_key_here"

client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="qwen-petnet-final/final",
                        help='Finetuned model to use. If not provided, use the base model.')
    parser.add_argument('--output', type=str, default="output_final.jsonl")
    args = parser.parse_args()

    qa_data = load_questions_and_reference_answers()

    if args.model_name is None:
        base_model, base_tokenizer = load_base_model()
    else:
        base_model, base_tokenizer = load_finedtuned_model(args.model_name)

    for item in qa_data:
        item["model_response"] = generate_response(
            base_model, base_tokenizer, item["question"]
        )

        if api_key == "your_api_key_here":
            print("Please set your API key in the code. Skipping evaluation...")
            continue

        evaluation_prompt = generate_evaluation_prompt(
            breed=item["breed"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            tag=item["tag"],
            candidate_answer=item["model_response"],
        )

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=evaluation_prompt,
        )
        json_response = re.sub(
            r"```json|```", "", response.choices[0].message.content
        ).strip()
        try:
            # Remove control characters that may cause JSON decoding issues
            mapping = dict.fromkeys(range(32))
            json_response = json_response.translate(mapping)

            judge_response = json.loads(json_response)
            item["score"] = judge_response.get("total_score", "")
            item["reason"] = judge_response.get("reason", "")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}. Response: {json_response}")
            continue

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output), "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
