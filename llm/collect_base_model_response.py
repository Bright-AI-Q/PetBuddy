"""
This module uses LLMs to judge the quality of generated data samples.
"""

import json
import os
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

output_base_model_responses = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_base.jsonl"
)

def load_questions_and_reference_answers():
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

# def generate_evaluation_prompt(breed, question, reference_answer, tag, candidate_answer):
#     return [
#         {
#             "role": "system",
#             "content": (
#                 "You are a professional veterinarian and pet care expert.",
#                 "Evaluate the AI assistant's response against 3 criteria (1-5 points each: 5=excellent, 1=poor)",
#                 "1. Breed Consistency: Does the response align with the given pet breed?",
#                 "2. Medical Safety: Is the advice safe (no hallucinations like toxic food recommendations)?",
#                 "3. Usefulness: Does it solve the user's problem with actionable steps?",
#             ),
#         }, 
#         {
#             "role": "user",
#             "content": (
#                 f"Background: Detected pet = {breed} {tag}\n"
#                 f"User Question: {question}\n"
#                 f"Reference Answer: {reference_answer}\n"
#                 f"AI Assistant Response: {candidate_answer}\n\n"
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 "Please provide a total score (out of 15) and a 1-sentence reason in the following JSON format.",
#                 "{\n"
#                 '  "total_score": <score 1-15>,\n'
#                 '  "reason": An explanation less than 200 words,\n'
#                 "}\n"
#                 "Only output the JSON object without any additional text."
#             ),
#         }
#     ]

# api_key = "your_api_key_here"

# client = openai.OpenAI(
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key
# )

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    # Format the prompt properly for chat models
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
    response = response[len(text):].strip()
    return response

if __name__ == "__main__":
    base_model, base_tokenizer = load_base_model()
    qa_data = load_questions_and_reference_answers()
    for data in qa_data:
        response = generate_response(base_model, base_tokenizer, qa_data["question"])
        print("Base model response:", response)
    qa_data["model_response"] = response

    with open(output_base_model_responses, "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")