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
import rag_engine

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

# Method to generate response with RAG
def generate_rag_response(model, tokenizer, user_query, rag_engine, max_new_tokens=256):
    """
    Integrates the RAG retrieval with your generation step.
    Optimized for Qwen models.
    """
    import torch
    
    # 1. Retrieve Context
    breed_name, context_text = rag_engine.retrieve_context(user_query)
    
    if not context_text:
        # Fallback for generic chit-chat or unknown breeds
        messages = [
            {"role": "system", "content": "You are a helpful dog expert assistant."},
            {"role": "user", "content": user_query}
        ]
    else:
        # 2. Construct RAG Prompt with retrieved context
        system_prompt = f"""You are a specialized veterinary assistant with access to a breed database.

You have been provided with documentation about the {breed_name}.

INSTRUCTIONS:
- Answer the user's question using ONLY information from the Reference Document below
- If the answer is not in the document, respond: "I don't have that information in my database."
- Do not make up facts or add information not present in the document
- Use exact statistics (weight, height, lifespan) as provided

### REFERENCE DOCUMENT:
{context_text}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    
    # 3. Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 4. Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    
    # 5. Generate Answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for factual accuracy
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 6. Extract only the newly generated tokens
    generated_ids = output_ids[0]
    prompt_len = inputs.input_ids.shape[-1]
    new_token_ids = generated_ids[prompt_len:]
    
    # 7. Decode the assistant's response
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    
    # Debug output
    print(f"Query: {user_query}")
    print(f"Breed: {breed_name if breed_name else 'Not detected'}")
    print(f"Response: {response}")
    print("=" * 50)
    
    return response

# Copied from evaluate_qwen_petnet.py, method to generate response without RAG
def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    # 1. Build chat-style prompt
    messages = [{"role": "user", "content": prompt}]
    # Since we want the model to generate the assistant's response, we add generation prompt
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
    print(prompt)
    print(response)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    return response

api_key = "your-api-key-here"

client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="qwen-petnet-final/final",
                        help='Finetuned model to use. If not provided, use the base model.')
    parser.add_argument('--output', type=str, default="output_final.jsonl")
    parser.add_argument('--use_rag', type=bool, default=False, help='Whether to use RAG')
    args = parser.parse_args()

    qa_data = load_questions_and_reference_answers()

    if args.model_name.lower() in ["none", "null", ""]:
        base_model, base_tokenizer = load_base_model()
    else:
        base_model, base_tokenizer = load_finedtuned_model(args.model_name)

    if args.use_rag:
        rag = rag_engine.DogBreedRAG("./pet_knowlege/pet_database/")

    for item in qa_data:
        if args.use_rag:
            item["model_response"] = generate_rag_response(base_model, base_tokenizer, item["question"], rag)
        else:
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
