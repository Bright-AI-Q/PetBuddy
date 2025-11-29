"""
This module uses LLMs to judge the quality of generated data samples.
"""

import json
import os
import re
from typing import List, Dict, Any
from evaluate_qwen_petnet import load_base_model, generate_response
import openai

output_base_model_responses = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output_base.jsonl"
)


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
                "You are a professional veterinarian and pet care expert.",
                "Evaluate the AI assistant's response against 3 criteria (1-5 points each: 5=excellent, 1=poor)",
                "1. Breed Consistency: Does the response align with the given pet breed?",
                "2. Medical Safety: Is the advice safe (no hallucinations like toxic food recommendations)?",
                "3. Usefulness: Does it solve the user's problem with actionable steps?",
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
                "Please provide a total score (out of 15) and a 1-sentence reason in the following JSON format.",
                "{\n"
                '  "total_score": <score 1-15>,\n'
                '  "reason": An explanation less than 200 words,\n'
                "}\n"
                "Only output the JSON object without any additional text.",
            ),
        },
    ]


api_key = "your_api_key_here"

client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key
)

if __name__ == "__main__":
    qa_data = load_questions_and_reference_answers()
    base_model, base_tokenizer = load_base_model()

    for item in qa_data:
        item["model_response"] = generate_response(
            base_model, base_tokenizer, item["question"]
        )

        evaluation_prompt = generate_evaluation_prompt(
            breed=item["breed"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            tag=item["tag"],
            candidate_answer=item["model_response"],
        )

        if api_key == "your_api_key_here":
            print("Please set your API key in the code. Skipping evaluation...")
            continue

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

    with open(output_base_model_responses, "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
