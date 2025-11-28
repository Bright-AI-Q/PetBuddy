"""
Purpose:
This generates the dataset_cat_train.jsonl and dataset_cat_val.jsonl files
for cat breeds based on the Oxford-IIIT Pets dataset using the Deepseek LLM API.

**Caution**:
The json output may sometimes be malformed.
Please manually verify and correct the generated dataset as needed.

Example of a malformed entry:
{"introduction": {"instruction": "Introduce this cat breed", "input": "Detected breed: Birman", "output": "The Birman is a gentle, affectionate cat breed known for its striking blue eyes, silky coat, and distinctive white 'gloves' on all four paws. Personality-wise, Birmans are sociable, calm, and devoted companions that thrive on human interaction, making them ideal for families. They typically weigh 6–12 pounds, with a medium to large, sturdy build. Common health issues include hypertrophic cardiomyopathy (a heart condition), kidney disease such as polycystic kidney disease, and dental problems. Regular veterinary check-ups, a balanced diet, and proper grooming can help maintain their health. Birmans are low-shedding and adapt well to indoor living, offering years of loyal friendship with their serene and friendly nature."}}

Should be corrected to:
{"instruction": "Introduce this cat breed", "input": "Detected breed: Birman", "output": "The Birman is a gentle, affectionate cat breed known for its striking blue eyes, silky coat, and distinctive white 'gloves' on all four paws. Personality-wise, Birmans are sociable, calm, and devoted companions that thrive on human interaction, making them ideal for families. They typically weigh 6–12 pounds, with a medium to large, sturdy build. Common health issues include hypertrophic cardiomyopathy (a heart condition), kidney disease such as polycystic kidney disease, and dental problems. Regular veterinary check-ups, a balanced diet, and proper grooming can help maintain their health. Birmans are low-shedding and adapt well to indoor living, offering years of loyal friendship with their serene and friendly nature."}
"""

import json
import openai
from sklearn.model_selection import train_test_split
import re

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_train = os.path.join(script_dir, "dataset_cat_train.jsonl")
output_val = os.path.join(script_dir, "dataset_cat_val.jsonl")

api_key = "your_api_key_here"

client = openai.OpenAI(
    base_url = "https://api.deepseek.com/v1",
    api_key = api_key
)

# Oxford-IIIT Pets dataset: 12 cat breeds
cat_breeds = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
]

prompt_templates = [
    "Generate a JSON entry for {breed} about introduction: {{'instruction': 'User question', 'input': 'Detected breed: {breed}', 'output': 'Expert answer'}}. User question must include 'Introduce this cat breed' or 'What is this cat?'. Expert answer (≤200 words): Covers personality, size, and common health issues—friendly and professional.",
    "Generate a JSON entry for {breed} about feeding: {{'instruction': 'User question', 'input': 'Detected breed: {breed}', 'output': 'Expert answer'}}. User question example: 'How often to feed a {breed} kitten?' or 'What food is best for {breed} with sensitive stomachs?'. Expert answer: Actionable advice tailored to the breed's dietary needs (e.g., portion sizes, food types to avoid).",
    "Generate a JSON entry for {breed} about health: {{'instruction': 'User question', 'input': 'Detected breed: {breed}', 'output': 'Expert answer'}}. User question example: 'Why is my {breed} over-grooming?' or 'How to calm a hyper {breed}?'. Expert answer: Safe, breed-specific advice + a note to consult a vet for severe cases.",
]

def main():
    dataset = []
    for breed in cat_breeds:
        print("Generating data for breed:", breed)
        for prompt_template in prompt_templates:
            prompt = prompt_template.format(breed=breed)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides expert pet care advice."},
                    {"role": "user", "content": prompt}
                ]
            )
            json_response = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
            dataset.append(json.loads(json_response))

    # Split into training (80%) and validation (20%)
    train_examples, val_examples = train_test_split(dataset, test_size=0.2, random_state=42)

    with open(output_train, "w", encoding="utf-8") as f:
        for item in train_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_val, "w", encoding="utf-8") as f:
        for item in val_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if api_key == "your_api_key_here":
        raise ValueError("Please set your API key")
    main()