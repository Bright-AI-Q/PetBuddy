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
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key = api_key
)

# Oxford-IIIT Pets dataset: 12 cat breeds
cat_breeds = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
]

# Base instruction templates from generate_dataset.py
INSTRUCTION_TEMPLATES = {
    "care": ["Give care instructions for a {breed}.", 
             "How should a {breed} be cared for?", 
             "What are the care requirements of a {breed}?"],
    "grooming": ["What grooming does a {breed} need?", 
                 "How do you groom a {breed}?", 
                 "Grooming instructions for a {breed}."],
    "exercise": ["How much exercise does a {breed} need?", 
                 "Exercise requirements for a {breed}.", 
                 "What activities should a {breed} do?"],
    "health": ["What health problems commonly affect the {breed}?", 
               "Health concerns for a {breed}.", 
               "List common diseases of a {breed}."],
    "temperament": ["Describe the temperament of a {breed}.", 
                    "Personality traits of a {breed}."],
    "family": ["Is the {breed} a good family cat?", 
               "How suitable is a {breed} for families?"],
    "overview": ["Give a general overview of the {breed}.", 
                 "Provide a summary of the {breed}."],
    "training": ["Provide training tips for a {breed}.", 
                 "How should you train a {breed}?", 
                 "Training recommendations for a {breed}."]
}

def generate_prompt_examples():
    prompt_templates = []
    for aspect, sample_templates in INSTRUCTION_TEMPLATES.items():
        prompt_templates.append("Generate a JSON entry for {breed} cats about " + aspect + ": {{'instruction': 'User question', 'input': 'Detected breed: {breed}', 'output': 'Expert answer'}}. User question examples: " + ", ".join([f"'{t}'" for t in sample_templates]))
    return prompt_templates

def main():
    dataset = []
    for breed in cat_breeds:
        print("Generating data for breed:", breed)
        for prompt_template in generate_prompt_examples():
            prompt = prompt_template.format(breed=breed)
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that provides expert pet care advice. "
                            "Your output must be breed-specific, concise (≤200 words), factual, "
                            "and written in a friendly professional tone."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            "You must output a **single JSON object** with exactly these 3 fields: "
                            "'instruction', 'input', and 'output'. "
                            "Do NOT include additional fields."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Never use placeholder phrases like 'User question'. "
                            "The 'instruction' field must contain the actual user question."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "The JSON must be strictly valid (no trailing commas, no control characters, no comments)."
                        )
                    }
                ]
            )
            json_response = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
            try:
                # Remove control characters that may cause JSON decoding issues
                mapping = dict.fromkeys(range(32))
                json_response = json_response.translate(mapping)

                dataset.append(json.loads(json_response))
            except json.JSONDecodeError as e:
                print(f"JSON decoding error for breed {breed} with prompt '{prompt}': {e}. Response: {json_response}")
                continue

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