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
from google import genai
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_train = os.path.join(script_dir, "dataset_cat_train.jsonl")
output_val = os.path.join(script_dir, "dataset_cat_val.jsonl")

api_key = "your_api_key_here"

# client = openai.OpenAI(
#     base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key = api_key
# )

client = genai.Client(api_key=api_key)


# Oxford-IIIT Pets dataset: 12 cat breeds
cat_breeds = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
]

# Base instruction templates from generate_dataset.py
INSTRUCTION_TEMPLATES = {
    "care": ["Give care instructions for a {breed}. If you don't have breed-specific info, provide general cat care instructions.", 
             "How should a {breed} be cared for? If you don't have breed-specific info, provide general cat care instructions.", 
             "What are the care requirements of a {breed}? If you don't have breed-specific info, provide general cat care instructions."],
    "grooming": ["What grooming does a {breed} need? If you don't have breed-specific info, provide general cat grooming instructions.", 
                 "How do you groom a {breed}? If you don't have breed-specific info, provide general cat grooming instructions.", 
                 "Grooming instructions for a {breed}. If you don't have breed-specific info, provide general cat grooming instructions."],
    "exercise": ["How much exercise does a {breed} need? If you don't have breed-specific info, provide general cat exercise instructions.", 
                 "Exercise requirements for a {breed}. If you don't have breed-specific info, provide general cat grooming instructions.", 
                 "What activities should a {breed} do? If you don't have breed-specific info, provide general cat grooming instructions."],
    "health": ["What health problems commonly affect the {breed}? If you don't have breed-specific info, provide general cat health problems.", 
               "Health concerns for a {breed}. If you don't have breed-specific info, provide general cat health concerns.", 
               "List common diseases of a {breed}. If you don't have breed-specific info, provide general cat diseases."],
    "temperament": ["Describe the temperament of a {breed}. If you don't have breed-specific info, provide general cat temperment information.", 
                    "Personality traits of a {breed}. If you don't have breed-specific info, provide general cat personality information."],
    "family": ["Is the {breed} a good family cat? If you don't have breed-specific info, provide general cat family compatibility information.", 
               "How suitable is a {breed} for families? If you don't have breed-specific info, provide general cat family compatibility information."],
    "overview": ["Give a general overview of the {breed}.", 
                 "Provide a summary of the {breed}."],
    "training": ["Provide training tips for a {breed}. If you don't have breed-specific info, provide general cat training instruction.", 
                 "How should you train a {breed}? If you don't have breed-specific info, provide general cat training instruction.", 
                 "Training recommendations for a {breed}. If you don't have breed-specific info, provide general cat training instruction."],
    "diet": ["What do {breed} eat? If you don't have breed-specific info, provide general cat feeding instruction." ,
             "What kind of food is best for a {breed}? If you don't have breed-specific info, provide general cat dietary instruction.",
             "{breed} dietary recommendation. If you don't have breed-specific info, provide general cat dietary information."]
}

def generate_prompt_examples():
    prompt_templates = []
    for aspect, sample_templates in INSTRUCTION_TEMPLATES.items():
        prompt_templates.append(
            "Generate a JSON entry for {breed} cats about "
            + aspect +
            ": {{{{'instruction': 'User question', 'input': 'Detected breed: {breed}', 'output': 'Expert answer'}}}}."
            " User question examples: " +
            ", ".join([f"'{t}'" for t in sample_templates])
        )
    return prompt_templates


def call_gemini(prompt):
    """Gemini wrapper — returns the text output only."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "You are a helpful assistant that provides expert pet care advice. "
            "Your output must be breed-specific, concise (≤200 words), factual, "
            "and written in a friendly professional tone.",
            prompt,
            "You must output a single JSON object with exactly these 3 fields: "
            "'instruction', 'input', and 'output'.",
            "Never use placeholder phrases like 'User question'. The 'instruction' field must contain the real user question.",
            "The JSON must be strictly valid. No trailing commas or comments."
        ]
    )

    return response.text

def main():
    dataset = []
    for breed in cat_breeds:
        print("Generating data for breed:", breed)
        for prompt_template in generate_prompt_examples():
            prompt = prompt_template.format(breed=breed)
            raw_output = call_gemini(prompt)
            cleaned = re.sub(r"```json|```", "", raw_output).strip()
            try:
                # Remove control characters that may cause JSON decoding issues
                mapping = dict.fromkeys(range(32))
                cleaned = cleaned.translate(mapping)

                dataset.append(json.loads(cleaned))
            except json.JSONDecodeError as e:
                print(f"JSON decoding error for breed {breed} with prompt '{prompt}': {e}. Response: {cleaned}")
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