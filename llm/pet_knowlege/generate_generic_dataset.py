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
output_train = os.path.join(script_dir, "dataset_generic_train.jsonl")

api_key = "your_api_key_here"

# client = openai.OpenAI(
#     base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key = api_key
# )

client = genai.Client(api_key=api_key)

specieses = [
    'Dog', 'Cat'
]

# Base instruction templates from generate_dataset.py
INSTRUCTION_TEMPLATES = {
    "care": ["Give care instructions for a {species}.", 
             "How should a {species} be cared for?", 
             "What are the care requirements of a {species}?"],
    "grooming": ["What grooming does a {species} need?", 
                 "How do you groom a {species}?", 
                 "Grooming instructions for a {species}."],
    "exercise": ["How much exercise does a {species} need?", 
                 "Exercise requirements for a {species}.", 
                 "What activities should a {species} do?"],
    "health": ["What health problems commonly affect the {species}?", 
               "Health concerns for a {species}.", 
               "List common diseases of a {species}."],
    "temperament": ["Describe the temperament of a {species}.", 
                    "Personality traits of a {species}."],
    "family": ["Is the {species} a good family pet?", 
               "How suitable is a {species} for families?"],
    "overview": ["Give a general overview of the {species}.", 
                 "Provide a summary of the {species}."],
    "training": ["Provide training tips for a {species}.", 
                 "How should you train a {species}?", 
                 "Training recommendations for a {species}."],
    "diet": ["What do {species} eat?",
             "What kind of food is best for a {species}?",
             "{species} dietary recommendation."]
}

def generate_prompt_examples():
    prompt_templates = []
    for aspect, sample_templates in INSTRUCTION_TEMPLATES.items():
        prompt_templates.append(
            "Generate a JSON entry for {species} about "
            + aspect +
            ": {{{{'instruction': 'User question', 'input': '', 'output': 'Expert answer'}}}}."
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
            "Your output must be species-specific, concise (≤200 words), factual, "
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
    for species in specieses:
        print("Generating data for breed:", species)
        for prompt_template in generate_prompt_examples():
            prompt = prompt_template.format(species=species)
            raw_output = call_gemini(prompt)
            cleaned = re.sub(r"```json|```", "", raw_output).strip()
            try:
                # Remove control characters that may cause JSON decoding issues
                mapping = dict.fromkeys(range(32))
                cleaned = cleaned.translate(mapping)

                dataset.append(json.loads(cleaned))
            except json.JSONDecodeError as e:
                print(f"JSON decoding error for species {species} with prompt '{prompt}': {e}. Response: {cleaned}")
                continue

    with open(output_train, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if api_key == "your_api_key_here":
        raise ValueError("Please set your API key")
    main()