import os
import json
import jsonlines
import random
from sklearn.model_selection import train_test_split

# Folder with JSON files
input_dir = "dog_database"
output_train = "dataset_train.jsonl"
output_val = "dataset_val.jsonl"

# Keywords for extracting sections
SECTION_KEYWORDS = {
    "care": ["care", "training", "exercise", "groom", "nutrition", "living", "health"],
    "grooming": ["groom", "coat", "brushing", "trimming"],
    "exercise": ["exercise", "play", "activity"],
    "health": ["health", "disease", "condition"],
    "temperament": ["temperament", "personality", "traits"],
    "family": ["family", "children", "ideal", "not ideal"],
    "overview": ["overview", "introduction", "summary", "final"],
    "training": ["training", "socialization", "commands", "tricks", "behaviors"]
}

# Base instruction templates
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
    "family": ["Is the {breed} a good family dog?", 
               "How suitable is a {breed} for families?"],
    "overview": ["Give a general overview of the {breed}.", 
                 "Provide a summary of the {breed}."],
    "training": ["Provide training tips for a {breed}.", 
                 "How should you train a {breed}?", 
                 "Training recommendations for a {breed}."]
}

# Extract sections matching keywords
def extract_sections(sections, keywords):
    keywords = [k.lower() for k in keywords]
    extracted = []
    for sec in sections:
        title = sec.get("section", "").lower()
        content = sec.get("content", "").strip()
        if any(k in title or k in content.lower() for k in keywords):
            extracted.append(content)
    return "\n".join(extracted)

# Generate examples for one breed, with multiple instruction variations
def make_examples(breed_data):
    breed = breed_data["breed"]
    sections = breed_data["sections"]
    examples = []

    for key, keywords in SECTION_KEYWORDS.items():
        text = extract_sections(sections, keywords)
        if not text.strip():
            text = "No information available."
            print(f"[WARN] No '{key}' data for {breed}. Using placeholder.")

        # For each instruction variation, create an example
        for instr_template in INSTRUCTION_TEMPLATES[key]:
            examples.append({
                "instruction": instr_template.format(breed=breed),
                "input": "",
                "output": text
            })
    return examples

# Main script
def main():
    all_examples = []

    if not os.path.exists(input_dir):
        print(f"Input folder '{input_dir}' not found!")
        return

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                breed_data = json.load(f)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        if "breed" not in breed_data or "sections" not in breed_data:
            print(f"[SKIP] Missing required fields in {fname}")
            continue

        examples = make_examples(breed_data)
        all_examples.extend(examples)
        print(f"Processed {fname}: {len(examples)} examples")

    # Shuffle all examples
    random.shuffle(all_examples)

    # Split into training (80%) and validation (20%)
    train_examples, val_examples = train_test_split(all_examples, test_size=0.2, random_state=42)

    # Write to jsonl files
    with jsonlines.open(output_train, "w") as writer:
        for ex in train_examples:
            writer.write(ex)

    with jsonlines.open(output_val, "w") as writer:
        for ex in val_examples:
            writer.write(ex)

    print(f"\nDONE — wrote {len(train_examples)} training examples to {output_train}")
    print(f"DONE — wrote {len(val_examples)} validation examples to {output_val}")

if __name__ == "__main__":
    main()
