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
    "grooming": ["groom", "brushing", "trimming", "grooming", "coat"],
    "exercise": ["exercise", "play", "activity"],
    "health": ["health concerns", "disease", "lifespan", "health issues"],
    "temperament": ["temperament", "personality", "traits"],
    "family": ["family", "children", "ideal", "not ideal"],
    "overview": ["overview", "introduction", "summary", "final"],
    "training": ["training", "socialization", "commands", "tricks", "behaviors"],
    "diet": ["feeding", "nutritition", "food", "diet"]
}

OVERVIEW_KEYWORDS = ["overview", "introduction", "summary", "final"]

INSTRUCTION_TEMPLATES = {
    "care": ["Give care instructions for a {breed}. If you don't have breed-specific info, provide general dog care instructions.", 
             "How should a {breed} be cared for? If you don't have breed-specific info, provide general dog care instructions.", 
             "What are the care requirements of a {breed}? If you don't have breed-specific info, provide general dog care instructions."],
    "grooming": ["What grooming does a {breed} need? If you don't have breed-specific info, provide general dog grooming instructions.", 
                 "How do you groom a {breed}? If you don't have breed-specific info, provide general dog grooming instructions.", 
                 "Grooming instructions for a {breed}. If you don't have breed-specific info, provide general dog grooming instructions."],
    "exercise": ["How much exercise does a {breed} need? If you don't have breed-specific info, provide general dog exercise instructions.", 
                 "Exercise requirements for a {breed}. If you don't have breed-specific info, provide general dog grooming instructions.", 
                 "What activities should a {breed} do? If you don't have breed-specific info, provide general dog grooming instructions."],
    "health": ["What health problems commonly affect the {breed}? If you don't have breed-specific info, provide general dog health problems.", 
               "Health concerns for a {breed}. If you don't have breed-specific info, provide general dog health concerns.", 
               "List common diseases of a {breed}. If you don't have breed-specific info, provide general dog diseases."],
    "temperament": ["Describe the temperament of a {breed}. If you don't have breed-specific info, provide general dog temperment information.", 
                    "Personality traits of a {breed}. If you don't have breed-specific info, provide general dog personality information."],
    "family": ["Is the {breed} a good family dog? If you don't have breed-specific info, provide general dog family compatibility information.", 
               "How suitable is a {breed} for families? If you don't have breed-specific info, provide general dog family compatibility information."],
    "overview": ["Give a general overview of the {breed}.", 
                 "Provide a summary of the {breed}."],
    "training": ["Provide training tips for a {breed}. If you don't have breed-specific info, provide general dog training instruction.", 
                 "How should you train a {breed}? If you don't have breed-specific info, provide general dog training instruction.", 
                 "Training recommendations for a {breed}. If you don't have breed-specific info, provide general dog training instruction."],
    "diet": ["What do {breed} eat? If you don't have breed-specific info, provide general dog feeding instruction." ,
             "What kind of food is best for a {breed}? If you don't have breed-specific info, provide general dog dietary instruction.",
             "{breed} dietary recommendation. If you don't have breed-specific info, provide general dog dietary information."]
}

def is_overview_section(section):
    """Returns True if the section appears to be an overview/summary section"""
    title = section.get("section", "").lower()
    content = section.get("content", "").lower()
    return any(keyword in title or keyword in content[:200] for keyword in OVERVIEW_KEYWORDS)

def deduplicate_paragraphs(text):
    """Remove duplicate paragraphs from text"""
    paragraphs = text.split('\n')
    seen = set()
    unique_paragraphs = []
    
    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        
        # Normalize for comparison (lowercase, remove extra spaces)
        para_normalized = " ".join(para_clean.lower().split())
        
        if para_normalized not in seen:
            unique_paragraphs.append(para_clean)
            seen.add(para_normalized)
    
    return "\n".join(unique_paragraphs)

def extract_sections(sections, keywords, section_type):
    keywords = [k.lower() for k in keywords]
    extracted = []
    
    for sec in sections:
        title = sec.get("section", "").lower()
        content = sec.get("content", "").strip()
        
        # For non-overview sections, skip if this section is an overview
        if section_type != "overview" and is_overview_section(sec):
            continue
        
        # Check if section matches the keywords
        if any(k in title or k in content.lower() for k in keywords):
            extracted.append(content)
    
    combined_text = "\n".join(extracted)
    
    # Deduplicate paragraphs before returning
    return deduplicate_paragraphs(combined_text)

def make_examples(breed_data):
    breed = breed_data["breed"]
    sections = breed_data["sections"]
    examples = []

    for key, keywords in SECTION_KEYWORDS.items():
        text = extract_sections(sections, keywords, section_type=key)
        
        if not text.strip():
            text = "No breed-specific information available."
            # add data from generic dataset to fall back on
            if key == 'diet':
                text = text + " Feeding your canine companion a proper diet is crucial for their health and happiness! The best approach for most dogs is a high-quality, complete and balanced commercial dog food appropriate for their life stage (puppy, adult, or senior). Look for a product with an AAFCO (Association of American Feed Control Officials) statement confirming it meets nutritional levels for their specific life stage. \n\nWhen choosing, consider ingredients – a named meat source should be among the first few. There are various options like dry kibble, wet food, or a combination. Your dog's breed, activity level, and any health conditions will influence the best choice, so always consult your veterinarian for personalized recommendations.\n\nRemember to provide fresh, clean water at all times. Treats should be given sparingly and account for no more than 10% of their daily caloric intake. Crucially, avoid feeding dogs human foods toxic to them, such as chocolate, grapes, onions, and xylitol."
            elif key == 'exercise':
                text = text + " The amount of exercise your dog needs daily varies significantly based on their breed, age, size, and overall health. Generally, most adult dogs benefit from at least 30 minutes to 2 hours of physical activity per day, often broken into multiple sessions.\n\nGreat activities include:\n*   **Daily Walks:** Essential for physical and mental stimulation. Vary routes to keep it interesting.\n*   **Fetch or Playtime:** Engaging games like fetch, tug-of-war, or frisbee can provide high-intensity bursts.\n*   **Running/Jogging:** For high-energy breeds and healthy dogs, this can be excellent. Ensure your dog is conditioned for it.\n*   **Dog Parks/Socialization:** Offers exercise and crucial social interaction, if your dog enjoys it.\n*   **Mental Stimulation:** Puzzle toys, training sessions, and scent games are equally important for brain health.\n\nAlways consider your dog's individual needs. Overweight, elderly, or very young puppies may need less, while working breeds typically require more. Monitor for signs of fatigue and provide fresh water. Consult your vet to tailor an exercise plan perfect for your canine companion!"
            elif key == 'care':
                text = text + " Caring for a dog brings immense joy and responsibility! Ensure your canine companion thrives with these essentials:\n\n*   **Nutrition:** Provide high-quality dog food appropriate for their age, size, and activity level. Always have fresh, clean water available.\n*   **Exercise:** Dogs need daily physical activity, like walks, runs, or playtime, to maintain a healthy weight and mental well-being.\n*   **Grooming:** Regular brushing helps keep their coat healthy and reduces shedding. Bathe them as needed, trim nails, and maintain dental hygiene with vet-approved products or professional cleanings.\n*   **Health:** Schedule routine veterinary check-ups, keep vaccinations up-to-date, and use parasite prevention. Monitor for any changes in behavior or appetite.\n*   **Training & Socialization:** Positive reinforcement training establishes good behavior and strengthens your bond. Early socialization helps them become well-adjusted adults.\n*   **Love & Environment:** Offer a safe, comfortable home with plenty of enrichment, toys, and affection. Your dog thrives on companionship!"
            elif key == 'grooming':
                text = text + " Regular grooming is vital for your dog's health and happiness! Start with consistent brushing, which helps remove loose fur, prevent mats, and distribute natural oils. The type of brush depends on their coat: slicker brushes for medium to long coats, pin brushes for longer, denser coats, and rubber curry brushes for short coats. Aim for daily brushing for long-haired breeds and a few times a week for others.\n\nBathe your dog every 1-3 months, or as needed if they get dirty, using a dog-specific shampoo to protect their skin. Don't forget nail trims every 3-4 weeks to prevent discomfort and foot problems; introduce this gently with positive reinforcement. Clean their ears weekly with a vet-approved solution, wiping only visible areas, and gently clean around their eyes if there's discharge.\n\nDaily dental care, like brushing their teeth with canine toothpaste, is crucial for preventing gum disease. Regular grooming sessions also provide a great opportunity to check for any lumps, bumps, or skin issues, keeping your furry friend in top condition!"
            elif key == 'health':
                text = text + " Dogs, like all pets, are susceptible to various health issues that owners should be mindful of. Common concerns include dental disease, such as gingivitis and periodontitis, which can lead to pain and systemic health problems if left untreated. Obesity is another widespread issue, contributing to conditions like arthritis, diabetes, and heart disease. Many dogs experience allergies (environmental, food, or flea-related) causing skin irritation and ear infections. Parasites, both internal (e.g., worms, heartworm) and external (e.g., fleas, ticks), are common threats requiring regular prevention. Orthopedic problems, such as arthritis, hip dysplasia, and patellar luxation, are also frequently seen, especially in certain breeds. Gastrointestinal upsets (vomiting, diarrhea) are often reported, sometimes due to dietary indiscretion. More serious conditions like various cancers and heart disease also commonly affect older dogs. Regular veterinary check-ups, preventative care, a balanced diet, and appropriate exercise are key to managing and preventing these health challenges."
            elif key == 'temperament':
                text = text + " Dogs are known for their incredibly diverse yet generally amiable temperaments, making them beloved companions. While personality varies significantly by breed, individual socialization, and training, common traits include loyalty, affection, and intelligence. Most dogs are highly social animals, thriving on companionship and interaction with their human families. They often display playful and energetic dispositions, enjoying activities that engage both their minds and bodies. Dogs possess a strong desire to please, making them highly trainable and eager to learn. They can also be protective of their loved ones and homes, exhibiting vigilance and courage. Their emotional range is vast, encompassing joy, curiosity, contentment, and sometimes anxiety. Overall, dogs are adaptable, devoted, and joyful creatures who form deep bonds and enrich the lives of their owners."
            elif key == 'family':
                text = text + " Dogs generally make wonderful family pets, offering companionship, loyalty, and joy. Their suitability, however, varies significantly by breed. Many breeds are known for their gentle, patient nature with children, while others might be better suited for families with older kids or active lifestyles. All dogs thrive on being part of a 'pack' and require consistent training, early socialization, and regular exercise to be well-adjusted family members. They need attention and love to truly integrate and flourish. With proper care and understanding of their individual needs, a dog can become an incredibly affectionate and devoted member of your family."
            elif key == 'training':
                text = text + " Starting early with basic obedience (sit, stay, come) and socialization is crucial for a new puppy. Always use positive reinforcement with treats, praise, and toys to reward desired behaviors, making training a fun and positive experience. Consistency from all household members with commands and rules will prevent confusion. Keep training sessions short (5-10 minutes) and engaging, as puppies have limited attention spans. For potty training, frequent outdoor trips (after waking, eating, playing, and every 1-2 hours) are essential; reward immediately for success. Introduce crate training positively as a safe den. Socialize your puppy by gently exposing them to new sights, sounds, people, and vaccinated dogs to foster confidence and good manners. Begin leash training indoors before moving outside. Patience and repetition are your best tools!"
            print(f"[WARN] No '{key}' data for {breed}. Using placeholder.")

        for instr_template in INSTRUCTION_TEMPLATES[key]:
            examples.append({
                "instruction": instr_template.format(breed=breed),
                "input": "",
                "output": text
            })
    
    return examples

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

    random.shuffle(all_examples)
    train_examples, val_examples = train_test_split(all_examples, test_size=0.2, random_state=42)

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