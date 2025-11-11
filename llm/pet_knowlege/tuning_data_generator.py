import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
import re
import os

# Data is outputed to dog_databse directory
os.makedirs("dog_database", exist_ok=True)


# To populate more websites to scrape here
urls = ["https://www.dog-breeds.net/affenpinscher/", 
        "https://www.dog-breeds.net/afghan-hound/",
        "https://www.dog-breeds.net/airedale-terrier/",
        ]
breed_url = {
    'Affenpinscher':"https://www.dog-breeds.net/affenpinscher/",
    'Afghan hound': "https://www.dog-breeds.net/afghan-hound/",
    'African hunting dog': "",
    'Airedale': "https://www.dog-breeds.net/airedale-terrier/",
    'American Staffordshire terrier': "https://www.dog-breeds.net/american-Staffordshire-terrier",
    'Appenzeller': "",
    'Australian terrier': "https://www.dog-breeds.net/australian-terrier",
    'Basenji': "https://www.dog-breeds.net/basenji",
    'Basset': "https://www.dog-breeds.net/basset-hound",
    'Beagle': "https://www.dog-breeds.net/beagle", 
    'Bedlington terrier': "https://www.dog-breeds.net/bedlington-terrier",
    'Bernese mountain dog': "https://www.dog-breeds.net/bernese-mountain-dog",
    'Black-and-tan Coonhound': "https://www.dog-breeds.net/black-and-tan-coonhound",
    'Blenheim spaniel': "",
    'Bloodhound': "https://www.dog-breeds.net/Bloodhound",
    'Bluetick': "",
    'Border collie': "https://www.dog-breeds.net/border-collie",
    'Border terrier': "https://www.dog-breeds.net/border-terrier",
    'Borzoi': "https://www.dog-breeds.net/Borzoi",
    'Boston bull': "https://www.dog-breeds.net/boston-terrier",
    'Bouvier des Flandres': "",
    'Boxer': "https://www.dog-breeds.net/Boxer",
    'Brabancon griffon': "",
    'Briard': "https://www.dog-breeds.net/Briard",
    'Brittany spaniel': "https://www.dog-breeds.net/Brittany-spaniel",
    'Bull mastiff': "https://www.dog-breeds.net/Bullmastiff",
    'Cairn': "https://www.dog-breeds.net/cairn-terrier",
    'Cardigan': "https://www.dog-breeds.net/cardigan-welsh-corgi",
    'Chesapeake Bay retriever': "",
    'Chihuahua': "https://www.dog-breeds.net/Chihuahua",
    'Chow': "https://www.dog-breeds.net/chow-chow",
    'Clumber': "https://www.dog-breeds.net/clumber-Spaniel",
    'Cocker spaniel': "https://www.dog-breeds.net/cocker-Spaniel",
    'Collie': "https://www.dog-breeds.net/collie",
    'Curly-coated retriever': "https://www.dog-breeds.net/curly-coated-retriever",
    'Dandie Dinmont': "https://www.dog-breeds.net/Dandie-dinmont-terrier",
    'Dhole': "",
    'Dingo': "", 
    'Doberman': "https://www.dog-breeds.net/doberman-pinscher",
    'English foxhound': "https://www.dog-breeds.net/English-foxhound",
    'English setter': "https://www.dog-breeds.net/English-setter",
    'English springer': "https://www.dog-breeds.net/English-springer-spaniel",
    'EntleBucher': "",
    'Eskimo dog': "",
    'Flat-coated retriever': "https://www.dog-breeds.net/flat-coated-retriever",
    'French bulldog': "https://www.dog-breeds.net/french-bulldog",
    'German shepherd': "https://www.dog-breeds.net/German-shepherd",
    'German short-haired pointer': "https://www.dog-breeds.net/German-shorthaired-pointer",
    'Giant schnauzer': "https://www.dog-breeds.net/giant-schnauzer",
    'Golden retriever': "https://www.dog-breeds.net/golden-retriever",
    'Gordon setter': "https://www.dog-breeds.net/gordon-setter",
    'Great Dane': "https://www.dog-breeds.net/great-dane",
    'Great Pyrenees': "https://www.dog-breeds.net/great-pyrenees",
    'Greater Swiss Mountain dog': "https://www.dog-breeds.net/greater-swiss-mountain-dog",
    'Groenendael':"", 
    'Ibizan hound': "https://www.dog-breeds.net/ibizan-hound",
    'Irish setter': "https://www.dog-breeds.net/irish-setter/",
    'Irish terrier': "",
    'Irish water spaniel': "https://www.dog-breeds.net/irish-water-spaniel",
    'Irish wolfhound': "https://www.dog-breeds.net/irish-wolfhound",
    'Italian greyhound': "https://www.dog-breeds.net/italian-greyhound",
    'Japanese spaniel': "https://www.dog-breeds.net/Japanese-chin",
    'Keeshond': "https://www.dog-breeds.net/Keeshond",
    'Kelpie': "",
    'Kerry blue terrier': "https://www.dog-breeds.net/kerry-blue-terrier",
    'Komondor': "https://www.dog-breeds.net/Komondor",
    'Kuvasz': "https://www.dog-breeds.net/Kuvasz",
    'Labrador retriever': "https://www.dog-breeds.net/labrador-retriever",
    'Lakeland terrier': "https://www.dog-breeds.net/lakeland-terrier",
    'Leonberg': "",
    'Lhasa': "https://www.dog-breeds.net/lhasa-apso",
    'Malamute': "",
    'Malinois': "",
    'Maltese dog': "https://www.dog-breeds.net/Maltese",
    'Mexican hairless': "",
    'Miniature pinscher': "https://www.dog-breeds.net/Miniature-pinscher",
    'Miniature poodle': "",
    'Miniature schnauzer': "https://www.dog-breeds.net/Miniature-schnauzer",
    'Newfoundland': "https://www.dog-breeds.net/Newfoundland",
    'Norfolk terrier': "https://www.dog-breeds.net/Norfolk-terrier",
    'Norwegian elkhound': "https://www.dog-breeds.net/Norwegian-elkhound",
    'Norwich terrier': "https://www.dog-breeds.net/Norwich-terrier",
    'Old English sheepdog': "https://www.dog-breeds.net/old-english-sheepdog",
    'Otterhound': "https://www.dog-breeds.net/Otterhound",
    'Papillon': "https://www.dog-breeds.net/Papillon",
    'Pekinese': "https://www.dog-breeds.net/Pekingese",
    'Pembroke': "https://www.dog-breeds.net/Pembroke-welsh-corgi",
    'Pomeranian': "https://www.dog-breeds.net/Pomeranian",
    'Pug': 
    'Redbone':
    'Rhodesian ridgeback':
    'Rottweiler':
    'Saint Bernard':
    'Saluki':
    'Samoyed':
    'Schipperke':
    'Scotch terrier':
    'Scottish deerhound':
    'Sealyham terrier':
    'Shetland sheepdog':
    'Shih-Tzu':
    'Siberian husky':
    'Silky terrier':
    'Soft-coated wheaten terrier':
    'Staffordshire bullterrier':
    'Standard poodle':
    'Standard schnauzer':
    'Sussex spaniel':
    'Tibetan mastiff':
    'Tibetan terrier':
    'Toy poodle':
    'Toy terrier':
    'Vizsla':
    'Walker hound':
    'Weimaraner':
    'Welsh springer spaniel':
    'Welsh Highland white terrier':
    'Whippet':
    'Wire-haired fox terrier':
    'Yorkshire terrier':
}
headers = {"User-Agent": "Mozilla/5.0"}

def scrape_dog_breed(url):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # --- Extract breed name ---
    title_tag = soup.select_one("title")
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        # clean up extra words like "Dog Breed Information"
        breed = title_text.split("Dog")[0].strip()
    else:
        # 2️⃣ Fallback: extract from URL (affenpinscher)
        breed = urlparse(url).path.strip("/").split("/")[-1].replace("-", " ").title()

    # --- Extract all text content sections ---
    sections = []
    current_section = {"section": "General", "content": ""}

    for tag in soup.select("h2, h3, p, li"):
        if tag.name in ["h2", "h3"]:
            # start a new section
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"section": tag.get_text(strip=True), "content": ""}
        else:
            text = tag.get_text(" ", strip=True)
            if text:
                current_section["content"] += text + " "

    if current_section["content"]:
        sections.append(current_section)

    # --- Clean unused sections ---
    cleaned_sections = [s for s in sections if s["section"].lower() != "general"]

    ## Clean links to other pages from scraped data
    for item in cleaned_sections:
        item["section"] = remove_emojis(item["section"])  # remove emoji from headers
        item["content"] = remove_link_sentences(item["content"]) # remove link sentences
        item["content"] = remove_emojis(item["content"]) # remove emojis from contents

    return {
        "breed": cleaned_sections[0]["section"],
        "sections": cleaned_sections[:-1] # last section is removed because it usually contains ads, copyrights, etc.
    }

def remove_emojis(text):
    # Remove all emojis and special symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002B00-\U00002BFF"  # Misc symbols & arrows
        "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
        "]",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)

def remove_link_sentences(text):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    # Filter sentences that likely correspond to links
    cleaned = [
        s for s in sentences
        if not re.search(r'\b(click here|read more|learn more|find out|see here|article|page|directory|Dog Health Dictionary|Healthy Dog Diet)\b', s, re.IGNORECASE)
    ]
    return " ".join(cleaned)

# Clean the breed name to use as a safe filename
def safe_filename(name):
    # Keep letters, numbers, dash, underscore; replace spaces with underscore
    name = re.sub(r"[^\w\s-]", "", name)
    name = name.strip().replace(" ", "_")
    return name

# --- Run scraper ---
for breed, url in breed_url: 
    data = scrape_dog_breed(url)
    data["breed"] = breed
    print(f"🐶 {breed}")
    for sec in data["sections"][:3]:
        print(f"\n🦴 {sec['section']}")
        print(sec["content"][:300] + "...")

    # --- Save to file ---
    filename = safe_filename(breed) + ".json"
    filepath = os.path.join("dog_database", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved to {filepath}")