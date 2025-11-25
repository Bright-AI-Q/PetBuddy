import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
import re
import os

# Data is outputed to dog_databse directory
os.makedirs("dog_database", exist_ok=True)

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
    'Pug': "https://www.dog-breeds.net/pug",
    'Redbone': "",
    'Rhodesian ridgeback': "https://www.dog-breeds.net/Rhodesian-ridgeback",
    'Rottweiler': "https://www.dog-breeds.net/Rottweiler",
    'Saint Bernard': "https://www.dog-breeds.net/saint-bernard",
    'Saluki': "https://www.dog-breeds.net/Saluki",
    'Samoyed': "https://www.dog-breeds.net/Samoyed",
    'Schipperke': "https://www.dog-breeds.net/Schipperke",
    'Scotch terrier': "https://www.dog-breeds.net/Scottish-terrier",
    'Scottish deerhound': "https://www.dog-breeds.net/Scottish-deerhound",
    'Sealyham terrier': "https://www.dog-breeds.net/sealyham-terrier/",
    'Shetland sheepdog': "https://www.dog-breeds.net/Shetland-sheepdog",
    'Shih-Tzu': "https://www.dog-breeds.net/shih-tzu",
    'Siberian husky': "https://www.dog-breeds.net/siberian-husky",
    'Silky terrier': "https://www.dog-breeds.net/silky-terrier",
    'Soft-coated wheaten terrier': "https://www.dog-breeds.net/soft-coated-wheaten-terrier",
    'Staffordshire bullterrier': "https://www.dog-breeds.net/Staffordshire-bull-terrier",
    'Standard poodle': "https://www.dog-breeds.net/Poodle",
    'Standard schnauzer': "https://www.dog-breeds.net/Standard-Schnauzer",
    'Sussex spaniel': "https://www.dog-breeds.net/sussex-spaniel",
    'Tibetan mastiff': "",
    'Tibetan terrier': "https://www.dog-breeds.net/Tibetan-terrier",
    'Toy poodle': "",
    'Toy terrier': "https://www.dog-breeds.net/toy-fox-terrier/",
    'Vizsla': "https://www.dog-breeds.net/Vizsla",
    'Walker hound':  "",
    'Weimaraner': "https://www.dog-breeds.net/Weimaraner",
    'Welsh springer spaniel': "https://www.dog-breeds.net/welsh-springer-spaniel",
    'Welsh Highland white terrier': "https://www.dog-breeds.net/west-highland-white-terrier",
    'Whippet': "https://www.dog-breeds.net/whippet",
    'Wire-haired fox terrier': "",
    'Yorkshire terrier': "https://www.dog-breeds.net/yorkshire-terrier",
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
        breed = title_text.split("Dog")[0].strip()
    else:
        breed = urlparse(url).path.strip("/").split("/")[-1].replace("-", " ").title()

    # --- Extract all text content sections ---
    sections = []
    # find headings we consider as section starts
    heading_tags = soup.find_all(re.compile(r"^h[2-4]$"))

    for heading in heading_tags:
        sec_title = heading.get_text(" ", strip=True)
        if not sec_title:
            continue

        # optional: clean emojis/prefix punctuation from heading
        clean_title = re.sub(r"^[^\w\s]+", "", sec_title).strip()
        if not clean_title:
            clean_title = sec_title

        # gather content until the next heading (h2-h4) or <hr>
        content_parts = []
        for sib in heading.next_siblings:
            # if sibling is another heading tag -> stop collecting
            if getattr(sib, "name", None) and re.match(r"^h[2-4]$", sib.name):
                break
            # stop at hr (footer divider)
            if getattr(sib, "name", None) and sib.name == "hr":
                break
            # if it's a Tag, extract its text (covers div, p, ul, etc.)
            if hasattr(sib, "get_text"):
                txt = sib.get_text(" ", strip=True)
                if txt:
                    content_parts.append(txt)
            # if it's a NavigableString, include text as well
            else:
                txt = str(sib).strip()
                if txt:
                    content_parts.append(txt)

        content = " ".join(content_parts).strip()

        # debug: if content empty, print small context (helps diagnosing why)
        if not content:
            # show what the heading's next element HTML looks like (first 200 chars)
            nxt = heading.find_next()
            snippet = ""
            if nxt is not None:
                snippet = str(nxt)[:200]
            print(f"⚠️ Empty section captured for heading: '{clean_title}'\n  next HTML snippet: {snippet}\n")

        sections.append({"section": clean_title, "content": content})

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
for breed, url in breed_url.items(): 
    if not url:
        print(f"Skipping {breed}: no URL provided.")
        continue
    
    data = scrape_dog_breed(url)
    data["breed"] = breed
    print(f"🐶 {breed}")
    for sec in data["sections"][:3]:
        print(f"\n🦴 {sec['section']}")
        print(sec["content"][:50] + "...")

    # --- Save to file ---
    filename = safe_filename(breed) + ".json"
    filepath = os.path.join("dog_database", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved to {filepath}")
