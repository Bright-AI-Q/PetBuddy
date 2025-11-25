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
for url in urls: 
    data = scrape_dog_breed(url)

    print(f"🐶 {data['breed']}")
    for sec in data["sections"][:3]:
        print(f"\n🦴 {sec['section']}")
        print(sec["content"][:300] + "...")

    # --- Save to file ---
    filename = safe_filename(data["breed"]) + ".json"
    filepath = os.path.join("dog_database", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved to {filepath}")