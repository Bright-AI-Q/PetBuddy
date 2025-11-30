"""
This prepares the test_samples.jsonl file.
Each sample is a json line with fields:
- breed
- question
- reference_answer
- tag (dog or cat). This is for filtering / debugging purposes (since we have a lot more dog breeds).
"""
from bs4 import BeautifulSoup
import json
import re
import requests
import random
from tuning_data_generator import breed_url

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_test_samples = os.path.join(script_dir, "test_samples.jsonl")

dog_breed_url = {
    "Affenpinscher": "",
    "Afghan hound": "",
    "African hunting dog": "",
    "Airedale": "https://www.everypaw.com/dog-insurance/dog-breed-guides/airedale-terrier-insurance-care-and-health-advice",
    "American Staffordshire terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-staffie",
    "Appenzeller": "",
    "Australian terrier": "",
    "Basenji": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-basenji",
    "Basset": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-basset-hound",
    "Beagle": "https://www.everypaw.com/dog-insurance/dog-breed-guides/beagle-insurance-care-and-health-advice",
    "Bedlington terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-bedlington-terrier",
    "Bernese mountain dog": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-bernese-mountain-dog",
    "Black-and-tan Coonhound": "",
    "Blenheim spaniel": "",
    "Bloodhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-bloodhound",
    "Bluetick": "",
    "Border collie": "https://www.everypaw.com/dog-insurance/dog-breed-guides/border-collie-insurance-care-and-health-advice",
    "Border terrier": "https://www.everypaw.com/dog-insurance/dog-breed-guides/boston-terrier-insurance-care-and-health-advice",
    "Borzoi": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-borzoi",
    "Boston bull": "https://www.everypaw.com/dog-insurance/dog-breed-guides/boston-terrier-insurance-care-and-health-advice",
    "Bouvier des Flandres": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-bouvier-des-flandres",
    "Boxer": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-boxer",
    "Brabancon griffon": "",
    "Briard": "",
    "Brittany spaniel": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-brittany",
    "Bull mastiff": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-bullmastiff",
    "Cairn": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-cairn-terrier",
    "Cardigan": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-cardigan-welsh-corgi",
    "Chesapeake Bay retriever": "",
    "Chihuahua": "https://www.everypaw.com/dog-insurance/dog-breed-guides/chihuahua-insurance-care-and-health-advice",
    "Chow": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-chow-chow",
    "Clumber": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-clumber-spaniels",
    "Cocker spaniel": "https://www.everypaw.com/dog-insurance/dog-breed-guides/cocker-spaniel-insurance-care-and-health-advice",
    "Collie": "",
    "Curly-coated retriever": "",
    "Dandie Dinmont": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-dandie-dinmont-terrier",
    "Dhole": "",
    "Dingo": "",
    "Doberman": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-doberman",
    "English foxhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-english-foxhound",
    "English setter": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-english-setter",
    "English springer": "https://www.everypaw.com/dog-insurance/dog-breed-guides/english-springer-spaniel-insurance-care-and-health-advice",
    "EntleBucher": "",
    "Eskimo dog": "",
    "Flat-coated retriever": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-flat-coated-retriever",
    "French bulldog": "https://www.everypaw.com/dog-insurance/dog-breed-guides/french-bulldog-insurance-care-and-health-advice",
    "German shepherd": "https://www.everypaw.com/dog-insurance/dog-breed-guides/german-shepherd-insurance-care-and-health-advice",
    "German short-haired pointer": "",
    "Giant schnauzer": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-giant-schnauzer",
    "Golden retriever": "https://www.everypaw.com/dog-insurance/dog-breed-guides/golden-retriever-insurance-care-and-health-advice",
    "Gordon setter": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-gordon-setter",
    "Great Dane": "https://www.everypaw.com/dog-insurance/dog-breed-guides/great-dane-insurance-care-and-health-advice",
    "Great Pyrenees": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-great-pyrenees",
    "Greater Swiss Mountain dog": "",
    "Groenendael": "",
    "Ibizan hound": "",
    "Irish setter": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-irish-setter",
    "Irish terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-irish-terrier",
    "Irish water spaniel": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-irish-water-spaniel",
    "Irish wolfhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-irish-wolfhound",
    "Italian greyhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-italian-greyhound",
    "Japanese spaniel": "",
    "Keeshond": "",
    "Kelpie": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-australian-kelpie",
    "Kerry blue terrier": "",
    "Komondor": "",
    "Kuvasz": "",
    "Labrador retriever": "https://www.everypaw.com/dog-insurance/dog-breed-guides/labrador-insurance-care-and-health-advice",
    "Lakeland terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-lakeland-terrier",
    "Leonberg": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-leonberger",
    "Lhasa": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-lhasa-apso",
    "Malamute": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-alaskan-malamute",
    "Malinois": "",
    "Maltese dog": "https://www.everypaw.com/dog-insurance/dog-breed-guides/maltese-insurance-care-and-health-advice",
    "Mexican hairless": "",
    "Miniature pinscher": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-miniature-pinscher",
    "Miniature poodle": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-miniature-poodle",
    "Miniature schnauzer": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-miniature-schnauzer",
    "Newfoundland": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-newfoundland",
    "Norfolk terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-norfolk-terrier",
    "Norwegian elkhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-norwegian-elkhound",
    "Norwich terrier": "",
    "Old English sheepdog": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-old-english-sheepdog",
    "Otterhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-otterhound",
    "Papillon": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-papillon",
    "Pekinese": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-pekingese",
    "Pembroke": "https://www.everypaw.com/dog-insurance/dog-breed-guides/pembroke-welsh-corgi-insurance-care-and-health-advice",
    "Pomeranian": "https://www.everypaw.com/dog-insurance/dog-breed-guides/pomeranian-insurance-care-and-health-advice",
    "Pug": "https://www.everypaw.com/dog-insurance/dog-breed-guides/pug-insurance-care-and-health-advice",
    "Redbone": "",
    "Rhodesian ridgeback": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-rhodesian-ridgeback",
    "Rottweiler": "https://www.everypaw.com/dog-insurance/dog-breed-guides/rottweiler-insurance-care-and-health-advice",
    "Saint Bernard": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-st-bernard",
    "Saluki": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-saluki",
    "Samoyed": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-samoyed",
    "Schipperke": "",
    "Scotch terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-scottish-terrier",
    "Scottish deerhound": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-deerhound",
    "Sealyham terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-sealyham-terrier",
    "Shetland sheepdog": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-shetland-sheepdog",
    "Shih-Tzu": "https://www.everypaw.com/dog-insurance/dog-breed-guides/shih-tzu-insurance-care-and-health-advice",
    "Siberian husky": "https://www.everypaw.com/dog-insurance/dog-breed-guides/siberian-husky-insurance-care-and-health-advice",
    "Silky terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-australian-silky-terrier",
    "Soft-coated wheaten terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-wheaten-terrier",
    "Staffordshire bullterrier": "https://www.everypaw.com/dog-insurance/dog-breed-guides/staffordshire-bull-terrier-insurance-care-and-health-advice",
    "Standard poodle": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-poodle",
    "Standard schnauzer": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-schnauzer",
    "Sussex spaniel": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-sussex-spaniel",
    "Tibetan mastiff": "",
    "Tibetan terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-tibetan-terrier",
    "Toy poodle": "",
    "Toy terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-russian-toy-terrier",
    "Vizsla": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-hungarian-vizsla",
    "Walker hound": "",
    "Weimaraner": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-weimaraner",
    "Welsh springer spaniel": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-welsh-springer-spaniel",
    "Welsh Highland white terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-welsh-terrier",
    "Whippet": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-whippet",
    "Wire-haired fox terrier": "https://www.borrowmydoggy.com/doggypedia/dog-breed-guides-wire-fox-terrier",
    "Yorkshire terrier": "https://www.everypaw.com/dog-insurance/dog-breed-guides/yorkshire-terrier-insurance-care-and-health-advice",
}

cat_breed_url = {
    "Abyssinian": "https://www.everypaw.com/cat-insurance/cat-breed-guides/abyssinian-cat-insurance-care-and-health-advice",
    "Bengal": "https://www.everypaw.com/cat-insurance/cat-breed-guides/bengal-cat-insurance-care-and-health-advice",
    "Birman": "",  # Not burmese
    "Bombay": "https://www.everypaw.com/cat-insurance/cat-breed-guides/bombay-cat-insurance-care-and-health-advice",
    "British_Shorthair": "https://www.everypaw.com/cat-insurance/cat-breed-guides/british-shorthair-cat-insurance-care-and-health-advice",
    "Egyptian_Mau": "https://www.everypaw.com/cat-insurance/cat-breed-guides/egyptian-mau-cat-insurance-care-and-health-advice",
    "Maine_Coon": "https://www.everypaw.com/cat-insurance/cat-breed-guides/maine-coon-cat-insurance-care-and-health-advice",
    "Persian": "https://www.everypaw.com/cat-insurance/cat-breed-guides/persian-cat-insurance-care-and-health-advice",
    "Ragdoll": "https://www.everypaw.com/cat-insurance/cat-breed-guides/ragdoll-cat-insurance-care-and-health-advice",
    "Russian_Blue": "https://www.everypaw.com/cat-insurance/cat-breed-guides/russian-blue-cat-insurance-care-and-health-advice",
    "Siamese": "https://www.everypaw.com/cat-insurance/cat-breed-guides/siamese-cat-insurance-care-and-health-advice",
    "Sphynx": "https://www.everypaw.com/cat-insurance/cat-breed-guides/sphynx-cat-insurance-care-and-health-advice",
}

num_classes_dogs = len(dog_breed_url)
num_classes_cats = len(cat_breed_url)
num_test_samples = 100

def scrape_everypaw_qna(url):
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    header = soup.find("h2", string=re.compile(r"Quick\s+.*?\s+Q\s*&?A", re.IGNORECASE))
    if not header:
        raise Exception("No 'Quick Q&A' section found.")
    qa_container = header.find_parent().find_next("div", class_="tabs")
    if not qa_container:
        raise Exception("Q&A tabs container not found below the header.")

    qa_pairs = []
    tabs = qa_container.find_all("div", class_="tab")

    for tab in tabs:
        question = tab.find("h3").get_text(strip=True)

        answers = tab.find("div", class_="tab-content").find_all("p")
        answer_text = "\n".join(p.get_text(strip=True) for p in answers)

        qa_pairs.append({
            "question": question,
            "answer": answer_text,
        })

    return qa_pairs

def scrape_borrowmydoggy_qna(url):
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    QUESTION_PATTERNS = [
        re.compile(r"What is the temperament of .*? like\?", re.IGNORECASE),
        re.compile(r"How much exercise does a .*? need\?", re.IGNORECASE),
        re.compile(r"Do .*? need a lot of grooming\?", re.IGNORECASE),
        re.compile(r"Are .*? easy to train\?", re.IGNORECASE),
        re.compile(r"What do .*? eat\?", re.IGNORECASE),
        re.compile(r"Are .*? healthy\?", re.IGNORECASE),
        re.compile(r"What Is the Temperament of .*? \?", re.IGNORECASE),
        re.compile(r"What Kind of Exercise Do .*? Need\?", re.IGNORECASE)
    ]

    text = soup.get_text(separator="\n")

    # Normalize whitespace a bit for easier slicing
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse multiple blank lines
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces

    # Find all question matches
    matches = []
    for pattern in QUESTION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "question": m.group(0),
            })

    # Sort by appearance in the document
    matches.sort(key=lambda m: m["start"])

    qa_pairs = []
    for i, m in enumerate(matches):
        answer_start = m["end"]
        if i + 1 < len(matches):
            answer_end = matches[i + 1]["start"]
        else:
            answer_end = len(text)

        answer = text[answer_start:answer_end].strip()

        # Optional: strip leading newlines
        answer = answer.lstrip("\n").strip()

        qa_pairs.append({
            "question": m["question"],
            "answer": answer,
        })

    return qa_pairs
    

def get_qna_data(breed_url, tag):
    all_breed_qna_data = []
    for key in breed_url:
        url = breed_url[key]
        if url:
            try:
                if "www.everypaw.com" in url:
                    qna_data = scrape_everypaw_qna(url)
                elif "www.borrowmydoggy.com" in url:
                    qna_data = scrape_borrowmydoggy_qna(url)
                else:
                    print(f"Unsupported URL for breed {key}: {url}")
                    continue
                qna_data = [{"breed": key, "question": item["question"], "reference_answer": item["answer"], "tag": tag} for item in qna_data]
                all_breed_qna_data.extend(qna_data)
            except Exception as e:
                print(f"Error processing breed {key}: {e}") 
    return all_breed_qna_data

if __name__ == "__main__":
    print("Scrapping test samples...")
    for key in dog_breed_url:
        if breed_url[key] == "":
            print("Skipping dog breed with no train/val data:", key)
            dog_breed_url[key] = ""
    dog_qna_data = get_qna_data(dog_breed_url, "dog")
    print("Dog data scraped:", len(dog_qna_data))
    
    cat_qna_data = get_qna_data(cat_breed_url, "cat")
    print("Cat data scraped:", len(cat_qna_data))

    num_cat_samples = int((num_classes_cats / (num_classes_dogs + num_classes_cats)) * num_test_samples)
    num_dog_samples = num_test_samples - num_cat_samples
    
    random.seed(42)
    random.shuffle(dog_qna_data)
    random.shuffle(cat_qna_data)

    cat_samples = cat_qna_data[:num_cat_samples]
    dog_samples = dog_qna_data[:num_dog_samples]

    all_samples = dog_samples + cat_samples

    print(f"Total test samples: {len(all_samples)} (Dogs: {len(dog_samples)}, Cats: {len(cat_samples)})")
    with open(output_test_samples, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    