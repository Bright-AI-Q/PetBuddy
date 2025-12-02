"""
PetMD Cat Breed Scraper
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os

os.makedirs("pet_database", exist_ok=True)

breed_url = {
    'Abyssinian': "https://www.petmd.com/cat/breeds/abyssinian",
    'Bengal': "https://www.petmd.com/cat/breeds/bengal",
    'Birman': "https://www.petmd.com/cat/breeds/birman",
    'Bombay': "https://www.petmd.com/cat/breeds/bombay",
    'British_Shorthair': "https://www.petmd.com/cat/breeds/british-shorthair",
    'Egyptian_Mau': "https://www.petmd.com/cat/breeds/egyptian-mau",
    'Maine_Coon': "https://www.petmd.com/cat/breeds/maine-coon",
    'Persian': "https://www.petmd.com/cat/breeds/persian",
    'Ragdoll': "https://www.petmd.com/cat/breeds/ragdoll",
    'Russian_Blue': "https://www.petmd.com/cat/breeds/russian-blue",
    'Siamese': "https://www.petmd.com/cat/breeds/siamese",
    'Sphynx': "https://www.petmd.com/cat/breeds/sphynx"
}

def deduplicate_paragraphs(text):
    """Remove duplicate paragraphs from text"""
    sentences = text.split('.')
    seen = set()
    unique_sentences = []
    
    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        
        # Normalize for comparison (lowercase, remove extra spaces)
        sent_normalized = " ".join(sent_clean.lower().split())
        
        if sent_normalized not in seen:
            unique_sentences.append(sent_clean)
            seen.add(sent_normalized)
    
    return ". ".join(unique_sentences)

def scrape_petmd_breed(breed, url):
    """
    Scraper specifically for petmd.com cat breeds
    Extracts all sections including health issues and FAQs
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
    
    # Extract breed name
    breed_name = breed
    h1 = soup.find('h1')
    if h1:
        breed_name = h1.get_text(strip=True)
    
    breed_data = {
        "breed": breed,
        "sections": []
    }
    
    # Add main title section
    breed_data["sections"].append({
        "section": breed_name,
        "content": ""
    })
    
    # Remove unwanted elements
    for unwanted in soup.find_all(['script', 'style', 'nav', 'footer']):
        unwanted.decompose()
    
    # Get all text
    full_text = soup.get_text(separator='\n', strip=True)
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    
    # === EXTRACT INTRO/ABOUT ===
    intro_paras = []
    first_p = soup.find('p')
    if first_p:
        # Get first 2-3 paragraphs as intro
        current = first_p
        count = 0
        while current and count < 3:
            if current.name == 'p':
                text = current.get_text(strip=True)
                if text and len(text) > 50:
                    intro_paras.append(text)
                    count += 1
            current = current.find_next_sibling()
    
    if intro_paras:
        breed_data["sections"].append({
            "section": f"About the {breed_name}",
            "content": " ".join(intro_paras)
        })
    
    # === EXTRACT OVERVIEW/STATS ===
    stats_content = extract_petmd_stats(soup, lines)
    if stats_content:
        breed_data["sections"].append({
            "section": "Overview",
            "content": deduplicate_paragraphs(stats_content)
        })
        
    # === EXTRACT APPEARANCE ===
    appearance_content = extract_petmd_section(soup, [
        'coat', 'colors', 'eyes', 'ears', 'appearance'
    ])
    if appearance_content:
        breed_data["sections"].append({
            "section": "Appearance",
            "content": deduplicate_paragraphs(appearance_content)
        })
    
    
    # === EXTRACT HEALTH ISSUES ===
    health_content = extract_petmd_health(soup)
    if health_content:
        breed_data["sections"].append({
            "section": "Health Issues and Concerns",
            "content": deduplicate_paragraphs(health_content)
        })
    
    # === EXTRACT NUTRITION/FEEDING ===
    nutrition_content = extract_petmd_section(soup, [
        'what to feed', 'how to feed', 'how much', 'nutritional tips'
    ])
    if nutrition_content:
        breed_data["sections"].append({
            "section": "Nutrition and Diet",
            "content": deduplicate_paragraphs(nutrition_content)
        })
    
    # === EXTRACT BEHAVIOR/TRAINING ===
    behavior_content = extract_petmd_section(soup, [
        'behavior', 'training', 'personality', 'temperament'
    ])
    if behavior_content:
        breed_data["sections"].append({
            "section": "Behavior and Training",
            "content": deduplicate_paragraphs(behavior_content)
        })
    
    # === EXTRACT GROOMING ===
    grooming_content = extract_petmd_section(soup, [
        'grooming guide', 'coat care', 'skin care', 'eye care', 'ear care'
    ])
    if grooming_content:
        breed_data["sections"].append({
            "section": "Grooming and Care",
            "content": deduplicate_paragraphs(grooming_content)
        })
    
    # === EXTRACT FAQ ===
    faq_content = extract_petmd_faq(soup)
    if faq_content:
        breed_data["sections"].append({
            "section": "Frequently Asked Questions (FAQ)",
            "content": deduplicate_paragraphs(faq_content)
        })
    
    # === EXTRACT CONSIDERATIONS ===
    considerations = extract_petmd_section(soup, ['considerations for pet parents'])
    if considerations:
        breed_data["sections"].append({
            "section": "Considerations for Pet Parents",
            "content": deduplicate_paragraphs(considerations)
        })
    
    return breed_data

def extract_petmd_stats(soup, lines):
    """Extract breed statistics - full paragraph with stats"""
    stats_content = []
    
    # Look for the "Caring for" section which typically has stats
    caring_heading = soup.find(['h2', 'h3'], 
        text=re.compile(r'caring\s+for', re.I))
    
    if caring_heading:
        # Get the next paragraph after "Caring for"
        next_p = caring_heading.find_next('p')
        if next_p:
            text = next_p.get_text(strip=True)
            if text and len(text) > 50:
                stats_content.append(text)
    
    # If no "Caring for" section, look for paragraphs with size/weight/lifespan
    if not stats_content:
        for p in soup.find_all('p', limit=15):
            text = p.get_text(strip=True)
            # Check if paragraph contains stats keywords
            if any(keyword in text.lower() for keyword in ['inches tall', 'pounds', 'weigh', 'stand']):
                if len(text) > 50:
                    stats_content.append(text)
                    break
    
    # Also look for lifespan paragraph
    lifespan_heading = soup.find(['h2', 'h3'], 
        text=re.compile(r'health\s+issues?', re.I))
    
    if lifespan_heading:
        # Check next paragraph for lifespan info
        next_p = lifespan_heading.find_next('p')
        if next_p:
            text = next_p.get_text(strip=True)
            if 'lifespan' in text.lower() or 'years' in text.lower():
                if text not in stats_content:
                    stats_content.append(text)
    
    return "\n\n".join(stats_content) if stats_content else ""

def extract_petmd_health(soup):
    """Extract health issues section"""
    health_content = []
    
    # Find "Health Issues" heading
    health_heading = soup.find(['h2', 'h3'], 
        text=re.compile(r'health\s+issues?', re.I))
    
    if health_heading:
        # Get content after this heading until next major section
        current = health_heading.find_next_sibling()
        
        while current:
            # Stop at next h2
            if current.name == 'h2':
                break
            
            # Look for disease headings (h3)
            if current.name == 'h3':
                disease_name = current.get_text(strip=True)
                disease_content = []
                
                # Get content for this disease
                disease_elem = current.find_next_sibling()
                while disease_elem:
                    if disease_elem.name in ['h2', 'h3']:
                        break
                    
                    if disease_elem.name == 'p':
                        text = disease_elem.get_text(strip=True)
                        if text:
                            disease_content.append(text)
                    
                    # Get lists (symptoms)
                    elif disease_elem.name in ['ul', 'ol']:
                        disease_content.append("Symptoms include:")
                        for li in disease_elem.find_all('li', recursive=False):
                            text = li.get_text(strip=True)
                            if text:
                                disease_content.append(f"  • {text}")
                    
                    disease_elem = disease_elem.find_next_sibling()
                
                if disease_content:
                    health_content.append(f"\n{disease_name}:\n" + "\n".join(disease_content))
            
            current = current.find_next_sibling()
    
    return "\n".join(health_content) if health_content else ""

def extract_petmd_section(soup, section_keywords):
    """Extract a specific section by keywords"""
    content = []
    
    # Find heading matching any keyword
    for keyword in section_keywords:
        heading = soup.find(['h2', 'h3', 'h4'], 
            text=re.compile(keyword, re.I))
        
        if heading:
            # Get the heading title
            section_title = heading.get_text(strip=True)
            
            # Get content after heading
            current = heading.find_next_sibling()
            while current:
                # Stop at next h2 or h3
                if current.name in ['h2', 'h3'] and current != heading:
                    break
                
                if current.name == 'p':
                    text = current.get_text(strip=True)
                    if text and len(text) > 20:
                        content.append(text)
                
                elif current.name in ['ul', 'ol']:
                    for li in current.find_all('li', recursive=False):
                        text = li.get_text(strip=True)
                        if text:
                            content.append(f"• {text}")
                
                elif current.name == 'h4':
                    # Sub-heading
                    sub_title = current.get_text(strip=True)
                    content.append(f"\n{sub_title}:")
                
                current = current.find_next_sibling()
    
    return " ".join(content) if content else ""

def extract_petmd_faq(soup):
    """Extract FAQ section"""
    faq_content = []
    
    # Find FAQ heading
    faq_heading = soup.find(['h2', 'h3'], 
        text=re.compile(r'faq|frequently\s+asked', re.I))
    
    if faq_heading:
        current = faq_heading.find_next_sibling()
        
        while current:
            if current.name == 'h2':
                break
            
            # Questions are typically h3 or h4
            if current.name in ['h3', 'h4']:
                question = current.get_text(strip=True)
                
                # Get answer (next paragraph)
                answer_elem = current.find_next_sibling('p')
                if answer_elem:
                    answer = answer_elem.get_text(strip=True)
                    faq_content.append(f"Q: {question}\nA: {answer}")
            
            current = current.find_next_sibling()
    
    return "\n\n".join(faq_content) if faq_content else ""

def save_to_jsonl(breed_data, output_file):
    """Save as JSONL"""
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(breed_data, f, ensure_ascii=False)
        f.write('\n')

def save_to_json(breed_data, output_file):
    """Save as JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(breed_data, f, indent=4, ensure_ascii=False)

def print_preview(breed_data):
    """Print preview"""
    print(f"\n{'='*70}")
    print(f"BREED: {breed_data['breed']}")
    print(f"{'='*70}")
    
    for i, section in enumerate(breed_data['sections'], 1):
        content = section['content']
        preview = content[:150] + "..." if len(content) > 150 else content
        print(f"\n[{i}] {section['section']}")
        print(f"    {preview}")
    
    print(f"\n{'='*70}")
    print(f"Total sections: {len(breed_data['sections'])}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    print("="*70)
    print("CAT BREED SCRAPER FROM PETMD")
    print("="*70)
    
    for breed, url in breed_url.items():
        print(f"Scraping: {url}\n")
        
        try:
            breed_data = scrape_petmd_breed(breed, url)
            
            if breed_data:
                filename = breed + ".json"
                filename = os.path.join("pet_database", filename)
                save_to_json(breed_data, filename)
                
                print("✓ SUCCESS! Files saved:")
                print(f"  • {filename}.json")
                
                print_preview(breed_data)
            else:
                print("✗ Failed to scrape")
    
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(2)