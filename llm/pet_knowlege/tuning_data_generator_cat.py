import requests
from bs4 import BeautifulSoup
import json
import re
import os 

os.makedirs("pet_database", exist_ok=True)

breed_url = {
    'Abyssinian': "https://www.wisdompanel.com/en-us/cat-breeds/abyssinian",
    'Bengal': "https://www.wisdompanel.com/en-us/cat-breeds/bengal",
    'Birman': "https://www.wisdompanel.com/en-us/cat-breeds/birman",
    'Bombay': "https://www.wisdompanel.com/en-us/cat-breeds/bombay",
    'British_Shorthair': "https://www.wisdompanel.com/en-us/cat-breeds/british-shorthair",
    'Egyptian_Mau': "https://www.wisdompanel.com/en-us/cat-breeds/egyptian-mau",
    'Maine_Coon': "https://www.wisdompanel.com/en-us/cat-breeds/maine-coon",
    'Persian': "https://www.wisdompanel.com/en-us/cat-breeds/persian",
    'Ragdoll': "https://www.wisdompanel.com/en-us/cat-breeds/ragdoll",
    'Russian_Blue': "https://www.wisdompanel.com/en-us/cat-breeds/russian-blue",
    'Siamese': "https://www.wisdompanel.com/en-us/cat-breeds/siamese-and-oriental-shorthair",
    'Sphynx': "https://www.wisdompanel.com/en-us/cat-breeds/sphynx"
}

GROUP_NAMES = [
    "Western",
    "African",
    "Asian",
    "Persian",
    "Siamese and oriental"
]

# Define the separator used *internally* in the health content
INTERNAL_SEPARATOR = "\n\n---\n\n"


def scrape_and_format_sections(url, breed_name):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        main_content_area = soup.find('main') or soup.find('div', class_=re.compile(r'content|article|main-body', re.I))

        if not main_content_area:
            print("Could not find the main content area. Targeting the whole body.")
            main_content_area = soup.body

        elements = main_content_area.find_all(['h2', 'h3', 'p', 'ul', 'ol', 'h1'])
        
        sections = []
        current_section_title = breed_name
        current_content = []
        
        def clean_text(text):
            text = text.get_text(strip=True)
            return re.sub(r'\s+', ' ', text).strip()

        for element in elements:
            if element.name in ['h1', 'h2', 'h3']:
                new_title = clean_text(element)
                
                if current_content:
                    sections.append({
                        "section": current_section_title,
                        "content": " ".join(current_content)
                    })
                
                current_section_title = new_title
                current_content = []

            elif element.name in ['p', 'ul', 'ol']:
                if element.find_parent('footer') or element.find_parent('nav'):
                    continue
                
                text = clean_text(element)
                if text:
                    if element.name in ['ul', 'ol']:
                        list_items = [clean_text(li) for li in element.find_all('li') if clean_text(li)]
                        text = " * " + " * ".join(list_items) 
                    
                    current_content.append(text)
        
        if current_content:
             sections.append({
                "section": current_section_title,
                "content": " ".join(current_content)
            })
            
        final_data = {
            "breed": breed_name,
            "sections": sections
        }
        
        return final_data

    except requests.exceptions.RequestException as e:
        print(f"Error during scraping: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- 2. To collect health conditions to one chunk ---
def consolidate_health_sections(sections_list, parent_section_title="Health"):
    """
    Consolidates detailed health sections into one 'Health' chunk, 
    stopping precisely when a non-health major heading is found.
    """
    
    consolidated_sections = []
    health_content_accumulator = []
    
    in_health_block = False
    parent_health_section = None

    # Helper function to finalize and clean the health chunk
    def finalize_health_chunk():
        nonlocal parent_health_section, health_content_accumulator
        
        if parent_health_section and health_content_accumulator:
            # Join content with the internal separator
            full_content = INTERNAL_SEPARATOR.join(health_content_accumulator)
            
            # Assign and append the completed, clean section
            parent_health_section['content'] = full_content
            consolidated_sections.append(parent_health_section)
        
        # Reset for the next block
        parent_health_section = None
        health_content_accumulator = []

    for section in sections_list:
        section_title = section['section'].strip()
        section_content = section['content'].strip()

        # If we are in the health block and hit the next section which is the group name
        if in_health_block and section_title in GROUP_NAMES:
            # 1. Finalize the health section (appends it to consolidated_sections)
            finalize_health_chunk()
            
            # 2. Reset flag
            in_health_block = False
            

        if parent_section_title in section_title and not in_health_block:
            in_health_block = True
            parent_health_section = {
                "section": parent_section_title, 
                "content": ""
            }
            health_content_accumulator.append(f"## {section_title}\n\n{section_content}")
            continue

        if in_health_block:
            health_content_accumulator.append(f"\n\n### {section_title}\n\n{section_content}")
            continue
            
        # If not in the health block (or just exited it), append the section normally
        if not in_health_block:
            consolidated_sections.append(section)


    # Final check to add the last processed health section if the loop ended while inside the block
    finalize_health_chunk()
        
    return consolidated_sections

def scrape_rag_pipeline(url, output_file, breed_name):
    """
    Executes the full pipeline: Scrape -> Consolidate -> Save.
    """
    print(f"Starting RAG data generation for: **{breed_name}**")
    
    
    structured_data = scrape_and_format_sections(url, breed_name)

    if not structured_data:
        print("Pipeline aborted: Failed to retrieve or structure content.")
        return

    raw_sections = structured_data['sections']
    
    print("Content scraped. Consolidating 'Health' sections...")
    final_sections = consolidate_health_sections(raw_sections, parent_section_title="Health")

    final_rag_data = {
        "breed": breed_name,
        "sections": final_sections
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_rag_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Pipeline complete. Successfully created RAG JSON file: **{output_file}**")
    print(f"Total sections created (after consolidation): **{len(final_rag_data['sections'])}**")


if __name__ == "__main__":
    for breed, url in breed_url.items():
        filename = breed + ".json"
        filename = os.path.join("pet_database", filename)
        scrape_rag_pipeline(url, filename, breed)