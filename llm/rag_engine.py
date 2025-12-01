import os
import json
import re

# CONFIGURATION
DB_PATH = "./pet_knowlege/dog_database/"

class DogBreedRAG:
    def __init__(self, db_path):
        self.db_path = db_path
        self.breed_index = self._build_index()

    def _build_index(self):
        """
        Scans the folder and creates a lookup dictionary.
        """
        index = {}
        if not os.path.exists(self.db_path):
            print(f"Warning: Database path {self.db_path} does not exist.")
            return index

        for filename in os.listdir(self.db_path):
            if filename.endswith(".json"):
                breed_key = filename.replace(".json", "").replace("_", " ").lower()
                index[breed_key] = filename
        return index

    def _normalize_text(self, text):
        """
        Normalize text for better matching:
        - Convert to lowercase
        - Remove plural 's' at word boundaries
        - Handle common variations
        """
        text = text.lower()
        # Remove plural 's' (e.g., "huskies" -> "husky", "terriers" -> "terrier")
        text = re.sub(r'\b(\w+)ies\b', r'\1y', text)  # huskies -> husky
        text = re.sub(r'\b(\w+)s\b', r'\1', text)     # terriers -> terrier
        return text

    def identify_breed(self, query):
        """
        Improved keyword search to find the breed in the user's query.
        Handles plurals and partial matches.
        """
        query_normalized = self._normalize_text(query)
        # sort by length descending to match "Cavalier King Charles" before "King Charles"
        sorted_breeds = sorted(self.breed_index.keys(), key=len, reverse=True)
        
        for breed_name in sorted_breeds:
            breed_normalized = self._normalize_text(breed_name)
            
            # Exact match after normalization
            if breed_normalized in query_normalized:
                return self.breed_index[breed_name]
        
        # Fallback: Try partial matching for compound breed names
        # e.g., "wheaten terrier" should match "soft coated wheaten terrier"
        query_words = set(query_normalized.split())
        
        for breed_name in sorted_breeds:
            breed_normalized = self._normalize_text(breed_name)
            breed_words = set(breed_normalized.split())
            
            # If query words are a subset of breed words (at least 2 words match)
            matching_words = query_words & breed_words
            if len(matching_words) >= 2:
                return self.breed_index[breed_name]
        
        return None

    def format_json_to_context(self, json_data):
        """
        Converts your specific JSON structure into clean text for the LLM.
        This saves tokens compared to dumping raw JSON braces/quotes.
        """
        text_parts = []
        
        # Add the Breed Name header
        breed_name = json_data.get("breed", "Unknown Breed")
        text_parts.append(f"BREED DOCUMENTATION: {breed_name}\n")
        
        # Iterate through sections
        for section in json_data.get("sections", []):
            title = section.get("section", "").strip()
            content = section.get("content", "").strip()
            
            # Skip empty sections
            if not title and not content:
                continue
            
            # Format as:
            # ## History and Origin
            # The Silky originated in Australia...
            text_parts.append(f"## {title}")
            if content:
                text_parts.append(content)
            text_parts.append("") # Add spacing
            
        return "\n".join(text_parts)

    def retrieve_context(self, query):
        """
        Main function to get the context string for a query.
        """
        filename = self.identify_breed(query)
        
        if not filename:
            return None, None

        filepath = os.path.join(self.db_path, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                formatted_context = self.format_json_to_context(data)
                print(f"Retrieving context from: {filepath}")
                return data.get("breed"), formatted_context
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return None, None
