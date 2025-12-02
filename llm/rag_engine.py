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
        - Handle common variations (Saint to st)
        """
        text = re.sub(r'\bst\.?\b', 'Saint', text, flags=re.IGNORECASE)

        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)

        # Plural normalization
        text = re.sub(r'\b(\w+)ies\b', r'\1y', text)  # huskies → husky
        text = re.sub(r'\b(\w+)s\b', r'\1', text)     # terriers → terrier

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def identify_breed(self, query):
        """
        Identify dog breed from user query.
        Handles:
        - plurals
        - spacing variations (bullmastiff, bull mastiff)
        - St. / Saint variations
        - partial subset matching
        """
        query_norm = self._normalize_text(query)
        query_nospace = query_norm.replace(" ", "")

        # Sort long breeds first so "cavalier king charles spaniel"
        # matches before "king charles"
        sorted_breeds = sorted(self.breed_index.keys(), key=len, reverse=True)

        for breed_name in sorted_breeds:
            breed_norm = self._normalize_text(breed_name)
            breed_nospace = breed_norm.replace(" ", "")

            # Case 1: Exact normalized match
            if breed_norm in query_norm:
                return self.breed_index[breed_name]

            # Case 2: Concatenated match
            # handles: "bullmastiff" and "bull mastiff"
            if breed_nospace in query_nospace:
                return self.breed_index[breed_name]

        # Case 3: Partial match by shared words
        query_words = set(query_norm.split())

        for breed_name in sorted_breeds:
            breed_norm = self._normalize_text(breed_name)
            breed_words = set(breed_norm.split())

            # two or more shared words → good match
            matching = query_words & breed_words
            if len(matching) >= 2:
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
