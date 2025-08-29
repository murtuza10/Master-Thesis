import os
import sys
import json
import torch
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.quantization_config import QuantizationMethod
from transformers import AutoConfig
from typing import Dict, List, Tuple, Optional, Any

sys.path.append(os.path.abspath('..'))

from Evaluation_Files.calculate_metrics_multiple import evaluate_all


class QwenNERPromptGenerator:
    """Generates specialized prompts for each Qwen model"""
    
    @staticmethod
    def generate_32b_ner_prompts(text: str) -> Tuple[str, str]:
        """Generate prompts for Qwen2.5-32B-Instruct (11 classes)"""
        system_prompt = """
        ### Instruction ###
You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying entities related to **Soil**, **Location**, and **Time Statements**.
Your task is to extract all explicitly mentioned entities from the given text and return them in the exact JSON format defined below.

### Entity Categories ###
1. **Soil**
   - soilTexture
   - soilDepth
   - soilBulkDensity
   - soilPH
   - soilOrganicCarbon
2. **Location**
   - country
   - city
   - latitude
   - longitude
3. **Time Statement**
   - startTime
   - endTime

### Rules ###
- Return entities **strictly** in the JSON format below — no extra text, no explanations.
- Each entity must include:
   - `"value"` — the exact string from the text.
   - `"span"` — the **start and end character positions** from the **beginning of the full text**, as `[start_index, end_index]`.
- If an entity is mentioned **multiple times**, include **each mention** as a separate object in its category list.
- For compound names like "winter wheat", annotate only the species name (e.g., `"wheat"`).
- Do **not infer** — extract **only** what is **explicitly** stated in the text.
- If no entity is found for a category, return an empty list.
- Use the keys exactly as listed (e.g., `"soilPH"`, `"latitude"`).

### Output Format ###
```json
{
  "Soil": [
    {"soilTexture": { "value": "", "span": [start_index, end_index] }},
    {"soilDepth": { "value": "", "span": [start_index, end_index] }},
    {"soilBulkDensity": { "value": "", "span": [start_index, end_index] }},
    {"soilPH": { "value": "", "span": [start_index, end_index] }},
    {"soilOrganicCarbon": { "value": "", "span": [start_index, end_index] }}
  ],
  "Location": [
    {"country": { "value": "", "span": [start_index, end_index] }},
    {"city": { "value": "", "span": [start_index, end_index] }},
    {"latitude": { "value": "", "span": [start_index, end_index] }},
    {"longitude": { "value": "", "span": [start_index, end_index] }}
  ],
  "Time Statement": [
    {"startTime": { "value": "", "span": [start_index, end_index] }},
    {"endTime": { "value": "", "span": [start_index, end_index] }}
  ]
}
```
        """
        
        user_prompt = f"""
        Your task is to fill the above JSON structure based on the input text.
        
        ### Text ###
        {text}
        """
        
        return system_prompt.strip(), user_prompt.strip()
    
    @staticmethod
    def generate_72b_ner_prompts(text: str) -> Tuple[str, str]:
        """Generate prompts for Qwen2.5-72B-Instruct (6 classes)"""
        system_prompt = """
        ### Instruction ###
You are an expert in Named Entity Recognition (NER) for agricultural texts, specializing in identifying entities related to **Crops**, **Soil**, **Location**, and **Time Statements**.
Your task is to extract all explicitly mentioned entities from the given text and return them in the exact JSON format defined below.

### Entity Categories ###
1. **Crops**
   - cropSpecies
   - cropVariety
2. **Soil**
   - soilReferenceGroup
   - soilAvailableNitrogen
3. **Location**
   - region
4. **Time Statement**
   - duration

### Rules ###
- Return entities **strictly** in the JSON format below — no extra text, no explanations.
- Each entity must include:
   - `"value"` — the exact string from the text.
   - `"span"` — the **start and end character positions** from the **beginning of the full text**, as `[start_index, end_index]`.
- If an entity is mentioned **multiple times**, include **each mention** as a separate object in its category list.
- For compound names like "winter wheat", annotate only the species name (e.g., `"wheat"`).
- Do **not infer** — extract **only** what is **explicitly** stated in the text.
- If no entity is found for a category, return an empty list.
- Use the keys exactly as listed (e.g., `"cropSpecies"`, `"region"`).

### Output Format ###
```json
{
  "Crops": [
    {"cropSpecies": { "value": "", "span": [start_index, end_index] }},
    {"cropVariety": { "value": "", "span": [start_index, end_index] }}
  ],
  "Soil": [
    {"soilReferenceGroup": { "value": "", "span": [start_index, end_index] }},
    {"soilAvailableNitrogen": { "value": "", "span": [start_index, end_index] }}
  ],
  "Location": [
    {"region": { "value": "", "span": [start_index, end_index] }}
  ],
  "Time Statement": [
    {"duration": { "value": "", "span": [start_index, end_index] }}
  ]
}
```
        """
        
        user_prompt = f"""
        Your task is to fill the above JSON structure based on the input text.
        
        ### Text ###
        {text}
        """
        
        return system_prompt.strip(), user_prompt.strip()


class HierarchicalQwenNER:
    """Hierarchical NER system using both Qwen2.5 models"""
    
    def __init__(self):
        self.model_32b = None
        self.tokenizer_32b = None
        self.model_72b = None
        self.tokenizer_72b = None
        self.prompt_generator = QwenNERPromptGenerator()
        
        # Define entity classes for each model
        self.model_32b_classes = [
            "soilTexture", "soilDepth", "soilBulkDensity", "soilPH", "soilOrganicCarbon",
            "country", "city", "latitude", "longitude", "startTime", "endTime"
        ]
        
        self.model_72b_classes = [
            "cropSpecies", "cropVariety", "soilReferenceGroup", 
            "soilAvailableNitrogen", "region", "duration"
        ]
    
    def load_model(self, model_path: str) -> Tuple[Any, Any]:
        """Load a single Qwen model with error handling"""
        print(f"Loading model from: {model_path}")
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Then load model with clean config
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Successfully loaded in FP16")
            return model, tokenizer
        except Exception as e:
            print(f"FP16 load failed: {e}")
            print("Trying with 4-bit quantization...")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("Successfully loaded in 4-bit")
            return model, tokenizer
    
    def load_both_models(self, model_32b_path: str, model_72b_path: str):
        """Load both Qwen models"""
        print("Loading 32B model...")
        self.model_32b, self.tokenizer_32b = self.load_model(model_32b_path)
        
        print("Loading 72B model...")
        self.model_72b, self.tokenizer_72b = self.load_model(model_72b_path)
    
    def perform_ner_single_model(self, model: Any, tokenizer: Any, 
                                text: str, system_prompt: str, user_prompt: str, 
                                max_length: int = 1512) -> str:
        """Perform NER using a single model"""
        # Build messages list according to role-based chat format
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenize the prompt
        input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
        attention_mask = torch.ones_like(input_ids)

        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1500,
            temperature=0.7,
            top_p=0.9,)

        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from model response with improved parsing"""
        try:
            # Method 1: Try to parse the entire response as JSON
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                pass
            
            # Method 2: Extract JSON from code blocks
            code_block_patterns = [
                r'``````',
                r'``````',
                r'`(.*?)`'
            ]
            
            for pattern in code_block_patterns:
                matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
                for match in matches:
                    try:
                        cleaned_json = match.strip()
                        if cleaned_json.startswith('{') and cleaned_json.endswith('}'):
                            return json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        continue
            
            # Method 3: Find balanced JSON objects
            json_objects = self._extract_balanced_json(response)
            for json_str in json_objects:
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and self._validate_ner_structure(parsed):
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            # Method 4: Try to clean and extract common formatting issues
            cleaned_response = self._clean_response(response)
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass
                
            print(f"Warning: No valid JSON found in response: {response[:200]}...")
            return self._get_empty_ner_structure()
            
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return self._get_empty_ner_structure()

    def _extract_balanced_json(self, text: str) -> List[str]:
        """Extract balanced JSON objects from text"""
        json_objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Found start of potential JSON
                brace_count = 0
                start = i
                j = i
                
                while j < len(text):
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found balanced JSON
                            potential_json = text[start:j+1]
                            json_objects.append(potential_json)
                            i = j + 1
                            break
                    j += 1
                else:
                    # No closing brace found
                    i += 1
            else:
                i += 1
        
        return json_objects

    def _clean_response(self, response: str) -> str:
        """Clean common formatting issues in JSON responses"""
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            "Here's the JSON:",
            "The extracted entities are:",
            "```"
        ]
        
        suffixes_to_remove = [
            "```",
            "Let me know if you need any clarification!",
            "I hope this helps!"
        ]
        
        cleaned = response.strip()
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove suffixes
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        
        # Find first { and last }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            cleaned = cleaned[first_brace:last_brace+1]
        
        return cleaned

    def _validate_ner_structure(self, parsed_json: Dict) -> bool:
        """Validate that the JSON has the expected NER structure"""
        expected_categories = ["Crops", "Soil", "Location", "Time Statement"]
        
        # Check if it has the main categories
        if not isinstance(parsed_json, dict):
            return False
        
        # Should have at least one of the expected categories
        has_valid_category = any(cat in parsed_json for cat in expected_categories)
        return has_valid_category

    def _get_empty_ner_structure(self) -> Dict:
        """Return empty NER structure as fallback"""
        return {
            "Crops": [],
            "Soil": [],
            "Location": [],
            "Time Statement": []
        }

    def merge_ner_results(self, result_32b: Dict, result_72b: Dict, text: str) -> Dict:
        """Merge results from both models - no overlap resolution needed"""
        merged_result = {
            "Crops": [],
            "Soil": [],
            "Location": [],
            "Time Statement": []
        }
        
        # Simply combine results from both models
        # Since they predict different entity types, conflicts are unlikely
        
        # Add 32B results
        for category, entities in result_32b.items():
            if entities and isinstance(entities, list):
                for entity_dict in entities:
                    if isinstance(entity_dict, dict):
                        # Validate entity has proper structure
                        valid_entity = {}
                        for entity_type, entity_data in entity_dict.items():
                            if (isinstance(entity_data, dict) and 'span' in entity_data and
                                isinstance(entity_data['span'], list) and len(entity_data['span']) >= 2):
                                valid_entity[entity_type] = entity_data
                        
                        if valid_entity:
                            merged_result[category].append(valid_entity)
        
        # Add 72B results
        for category, entities in result_72b.items():
            if entities and isinstance(entities, list):
                for entity_dict in entities:
                    if isinstance(entity_dict, dict):
                        # Validate entity has proper structure
                        valid_entity = {}
                        for entity_type, entity_data in entity_dict.items():
                            if (isinstance(entity_data, dict) and 'span' in entity_data and
                                isinstance(entity_data['span'], list) and len(entity_data['span']) >= 2):
                                valid_entity[entity_type] = entity_data
                        
                        if valid_entity:
                            merged_result[category].append(valid_entity)
        
        return merged_result

    
    def perform_hierarchical_ner(self, text: str, max_length: int = 1512) -> str:
        """Perform hierarchical NER using both models"""
        print("Performing hierarchical NER...")
        
        # Generate prompts for 32B model
        system_prompt_32b, user_prompt_32b = self.prompt_generator.generate_32b_ner_prompts(text)
        
        # Generate prompts for 72B model  
        system_prompt_72b, user_prompt_72b = self.prompt_generator.generate_72b_ner_prompts(text)
        
        # Run NER with 32B model
        print("Running NER with 32B model...")
        response_32b = self.perform_ner_single_model(
            self.model_32b, self.tokenizer_32b, text, 
            system_prompt_32b, user_prompt_32b, max_length
        )
        
        # Run NER with 72B model
        print("Running NER with 72B model...")
        response_72b = self.perform_ner_single_model(
            self.model_72b, self.tokenizer_72b, text,
            system_prompt_72b, user_prompt_72b, max_length
        )
        print(f"response_32b: {response_32b}")
        print(f"response_72b: {response_72b}")
        # Extract JSON from responses
        result_32b = self.extract_json_from_response(response_32b)
        result_72b = self.extract_json_from_response(response_72b)
        
        
        # Merge results
        merged_result = self.merge_ner_results(result_32b, result_72b, text)
        print(f"merged_result: {merged_result}")

        # Format final response similar to original format
        final_response = json.dumps(merged_result, indent=2, ensure_ascii=False)
        
        return final_response


def process_text_files_hierarchical(input_dir: str, ner_system: HierarchicalQwenNER, 
                                   output_dir: str, max_length: int):
    """Process text files using hierarchical NER system"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_text_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, filename.replace(".txt", "_annotated.txt"))

            if os.path.exists(output_text_path):
                print(f"Skipping {filename} (already processed).")
                continue

            with open(input_text_path, "r", encoding="utf-8") as file:
                text = file.read()

            print(f"Processing {filename}...")
            ner_result = ner_system.perform_hierarchical_ner(text, max_length)

            with open(output_text_path, "w", encoding="utf-8") as file:
                file.write(ner_result)

            print(f"Saved: {output_text_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform hierarchical NER using Qwen2.5 models.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input text files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save annotated output files.")
    parser.add_argument("--output_dir_json", required=True, help="Directory to save extracted json output from annotated files.")
    parser.add_argument("--model_name", required=True, help="Name of the model used for annotation.")
    parser.add_argument("--model_32b_path", required=True, help="Path to the Qwen2.5-32B-Instruct model.")
    parser.add_argument("--model_72b_path", required=True, help="Path to the Qwen2.5-72B-Instruct model.")
    parser.add_argument("--max_length", type=int, default=1512, help="Maximum length for tokenized input.")

    args = parser.parse_args()
    
    # Initialize hierarchical NER system
    ner_system = HierarchicalQwenNER()
    
    # Load both models
    ner_system.load_both_models(args.model_32b_path, args.model_72b_path)
    
    # Process text files
    process_text_files_hierarchical(args.input_dir, ner_system, args.output_dir, args.max_length)
    
    # Run evaluation
    xmi_dir = "/home/s27mhusa_hpc/Master-Thesis/XMI_Files"
    evaluate_all(args.model_name,
        args.input_dir,
        args.output_dir,
        args.output_dir_json,
        0,
        xmi_dir
    )

    print("Hierarchical NER processing complete.")


if __name__ == "__main__":
    main()