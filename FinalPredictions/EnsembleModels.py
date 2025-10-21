import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForCausalLM,
    AutoConfig
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import json
import os
import requests
import faiss
from sentence_transformers import SentenceTransformer
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from seqeval.scheme import IOB2

class MultiModelNER:
    """
    Multi-Model Named Entity Recognition system that uses different specialized models
    for different entity types with model-specific label lists.
    """
    
    def __init__(self, global_label_list: List[str]):
        self.global_label_list = global_label_list
        self.global_id2label = {i: label for i, label in enumerate(global_label_list)}
        self.global_label2id = {label: i for i, label in enumerate(global_label_list)}
        
        # Model-specific label lists
        self.model_label_lists = {
            'roberta_all_specific': [
                "O", "B-soilReferenceGroup", "I-soilReferenceGroup", 
                "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture",
                "B-startTime", "I-startTime", "B-endTime", "I-endTime", 
                "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies",
                "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth",
                "B-region", "I-region", "B-country", "I-country", 
                "B-longitude", "I-longitude", "B-latitude", "I-latitude",
                "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", 
                "B-soilBulkDensity", "I-soilBulkDensity"
            ],
            'agribert_all_specific': [
                "O", "B-soilReferenceGroup", "I-soilReferenceGroup", 
                "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture",
                "B-startTime", "I-startTime", "B-endTime", "I-endTime", 
                "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies",
                "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth",
                "B-region", "I-region", "B-country", "I-country", 
                "B-longitude", "I-longitude", "B-latitude", "I-latitude",
                "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", 
                "B-soilBulkDensity", "I-soilBulkDensity"
            ],
            'roberta_english_specific': [
                "O", "B-soilReferenceGroup", "I-soilReferenceGroup", 
                "B-soilOrganicCarbon", "I-soilOrganicCarbon", "B-soilTexture", "I-soilTexture",
                "B-startTime", "I-startTime", "B-endTime", "I-endTime", 
                "B-city", "I-city", "B-duration", "I-duration", "B-cropSpecies", "I-cropSpecies",
                "B-soilAvailableNitrogen", "I-soilAvailableNitrogen", "B-soilDepth", "I-soilDepth",
                "B-region", "I-region", "B-country", "I-country", 
                "B-longitude", "I-longitude", "B-latitude", "I-latitude",
                "B-cropVariety", "I-cropVariety", "B-soilPH", "I-soilPH", 
                "B-soilBulkDensity", "I-soilBulkDensity"
            ],
            'xlm_roberta_broad': [
                "O", "B-Crop", "I-Crop", "B-TimeStatement", "I-TimeStatement",
                "B-Location", "I-Location"
            ]
        }
        
        # Entity mapping for XLM RoBERTa Broad model
        self.entity_mapping = {
            'xlm_roberta_broad': {
                'Location': 'locationName',  # Map Location to locationName
                'Crop': 'cropSpecies',
                'TimeStatement': 'startTime'  # Can be mapped as needed
            }
        }
        
        # Model configurations with updated entity names
        self.model_configs = {
            'roberta_all_specific': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_specific_22',
                'type': 'token_classification',
                'entities': ['cropSpecies', 'soilReferenceGroup', 'endTime', 'duration','startTime', 'soilOrganicCarbon'],
                'languages': ['en', 'de']
            },
            'qwen_2.5_7b': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/Qwen2.5-7B-Instruct',
                'type': 'causal_lm',
                'entities': ['cropVariety', 'soilAvailableNitrogen', 'soilBulkDensity'],
                'languages': ['en', 'de']
            },
            'xlm_roberta_broad': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_nosoil_29',
                'type': 'token_classification',
                'entities': ['locationName'],  # Uses Location internally, mapped to locationName
                'languages': ['en', 'de']
            },
            'roberta_english_specific': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-english_specific_final_model_regularized_saved_broad_3.0-21sept',
                'type': 'token_classification',
                'entities': ['startTime', 'soilOrganicCarbon'],
                'languages': ['en']
            },
            'agribert_all_specific': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/agribert-new_final_model_regularized_saved_specific_22-3',
                'type': 'token_classification',
                'entities': ['soilDepth', 'soilPH'],
                'languages': ['en', 'de']
            },
            'deepseek': {
                'path': None,  # API-based
                'type': 'api',
                'entities': ['soilTexture'],
                'languages': ['en', 'de']
            }
        }
        
        # Language-specific overrides
        self.language_overrides = {
            'startTime': {
                'de': 'roberta_all_specific'
            },
            'soilOrganicCarbon': {
                'de': 'roberta_all_specific'
            }
        }
        
        self.models = {}
        self.tokenizers = {}
        self.model_id2label = {}  # Store model-specific id2label mappings
        self.model_label2id = {}  # Store model-specific label2id mappings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FAISS index and corpus for retrieval-based few-shot learning
        self.embedder = None
        self.faiss_index = None
        self.corpus_texts = []
        self.corpus_entities = []
        
        # Updated dataset paths
        self.embeddings_dataset = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/ner_dataset_input_output.jsonl"
        self.test_dataset_en_path = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/Test_NER_dataset_English.json"
        self.test_dataset_de_path = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/Test_NER_dataset_German.json"
    
    def get_model_labels(self, model_name: str) -> Tuple[Dict, Dict]:
        """
        Get model-specific label mappings.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (id2label, label2id) dictionaries
        """
        if model_name in self.model_label_lists:
            labels = self.model_label_lists[model_name]
            id2label = {i: label for i, label in enumerate(labels)}
            label2id = {label: i for i, label in enumerate(labels)}
            return id2label, label2id
        else:
            # Default to global labels
            return self.global_id2label, self.global_label2id
    
    def map_entity_name(self, model_name: str, entity_name: str) -> str:
        """
        Map entity name from model-specific to global entity name.
        
        Args:
            model_name: Name of the model
            entity_name: Entity name from model prediction
            
        Returns:
            Mapped entity name
        """
        if model_name in self.entity_mapping:
            # Extract entity type from BIO tag
            if entity_name.startswith('B-') or entity_name.startswith('I-'):
                prefix = entity_name[:2]
                entity_type = entity_name[2:]
                
                # Map entity type if mapping exists
                if entity_type in self.entity_mapping[model_name]:
                    mapped_type = self.entity_mapping[model_name][entity_type]
                    return f"{prefix}{mapped_type}"
        
        return entity_name
    
    def load_model_token_classification(self, model_path: str, model_name: str):
        """Load standard token classification model with model-specific labels."""
        print(f"Loading token classification model: {model_name}")
        
        # Get model-specific labels
        id2label, label2id = self.get_model_labels(model_name)
        
        # Store for later use
        self.model_id2label[model_name] = id2label
        self.model_label2id[model_name] = label2id
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with custom label mapping
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id
        )
        
        model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_model_causal_lm(self, model_path: str, model_name: str):
        """Load causal LM model (Qwen/Llama)."""
        print(f"Loading causal LM model from: {model_path}")
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Then load model with clean config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"Successfully loaded {model_name} in FP16")
        return model, tokenizer
    
    def load_models(self):
        """Load all required models and tokenizers."""
        print("Loading models...")
        for model_name, config in self.model_configs.items():
            try:
                if config['type'] == 'api':
                    print(f"Skipping {model_name} - API-based model")
                    continue
                elif config['type'] == 'causal_lm':
                    model, tokenizer = self.load_model_causal_lm(config['path'], model_name)
                elif config['type'] == 'token_classification':
                    model, tokenizer = self.load_model_token_classification(config['path'], model_name)
                else:
                    raise ValueError(f"Unknown model type: {config['type']}")
                
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                
                print(f"✓ {model_name} loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {str(e)}")
    
    def load_hf_dataset(self, dataset_path: str) -> Dataset:
        """Load Hugging Face dataset from JSON file."""
        print(f"Loading dataset from: {dataset_path}")
        dataset = Dataset.from_json(dataset_path)
        print(f"✓ Loaded {len(dataset)} examples")
        return dataset
    
    def convert_hf_dataset_to_texts_entities(self, dataset: Dataset) -> Tuple[List[str], List[List]]:
        """Convert Hugging Face Dataset to lists of texts and entities."""
        texts = []
        entities_list = []
        
        for example in dataset:
            # Extract text
            if 'tokens' in example:
                text = ' '.join(example['tokens'])
            elif 'text' in example:
                text = example['text']
            else:
                if 'input' in example:
                    input_field = example['input']
                    text = input_field.split("\n### Text ###\n", 1)[-1] if "\n### Text ###\n" in input_field else input_field
                else:
                    print(f"Warning: Could not find text field in example: {example.keys()}")
                    continue
            
            # Extract entities
            entities = []
            if 'ner_tags' in example:
                tokens = example['tokens'] if 'tokens' in example else text.split()
                ner_tags = example['ner_tags']
                
                current_entity = None
                start_idx = 0
                
                for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
                    if isinstance(tag, int):
                        tag = self.global_id2label.get(tag, 'O')
                    
                    if tag.startswith('B-'):
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            'text': token,
                            'label': tag[2:],
                            'start': start_idx,
                            'end': start_idx + len(token)
                        }
                    elif tag.startswith('I-') and current_entity:
                        current_entity['text'] += ' ' + token
                        current_entity['end'] = start_idx + len(token)
                    else:
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None
                    
                    start_idx += len(token) + 1
                
                if current_entity:
                    entities.append(current_entity)
            
            elif 'entities' in example:
                entities = example['entities']
            elif 'output' in example:
                output_field = example['output']
                try:
                    entities = json.loads(output_field) if isinstance(output_field, str) else output_field
                    if isinstance(entities, dict) and 'entities' in entities:
                        entities = entities['entities']
                except Exception:
                    entities = []
            
            texts.append(text)
            entities_list.append(entities)
        
        return texts, entities_list
    
    def build_or_load_index(self, embeddings: np.ndarray, index_path: str = None):
        """Build FAISS index from embeddings."""
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        if index_path:
            faiss.write_index(index, index_path)
        return index
    
    def embed_texts(self, texts: List[str], batch_size: int = 64):
        """Embed texts using sentence transformer."""
        if self.embedder is None:
            print("Loading sentence transformer for embeddings...")
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        embs = self.embedder.encode(
            texts, 
            convert_to_numpy=True, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        return embs
    
    def get_similar_examples(self, query_text: str, top_k: int = 5):
        """Retrieve similar examples using FAISS."""
        if self.faiss_index is None or self.embedder is None:
            raise ValueError("FAISS index not initialized. Call setup_retrieval() first.")
        
        q = self.embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.faiss_index.search(q, top_k)
        idxs = I[0].tolist()
        examples = [(self.corpus_texts[i], self.corpus_entities[i]) for i in idxs]
        return examples, (D[0].tolist(), idxs)
    
    def setup_retrieval(self, dataset_path: str = None, index_path: str = None):
        """Setup FAISS-based retrieval system."""
        print("Setting up retrieval system...")
        
        if dataset_path is None:
            dataset_path = self.embeddings_dataset
        
        dataset = self.load_hf_dataset(dataset_path)
        texts, entities = self.convert_hf_dataset_to_texts_entities(dataset)
        self.corpus_texts = texts
        self.corpus_entities = entities
        
        if index_path and os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}")
            self.faiss_index = faiss.read_index(index_path)
            if self.embedder is None:
                self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            print("Embedding corpus texts...")
            embeddings = self.embed_texts(texts)
            print("Building FAISS index...")
            self.faiss_index = self.build_or_load_index(embeddings, index_path)
        
        print(f"✓ Retrieval system ready with {len(self.corpus_texts)} examples")
    
    def generate_ner_prompts(self, text: str, entity_types: List[str] = None):
        """Generate system and user prompts for LLM-based NER."""
        if entity_types is None:
            entity_types = []
        
        entity_descriptions = {
            'cropVariety': 'Specific cultivar/variety name (e.g., "Golden Delicious")',
            'soilAvailableNitrogen': 'Nitrogen is present in a soil sample that is available to plants. Please only annotate explicit mentions of the available nitrogen. Make sure it is related to the nitrogen in the soil and not in fertilizers, etc',
            'soilBulkDensity': 'The dry weight of soil divided by its volume. Please annotate the term “bulk density” if it is mentioned in a text: ',
            'soilTexture': 'Soil texture measures the proportion of sand, silt, and clay-sized particles in a soil sample. Please annotate a soil texture if it is part of a soil texture classification such as the USDA Soil Texture Classification, consisting of 12 different soil textures or the soil textures of the Bodenkundliche Kartieranleitung',
        }
        
        if entity_types:
            entity_list = "\n".join([f"- {et}: {entity_descriptions.get(et, 'Agricultural entity')}" 
                                    for et in entity_types])
        else:
            entity_list = "\n".join([f"- {et}: {desc}" for et, desc in entity_descriptions.items()])
        
        system_prompt = f"""You are an expert in agricultural Named Entity Recognition (NER).
Your task is to identify and extract specific entities from agricultural text.

Entity Types to Extract:
{entity_list}

Instructions:
1. Read the input text carefully
2. Identify all entities that match the specified types
3. Return the results as a JSON object with the following structure:
{{
    "entities": [
        {{"text": "entity text", "label": "entity_type", "start": start_index, "end": end_index}},
        ...
    ]
}}

Few-shot Examples:
"""
        
        user_prompt = f"Extract entities from the following text:\n\n{text}"
        return system_prompt, user_prompt
    
    def perform_ner_deepseek(self, text: str, max_length: int = 512, examples: List = None):
        """Perform NER using DeepSeek API."""
        if examples is None:
            if self.faiss_index is not None:
                examples, _ = self.get_similar_examples(text, top_k=5)
            else:
                examples = []
        
        system_prompt, user_prompt = self.generate_ner_prompts(text)
        
        blocks = []
        for i, (example_text, ents) in enumerate(examples):
            blocks.append(f"### Example {i} ###:\nInput Text:\n {example_text}\nOutput: {json.dumps(ents, ensure_ascii=False)}")
        blocks_str = "\n\n".join(blocks)
        system_prompt += blocks_str
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "provider": {"sort": "price"},
                    "temperature": 0.7,
                    "max_tokens": max_length,
                    "top_p": 0.90,
                })
            )
            return response.json()
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return {"entities": []}
    
    def perform_ner_llama(self, model, tokenizer, text: str, max_length: int = 1500, examples: List = None):
        """Perform NER using Llama model with chat template."""
        if examples is None:
            if self.faiss_index is not None:
                examples, _ = self.get_similar_examples(text, top_k=5)
            else:
                examples = []
        
        system_prompt, user_prompt = self.generate_ner_prompts(text)
        
        blocks = []
        for i, (example_text, ents) in enumerate(examples):
            blocks.append(f"### Example {i} ###:\nInput Text:\n {example_text}\nOutput: {json.dumps(ents, ensure_ascii=False)}")
        blocks_str = "\n\n".join(blocks)
        system_prompt += blocks_str
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        device = model.device
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids)
        
        output_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
        )
        
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def perform_ner_causal_lm(self, text: str, tokens: List[str], model_name: str, 
                             target_entity: str, max_length: int = 1500) -> List[str]:
        """Perform NER using causal LM model."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        examples = None
        if self.faiss_index is not None:
            examples, _ = self.get_similar_examples(text, top_k=5)
        
        response = self.perform_ner_llama(model, tokenizer, text, max_length, examples)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                entities = result.get('entities', [])
            else:
                entities = []
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response from {model_name}")
            entities = []
        
        bio_tags = ['O'] * len(tokens)
        
        for entity in entities:
            if target_entity.lower() not in entity.get('label', '').lower():
                continue
            
            entity_text = entity.get('text', '')
            if not entity_text:
                continue
            
            entity_words = entity_text.split()
            for i in range(len(tokens) - len(entity_words) + 1):
                if tokens[i:i+len(entity_words)] == entity_words or \
                   ' '.join(tokens[i:i+len(entity_words)]).lower() == entity_text.lower():
                    bio_tags[i] = f"B-{target_entity}"
                    for j in range(i+1, i+len(entity_words)):
                        bio_tags[j] = f"I-{target_entity}"
                    break
        
        return bio_tags
    
    def get_model_for_entity(self, entity_type: str, language: str = 'en') -> Tuple[str, str, object, object]:
        """Get the appropriate model for a given entity type and language."""
        if entity_type in self.language_overrides:
            if language in self.language_overrides[entity_type]:
                override_model = self.language_overrides[entity_type][language]
                config = self.model_configs[override_model]
                return (override_model, config['type'], 
                       self.models.get(override_model), 
                       self.tokenizers.get(override_model))
        
        for model_name, config in self.model_configs.items():
            if entity_type in config['entities'] and language in config['languages']:
                return (model_name, config['type'],
                       self.models.get(model_name), 
                       self.tokenizers.get(model_name))
        
        raise ValueError(f"No model found for entity type '{entity_type}' and language '{language}'")
    
    def predict_entity_with_model(self, text: str, tokens: List[str], model_name: str, 
                                  model_type: str, target_entity: str) -> List[str]:
        """Predict entities using a specific model."""
        if model_type == 'api':
            response = self.perform_ner_deepseek(text)
            
            bio_tags = ['O'] * len(tokens)
            try:
                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                        entities = result.get('entities', [])
                        
                        for entity in entities:
                            if target_entity.lower() not in entity.get('label', '').lower():
                                continue
                            
                            entity_text = entity.get('text', '')
                            entity_words = entity_text.split()
                            
                            for i in range(len(tokens) - len(entity_words) + 1):
                                if ' '.join(tokens[i:i+len(entity_words)]).lower() == entity_text.lower():
                                    bio_tags[i] = f"B-{target_entity}"
                                    for j in range(i+1, i+len(entity_words)):
                                        bio_tags[j] = f"I-{target_entity}"
                                    break
            except Exception as e:
                print(f"Error parsing DeepSeek response: {str(e)}")
            
            return bio_tags
        
        elif model_type == 'causal_lm':
            return self.perform_ner_causal_lm(text, tokens, model_name, target_entity)
        
        elif model_type == 'token_classification':
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Get model-specific id2label mapping
            id2label = self.model_id2label.get(model_name, self.global_id2label)
            
            encoding = tokenizer(tokens, is_split_into_words=True, 
                               return_tensors='pt', truncation=True, 
                               padding=True, return_offsets_mapping=True)
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
            
            # Map entity names if needed (e.g., Location -> locationName)
            predicted_labels = [self.map_entity_name(model_name, label) for label in predicted_labels]
            
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            previous_word_id = None
            
            for word_id, label in zip(word_ids, predicted_labels):
                if word_id is None:
                    continue
                if word_id != previous_word_id:
                    if target_entity in label or label == 'O':
                        aligned_labels.append(label)
                    else:
                        aligned_labels.append('O')
                    previous_word_id = word_id
            
            while len(aligned_labels) < len(tokens):
                aligned_labels.append('O')
            aligned_labels = aligned_labels[:len(tokens)]
            
            return aligned_labels
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def merge_predictions(self, all_predictions: Dict[str, List[str]], tokens: List[str]) -> List[str]:
        """Merge predictions from different models."""
        final_labels = ['O'] * len(tokens)
        
        for entity_type, predictions in all_predictions.items():
            for i, pred in enumerate(predictions):
                if i >= len(final_labels):
                    break
                if pred != 'O':
                    if final_labels[i] == 'O':
                        final_labels[i] = pred
        
        return final_labels
    
    def predict(self, text: str, tokens: List[str] = None, language: str = 'en') -> Dict:
        """Perform multi-model NER prediction."""
        if tokens is None:
            tokens = text.split()
        
        all_predictions = {}
        entity_types = set()
        
        for config in self.model_configs.values():
            entity_types.update(config['entities'])
        
        for entity_type in entity_types:
            try:
                model_name, model_type, model, tokenizer = self.get_model_for_entity(entity_type, language)
                print(f"Processing {entity_type} with {model_name} ({model_type})")
                
                predictions = self.predict_entity_with_model(
                    text, tokens, model_name, model_type, entity_type
                )
                all_predictions[entity_type] = predictions
            except ValueError as e:
                print(f"Warning: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing {entity_type}: {str(e)}")
                continue
        
        final_predictions = self.merge_predictions(all_predictions, tokens)
        
        return {
            'tokens': tokens,
            'labels': final_predictions,
            'entity_predictions': all_predictions
        }
    
    def predict_dataset(self, dataset: Dataset, language: str = 'en') -> List[Dict]:
        """Perform predictions on a Hugging Face Dataset."""
        results = []
        
        for i, example in enumerate(dataset):
            print(f"\nProcessing example {i+1}/{len(dataset)}")
            
            if 'tokens' in example:
                tokens = example['tokens']
                text = ' '.join(tokens)
            elif 'text' in example:
                text = example['text']
                tokens = None
            else:
                print(f"Warning: Could not find text/tokens in example {i}")
                continue
            
            result = self.predict(text, tokens=tokens, language=language)
            
            result['example_id'] = i
            if 'ner_tags' in example:
                gold_tags = []
                for tag in example['ner_tags']:
                    if isinstance(tag, int):
                        gold_tags.append(self.global_id2label.get(tag, 'O'))
                    else:
                        gold_tags.append(tag)
                result['gold_labels'] = gold_tags
            
            results.append(result)
        
        return results
    
    def calculate_f1_scores(self, predictions: List[Dict], output_file: str = None) -> Dict:
        """Calculate F1 scores using seqeval."""
        print("\n" + "="*60)
        print("CALCULATING F1 SCORES")
        print("="*60)
        
        y_true = []
        y_pred = []
        
        for pred in predictions:
            if 'gold_labels' in pred and 'labels' in pred:
                y_true.append(pred['gold_labels'])
                y_pred.append(pred['labels'])
        
        if not y_true or not y_pred:
            print("Error: No valid predictions with gold labels found!")
            return {}
        
        precision = precision_score(y_true, y_pred, mode='strict', scheme=IOB2)
        recall = recall_score(y_true, y_pred, mode='strict', scheme=IOB2)
        f1 = f1_score(y_true, y_pred, mode='strict', scheme=IOB2)
        accuracy = accuracy_score(y_true, y_pred)
        
        report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2, digits=4)
        
        print("\n" + "="*60)
        print("OVERALL METRICS")
        print("="*60)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("OVERALL METRICS\n")
                f.write("="*60 + "\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall:    {recall:.4f}\n")
                f.write(f"F1 Score:  {f1:.4f}\n")
                f.write(f"Accuracy:  {accuracy:.4f}\n\n")
                f.write("="*60 + "\n")
                f.write("DETAILED CLASSIFICATION REPORT\n")
                f.write("="*60 + "\n")
                f.write(report)
            print(f"\nReport saved to: {output_file}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions in CoNLL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                tokens = pred['tokens']
                labels = pred['labels']
                gold_labels = pred.get('gold_labels', [])
                
                for j, token in enumerate(tokens):
                    label = labels[j] if j < len(labels) else 'O'
                    
                    if gold_labels:
                        gold = gold_labels[j] if j < len(gold_labels) else 'O'
                        f.write(f"{token}\t{gold}\t{label}\n")
                    else:
                        f.write(f"{token}\t{label}\n")
                
                f.write("\n")
        
        print(f"Predictions saved to {output_path}")


# Usage Example
def main():
    # Global label list for dataset
    global_label_list = ["O","B-cropSpecies","I-cropSpecies","B-cropVariety","I-cropVariety","B-locationName","I-locationName","B-startTime","I-startTime","B-endTime","I-endTime","B-duration","I-duration","B-soilDepth","I-soilDepth","B-soilReferenceGroup","I-soilReferenceGroup","B-soilOrganicCarbon","I-soilOrganicCarbon","B-soilTexture","I-soilTexture","B-soilBulkDensity","I-soilBulkDensity","B-soilAvailableNitrogen","I-soilAvailableNitrogen","B-soilPH","I-soilPH"]

    
    # Initialize multi-model NER
    ner = MultiModelNER(global_label_list)
    
    # Setup retrieval
    print("\n" + "="*60)
    print("SETTING UP RETRIEVAL SYSTEM")
    print("="*60)
    ner.setup_retrieval(index_path="faiss_index_specific.bin")
    
    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    ner.load_models()
    
    # Load and predict on German test dataset
    print("\n" + "="*60)
    print("GERMAN TEST DATASET EVALUATION")
    print("="*60)
    test_dataset_de = ner.load_hf_dataset(ner.test_dataset_de_path)
    de_results = ner.predict_dataset(test_dataset_de, language='de')
    ner.save_predictions(de_results, 'predictions_german_multimodel.conll')
    metrics_de = ner.calculate_f1_scores(de_results, output_file='f1_scores_german_multimodel.txt')
    
    # Load and predict on English test dataset
    print("\n" + "="*60)
    print("ENGLISH TEST DATASET EVALUATION")
    print("="*60)
    test_dataset_en = ner.load_hf_dataset(ner.test_dataset_en_path)
    en_results = ner.predict_dataset(test_dataset_en, language='en')
    ner.save_predictions(en_results, 'predictions_english_multimodel.conll')
    metrics_en = ner.calculate_f1_scores(en_results, output_file='f1_scores_english_multimodel.txt')
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"German F1 Score:  {metrics_de.get('f1_score', 0):.4f}")
    print(f"English F1 Score: {metrics_en.get('f1_score', 0):.4f}")


if __name__ == "__main__":
    main()
