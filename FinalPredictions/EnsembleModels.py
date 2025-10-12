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

class MultiModelNER:
    """
    Multi-Model Named Entity Recognition system that uses different specialized models
    for different entity types, including LLM-based models (Qwen, Llama) and API-based models (DeepSeek).
    """
    
    def __init__(self, label_list: List[str]):
        self.label_list = label_list
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}
        
        # Model configurations
        self.model_configs = {
            'roberta_all_specific': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_specific_22',
                'type': 'token_classification',
                'entities': ['cropSpecies', 'soilReferenceGroup', 'endTime', 'duration'],
                'languages': ['en', 'de']
            },
            'qwen_2.5_7b': {
                'path': 'Qwen/Qwen2.5-7B-Instruct',
                'type': 'causal_lm',
                'entities': ['cropVariety', 'soilAvailableNitrogen', 'soilBulkDensity'],
                'languages': ['en', 'de']
            },
            'xlm_roberta_broad': {
                'path': '/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/roberta-en-de_final_model_regularized_saved_nosoil_29',
                'type': 'token_classification',
                'entities': ['locationName'],
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FAISS index and corpus for retrieval-based few-shot learning
        self.embedder = None
        self.faiss_index = None
        self.corpus_texts = []
        self.corpus_entities = []
        
        # Dataset paths
        self.test_dataset_en_path = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/Test_NER_dataset_English_Specific.json"
        self.train_dataset_de_path = "/home/s27mhusa_hpc/Master-Thesis/Dataset19September/NER_dataset_sentence_train_final.json"
    
    def load_model_token_classification(self, model_path: str, model_name: str):
        """Load standard token classification model."""
        print(f"Loading token classification model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, tokenizer
    
    def load_model_causal_lm(self, model_path: str, model_name: str):
        """Load causal LM model (Qwen/Llama) using your specific method."""
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
        """
        Load Hugging Face dataset from JSON file.
        
        Args:
            dataset_path: Path to JSON dataset file
            
        Returns:
            Hugging Face Dataset object
        """
        print(f"Loading dataset from: {dataset_path}")
        dataset = Dataset.from_json(dataset_path)
        print(f"✓ Loaded {len(dataset)} examples")
        return dataset
    
    def convert_hf_dataset_to_texts_entities(self, dataset: Dataset) -> Tuple[List[str], List[List]]:
        """
        Convert Hugging Face Dataset to lists of texts and entities.
        
        Args:
            dataset: Hugging Face Dataset object
            
        Returns:
            Tuple of (texts, entities)
        """
        texts = []
        entities_list = []
        
        for example in dataset:
            # Extract text - handle different possible field names
            if 'tokens' in example:
                text = ' '.join(example['tokens'])
            elif 'text' in example:
                text = example['text']
            else:
                # Check if there's an 'input' field that might contain the text
                if 'input' in example:
                    input_field = example['input']
                    text = input_field.split("\n### Text ###\n", 1)[-1] if "\n### Text ###\n" in input_field else input_field
                else:
                    print(f"Warning: Could not find text field in example: {example.keys()}")
                    continue
            
            # Extract entities - handle different formats
            entities = []
            if 'ner_tags' in example:
                # Convert BIO tags to entity format
                tokens = example['tokens'] if 'tokens' in example else text.split()
                ner_tags = example['ner_tags']
                
                current_entity = None
                start_idx = 0
                
                for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
                    if isinstance(tag, int):
                        tag = self.id2label.get(tag, 'O')
                    
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
                    
                    start_idx += len(token) + 1  # +1 for space
                
                if current_entity:
                    entities.append(current_entity)
            
            elif 'entities' in example:
                entities = example['entities']
            elif 'output' in example:
                # Parse output field
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
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            index_path: Optional path to save the index
            
        Returns:
            FAISS index
        """
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        if index_path:
            faiss.write_index(index, index_path)
        return index
    
    def embed_texts(self, texts: List[str], batch_size: int = 64):
        """
        Embed texts using sentence transformer.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding
            
        Returns:
            Numpy array of embeddings
        """
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
        """
        Retrieve similar examples using FAISS.
        
        Args:
            query_text: Query text
            top_k: Number of similar examples to retrieve
            
        Returns:
            Tuple of (examples, (distances, indices))
        """
        if self.faiss_index is None or self.embedder is None:
            raise ValueError("FAISS index not initialized. Call setup_retrieval() first.")
        
        q = self.embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.faiss_index.search(q, top_k)
        idxs = I[0].tolist()
        examples = [(self.corpus_texts[i], self.corpus_entities[i]) for i in idxs]
        return examples, (D[0].tolist(), idxs)
    
    def setup_retrieval(self, dataset_path: str = None, index_path: str = None, use_german: bool = False):
        """
        Setup FAISS-based retrieval system using Hugging Face datasets.
        
        Args:
            dataset_path: Path to training dataset (JSON format). If None, uses default German dataset
            index_path: Optional path to save/load FAISS index
            use_german: If True, use German dataset, else use English
        """
        print("Setting up retrieval system...")
        
        # Use default path if not provided
        if dataset_path is None:
            dataset_path = self.train_dataset_de_path if use_german else self.test_dataset_en_path
        
        # Load Hugging Face dataset
        dataset = self.load_hf_dataset(dataset_path)
        
        # Convert to texts and entities
        texts, entities = self.convert_hf_dataset_to_texts_entities(dataset)
        self.corpus_texts = texts
        self.corpus_entities = entities
        
        # Check if index already exists
        if index_path and os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}")
            self.faiss_index = faiss.read_index(index_path)
            # Still need to load embedder
            if self.embedder is None:
                self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            # Embed texts
            print("Embedding corpus texts...")
            embeddings = self.embed_texts(texts)
            
            # Build index
            print("Building FAISS index...")
            self.faiss_index = self.build_or_load_index(embeddings, index_path)
        
        print(f"✓ Retrieval system ready with {len(self.corpus_texts)} examples")
    
    def generate_ner_prompts(self, text: str, entity_types: List[str] = None):
        """
        Generate system and user prompts for LLM-based NER.
        
        Args:
            text: Input text
            entity_types: List of entity types to extract (optional)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if entity_types is None:
            entity_types = []
        
        entity_descriptions = {
            'cropVariety': 'Specific variety or cultivar name of a crop (e.g., "Winter Wheat", "Golden Harvest")',
            'soilAvailableNitrogen': 'Measurements or values related to nitrogen availability in soil',
            'soilBulkDensity': 'Measurements of soil bulk density, typically in g/cm³ or kg/m³',
            'soilTexture': 'Description of soil texture (e.g., "sandy loam", "clay", "silt")',
            'cropSpecies': 'Type of crop species (e.g., "Wheat", "Corn", "Rice")',
            'locationName': 'Geographic location or place name',
            'startTime': 'Start time or date of an event or period',
            'endTime': 'End time or date of an event or period',
            'soilDepth': 'Depth measurement of soil',
            'soilPH': 'pH value of soil',
            'soilOrganicCarbon': 'Organic carbon content in soil',
            'soilReferenceGroup': 'Soil classification or reference group'
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
        """
        Perform NER using DeepSeek API.
        
        Args:
            text: Input text
            max_length: Maximum token length for response
            examples: Few-shot examples
            
        Returns:
            API response dictionary
        """
        if examples is None:
            # Use retrieval to get similar examples
            if self.faiss_index is not None:
                examples, _ = self.get_similar_examples(text, top_k=3)
            else:
                examples = []
        
        system_prompt, user_prompt = self.generate_ner_prompts(text)
        
        # Add few-shot examples
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
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    "provider": {
                        "sort": "price"
                    },
                    "temperature": 0.7,
                    "max_tokens": max_length,
                    "top_p": 0.90,
                })
            )
            return response.json()
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            return {"entities": []}
    
    def perform_ner_llama(self, model, tokenizer, text: str, max_length: int = 1500, 
                         examples: List = None):
        """
        Perform NER using Llama model with chat template.
        
        Args:
            model: Llama model
            tokenizer: Llama tokenizer
            text: Input text
            max_length: Maximum new tokens to generate
            examples: Few-shot examples
            
        Returns:
            Generated response text
        """
        if examples is None:
            # Use retrieval to get similar examples
            if self.faiss_index is not None:
                examples, _ = self.get_similar_examples(text, top_k=5)
            else:
                examples = []
        
        system_prompt, user_prompt = self.generate_ner_prompts(text)
        
        # Add few-shot examples
        blocks = []
        for i, (example_text, ents) in enumerate(examples):
            blocks.append(f"### Example {i} ###:\nInput Text:\n {example_text}\nOutput: {json.dumps(ents, ensure_ascii=False)}")
        blocks_str = "\n\n".join(blocks)
        system_prompt += blocks_str
        
        # Build messages list according to role-based chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        device = model.device
        
        # Tokenize the prompt using chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate response
        output_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Decode response
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def perform_ner_causal_lm(self, text: str, tokens: List[str], model_name: str, 
                             target_entity: str, max_length: int = 1500) -> List[str]:
        """
        Perform NER using causal LM model (Qwen/Llama) with prompting.
        
        Args:
            text: Input text
            tokens: Pre-tokenized tokens
            model_name: Name of the model to use
            target_entity: Entity type to extract
            max_length: Maximum new tokens to generate
            
        Returns:
            List of BIO tags for each token
        """
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Get similar examples using retrieval if available
        examples = None
        if self.faiss_index is not None:
            examples, _ = self.get_similar_examples(text, top_k=5)
        
        # Use Llama-style inference
        response = self.perform_ner_llama(model, tokenizer, text, max_length, examples)
        
        # Parse JSON response to extract entities
        try:
            # Extract JSON from response
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
        
        # Convert entities to BIO tags
        bio_tags = ['O'] * len(tokens)
        
        for entity in entities:
            if target_entity.lower() not in entity.get('label', '').lower():
                continue
            
            entity_text = entity.get('text', '')
            if not entity_text:
                continue
            
            # Find entity in tokens
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
        """
        Get the appropriate model and tokenizer for a given entity type and language.
        
        Args:
            entity_type: The entity type (e.g., 'cropSpecies')
            language: Language code ('en' or 'de')
            
        Returns:
            Tuple of (model_name, model_type, model, tokenizer)
        """
        # Check for language-specific overrides
        if entity_type in self.language_overrides:
            if language in self.language_overrides[entity_type]:
                override_model = self.language_overrides[entity_type][language]
                config = self.model_configs[override_model]
                return (override_model, config['type'], 
                       self.models.get(override_model), 
                       self.tokenizers.get(override_model))
        
        # Find the model that handles this entity type
        for model_name, config in self.model_configs.items():
            if entity_type in config['entities'] and language in config['languages']:
                return (model_name, config['type'],
                       self.models.get(model_name), 
                       self.tokenizers.get(model_name))
        
        raise ValueError(f"No model found for entity type '{entity_type}' and language '{language}'")
    
    def predict_entity_with_model(self, text: str, tokens: List[str], model_name: str, 
                                  model_type: str, target_entity: str) -> List[str]:
        """
        Predict entities using a specific model, filtering for target entity type.
        
        Args:
            text: Input text
            tokens: Pre-tokenized tokens
            model_name: Name of the model to use
            model_type: Type of model ('token_classification', 'causal_lm', 'api')
            target_entity: Entity type to extract
            
        Returns:
            List of BIO tags for each token
        """
        if model_type == 'api':
            # Handle DeepSeek API
            response = self.perform_ner_deepseek(text)
            
            # Parse response and convert to BIO tags
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
                        
                        # Convert to BIO tags
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
            # Handle Qwen/Llama
            return self.perform_ner_causal_lm(text, tokens, model_name, target_entity)
        
        elif model_type == 'token_classification':
            # Handle standard token classification models
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Tokenize
            encoding = tokenizer(tokens, is_split_into_words=True, 
                               return_tensors='pt', truncation=True, 
                               padding=True, return_offsets_mapping=True)
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Convert predictions to labels
            predicted_labels = [self.id2label[pred.item()] for pred in predictions[0]]
            
            # Align subword tokens back to word tokens
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            previous_word_id = None
            
            for word_id, label in zip(word_ids, predicted_labels):
                if word_id is None:
                    continue
                if word_id != previous_word_id:
                    # Filter: only keep labels for target entity
                    if target_entity in label or label == 'O':
                        aligned_labels.append(label)
                    else:
                        aligned_labels.append('O')
                    previous_word_id = word_id
            
            # Pad or truncate to match original tokens length
            while len(aligned_labels) < len(tokens):
                aligned_labels.append('O')
            aligned_labels = aligned_labels[:len(tokens)]
            
            return aligned_labels
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def merge_predictions(self, all_predictions: Dict[str, List[str]], 
                         tokens: List[str]) -> List[str]:
        """
        Merge predictions from different models into a single prediction.
        
        Args:
            all_predictions: Dictionary mapping entity types to their predictions
            tokens: Original tokens
            
        Returns:
            Final merged predictions
        """
        final_labels = ['O'] * len(tokens)
        
        # Priority order for resolving conflicts
        for entity_type, predictions in all_predictions.items():
            for i, pred in enumerate(predictions):
                if i >= len(final_labels):
                    break
                if pred != 'O':
                    if final_labels[i] == 'O':
                        final_labels[i] = pred
        
        return final_labels
    
    def predict(self, text: str, tokens: List[str] = None, language: str = 'en') -> Dict:
        """
        Perform multi-model NER prediction on input text.
        
        Args:
            text: Input text
            tokens: Pre-tokenized tokens (optional)
            language: Language code ('en' or 'de')
            
        Returns:
            Dictionary with tokens and predicted labels
        """
        # Tokenize if not provided
        if tokens is None:
            tokens = text.split()
        
        all_predictions = {}
        entity_types = set()
        
        # Collect all unique entity types
        for config in self.model_configs.values():
            entity_types.update(config['entities'])
        
        # Get predictions for each entity type
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
        
        # Merge all predictions
        final_predictions = self.merge_predictions(all_predictions, tokens)
        
        return {
            'tokens': tokens,
            'labels': final_predictions,
            'entity_predictions': all_predictions
        }
    
    def predict_dataset(self, dataset: Dataset, language: str = 'en') -> List[Dict]:
        """
        Perform predictions on a Hugging Face Dataset.
        
        Args:
            dataset: Hugging Face Dataset object
            language: Language code ('en' or 'de')
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, example in enumerate(dataset):
            print(f"\nProcessing example {i+1}/{len(dataset)}")
            
            # Extract text and tokens
            if 'tokens' in example:
                tokens = example['tokens']
                text = ' '.join(tokens)
            elif 'text' in example:
                text = example['text']
                tokens = None
            else:
                print(f"Warning: Could not find text/tokens in example {i}")
                continue
            
            # Perform prediction
            result = self.predict(text, tokens=tokens, language=language)
            
            # Add original data to result
            result['example_id'] = i
            if 'ner_tags' in example:
                result['gold_labels'] = example['ner_tags']
            
            results.append(result)
        
        return results
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """
        Save predictions to a file in CoNLL format.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Path to save the predictions
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                tokens = pred['tokens']
                labels = pred['labels']
                gold_labels = pred.get('gold_labels', [])
                
                for j, token in enumerate(tokens):
                    label = labels[j] if j < len(labels) else 'O'
                    
                    # Include gold labels if available
                    if gold_labels:
                        gold = gold_labels[j] if j < len(gold_labels) else 'O'
                        if isinstance(gold, int):
                            gold = self.id2label.get(gold, 'O')
                        f.write(f"{token}\t{gold}\t{label}\n")
                    else:
                        f.write(f"{token}\t{label}\n")
                
                f.write("\n")  # Blank line between sentences
        
        print(f"Predictions saved to {output_path}")


# Usage Example
def main():
    # Define label list
    label_list = [
        "O",
        "B-soilReferenceGroup", "I-soilReferenceGroup",
        "B-soilOrganicCarbon", "I-soilOrganicCarbon",
        "B-soilTexture", "I-soilTexture",
        "B-startTime", "I-startTime",
        "B-endTime", "I-endTime",
        "B-cropSpecies", "I-cropSpecies",
        "B-soilAvailableNitrogen", "I-soilAvailableNitrogen",
        "B-soilDepth", "I-soilDepth",
        "B-locationName", "I-locationName",
        "B-cropVariety", "I-cropVariety",
        "B-soilPH", "I-soilPH",
        "B-soilBulkDensity", "I-soilBulkDensity"
    ]
    
    # Initialize multi-model NER
    ner = MultiModelNER(label_list)
    
    # Setup retrieval system for German dataset
    print("\n=== Setting up retrieval system ===")
    ner.setup_retrieval(use_german=True, index_path="faiss_index_german.bin")
    
    # Load all models
    print("\n=== Loading models ===")
    ner.load_models()
    
    # Load test datasets
    print("\n=== Loading test datasets ===")
    test_dataset_en = ner.load_hf_dataset(ner.test_dataset_en_path)
    test_dataset_de = ner.load_hf_dataset(ner.train_dataset_de_path)
    
    # Predict on English test dataset
    print("\n=== Predicting on English test set ===")
    en_results = ner.predict_dataset(test_dataset_en, language='en')
    ner.save_predictions(en_results, 'predictions_english.conll')
    
    # Predict on German test dataset (using first 10 examples)
    print("\n=== Predicting on German test set ===")
    de_results = ner.predict_dataset(test_dataset_de.select(range(10)), language='de')
    ner.save_predictions(de_results, 'predictions_german.conll')
    
    print("\n=== Predictions complete ===")


if __name__ == "__main__":
    main()
