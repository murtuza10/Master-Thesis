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

from typing import Dict, List, Set
import warnings
from dotenv import load_dotenv
import os

load_dotenv()

class MultiLevelNERProcessor:
    """
    Process NER at different granularities:
    - Sentence-level for encoder models (token classification)
    - Document-level for LLMs (causal_lm and API)
    """
    
    def __init__(self, ner_system):
        self.ner = ner_system
    
    def split_into_sentences_simple(self, text: str, tokens: List[str]) -> List[Dict]:
        """Split tokens into sentences based on punctuation."""
        sentences = []
        current_sentence_tokens = []
        start_idx = 0
        
        for i, token in enumerate(tokens):
            current_sentence_tokens.append(token)
            
            # Check if this token ends a sentence
            if token in ['.', '!', '?'] or (len(token) > 1 and token[-1] in '.!?'):
                sent_text = ' '.join(current_sentence_tokens)
                sentences.append({
                    'text': sent_text,
                    'tokens': current_sentence_tokens.copy(),
                    'start_idx': start_idx,
                    'end_idx': i + 1
                })
                current_sentence_tokens = []
                start_idx = i + 1
        
        # Handle remaining tokens
        if current_sentence_tokens:
            sent_text = ' '.join(current_sentence_tokens)
            sentences.append({
                'text': sent_text,
                'tokens': current_sentence_tokens,
                'start_idx': start_idx,
                'end_idx': len(tokens)
            })
        
        return sentences if sentences else [{'text': text, 'tokens': tokens, 'start_idx': 0, 'end_idx': len(tokens)}]
    
    def predict_sentence_level(self, text: str, tokens: List[str], model_name: str,
                              model_type: str, entity_type: str) -> List[str]:
        """Predict entities at sentence level for encoder models."""
        sentences = self.split_into_sentences_simple(text, tokens)
        print(f"  → Processing {len(sentences)} sentences")
        
        all_predictions = ['O'] * len(tokens)
        
        for sentence in sentences:
            sent_tokens = sentence['tokens']
            start_idx = sentence['start_idx']
            
            if not sent_tokens:
                continue
            
            # Get predictions for this sentence
            sent_predictions = self.ner.predict_entity_with_model(
                sentence['text'], sent_tokens, model_name, model_type, entity_type
            )
            
            # Place predictions back into document
            for i, pred in enumerate(sent_predictions):
                doc_idx = start_idx + i
                if doc_idx < len(all_predictions):
                    all_predictions[doc_idx] = pred
        
        return all_predictions
    
    def predict_document_level(self, text: str, tokens: List[str], model_name: str,
                               model_type: str, entity_type: str) -> List[str]:
        """Predict entities at document level for LLMs."""
        print(f"  → Processing full document")
        
        predictions = self.ner.predict_entity_with_model(
            text, tokens, model_name, model_type, entity_type
        )
        
        return predictions
    
    def predict_with_granularity(self, text: str, tokens: List[str], model_name: str,
                                model_type: str, entity_type: str) -> List[str]:
        """Predict using appropriate granularity based on model type."""
        if model_type == 'token_classification':
            return self.predict_sentence_level(text, tokens, model_name, model_type, entity_type)
        elif model_type in ['causal_lm', 'api']:
            return self.predict_document_level(text, tokens, model_name, model_type, entity_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class EntitySpecificMerger:
    """
    Merge predictions from models where each model handles specific entities only.
    No conflicts should occur by design.
    """
    
    def __init__(self, model_configs: Dict):
        """
        Initialize with model configurations.
        
        Args:
            model_configs: Dictionary of model configurations with 'entities' key
        """
        self.model_configs = model_configs
        self._validate_no_overlaps()
    
    def _validate_no_overlaps(self):
        """
        Validate that no entity is assigned to multiple models (per language).
        Warns if overlaps are detected.
        """
        entity_to_models = {}
        
        for model_name, config in self.model_configs.items():
            for entity in config.get('entities', []):
                if entity not in entity_to_models:
                    entity_to_models[entity] = []
                entity_to_models[entity].append(model_name)
        
        overlaps = {entity: models for entity, models in entity_to_models.items() 
                   if len(models) > 1}
        
        if overlaps:
            warnings.warn(
                f"Entity overlaps detected - multiple models assigned to same entities:\n"
                f"{overlaps}\n"
                f"This may cause conflicts during merging."
            )
    
    def get_assigned_entities(self, model_name: str) -> Set[str]:
        """Get the set of entities assigned to a specific model."""
        if model_name not in self.model_configs:
            return set()
        return set(self.model_configs[model_name].get('entities', []))
    
    def filter_predictions_by_entity(self, predictions: List[str], 
                                    allowed_entities: Set[str]) -> List[str]:
        """
        Filter BIO tags to only include allowed entities.
        Any tags for non-allowed entities are converted to 'O'.
        
        Args:
            predictions: List of BIO tags
            allowed_entities: Set of allowed entity types
        
        Returns:
            Filtered list of BIO tags
        """
        filtered = []
        
        for tag in predictions:
            if tag == 'O':
                filtered.append('O')
            elif tag.startswith('B-') or tag.startswith('I-'):
                entity_type = tag[2:]
                if entity_type in allowed_entities:
                    filtered.append(tag)
                else:
                    # This entity is not handled by this model
                    filtered.append('O')
            else:
                # Unknown tag format, keep as is
                filtered.append(tag)
        
        return filtered
    
    def merge_predictions_entity_specific(
        self,
        all_predictions: Dict[str, List[str]],
        tokens: List[str],
        model_assignments: Dict[str, str] = None
    ) -> Dict:
        """
        Merge predictions where each entity type comes from one specific model.
        
        Args:
            all_predictions: Dict mapping entity_type -> BIO tag predictions
            tokens: List of tokens
            model_assignments: Dict mapping entity_type -> model_name (optional)
        
        Returns:
            Dictionary with:
                - 'labels': Final merged BIO tags
                - 'warnings': List of any warnings/issues
                - 'entity_sources': Dict showing which model provided each entity
        """
        final_labels = ['O'] * len(tokens)
        warnings_list = []
        entity_sources = {}
        
        # Track which positions are already filled to detect conflicts
        filled_positions = {}  # position -> (entity_type, model_name)
        
        for entity_type, predictions in all_predictions.items():
            # Validate prediction length
            if len(predictions) != len(tokens):
                warning = (f"Skipping {entity_type}: prediction length mismatch "
                          f"({len(predictions)} vs {len(tokens)} tokens)")
                warnings_list.append(warning)
                print(f"Warning: {warning}")
                continue
            
            # Get model name for this entity
            model_name = model_assignments.get(entity_type, 'unknown') if model_assignments else 'unknown'
            
            # Get allowed entities for this model
            allowed_entities = self.get_assigned_entities(model_name)
            
            # Filter predictions to only include assigned entities
            if allowed_entities and entity_type not in allowed_entities:
                warning = (f"Entity '{entity_type}' not in assigned entities "
                          f"for model '{model_name}': {allowed_entities}")
                warnings_list.append(warning)
                print(f"Warning: {warning}")
                # Still process, but flag it
            
            # Filter the predictions
            filtered_predictions = self.filter_predictions_by_entity(
                predictions, {entity_type}
            )
            
            # Merge into final labels
            for i, pred in enumerate(filtered_predictions):
                if pred != 'O':
                    # Check for conflicts
                    if final_labels[i] != 'O':
                        existing_entity = final_labels[i][2:] if final_labels[i] != 'O' else None
                        existing_model = filled_positions.get(i, ('unknown', 'unknown'))[1]
                        
                        warning = (f"Conflict at position {i} (token: '{tokens[i]}'): "
                                 f"{existing_entity} from {existing_model} vs "
                                 f"{entity_type} from {model_name}")
                        warnings_list.append(warning)
                        print(f"Warning: {warning}")
                        
                        # Keep existing (first one wins)
                        continue
                    
                    # Assign this prediction
                    final_labels[i] = pred
                    filled_positions[i] = (entity_type, model_name)
                    
                    # Track source
                    if entity_type not in entity_sources:
                        entity_sources[entity_type] = model_name
        
        # Validate BIO sequence integrity
        final_labels = self._validate_bio_sequence(final_labels, warnings_list)
        
        return {
            'labels': final_labels,
            'warnings': warnings_list,
            'entity_sources': entity_sources,
            'conflict_count': len([w for w in warnings_list if 'Conflict' in w])
        }
    
    def _validate_bio_sequence(self, tags: List[str], warnings_list: List[str]) -> List[str]:
        """
        Validate and fix BIO tag sequences.
        Reports issues via warnings_list.
        """
        validated = []
        prev_tag = 'O'
        fixes_made = 0
        
        for i, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]
                
                # Check if this I- tag follows appropriate B- or I- tag
                prev_entity = prev_tag[2:] if prev_tag.startswith(('B-', 'I-')) else None
                
                if prev_tag == 'O' or prev_entity != entity_type:
                    # Invalid I- tag, convert to B-
                    validated.append(f'B-{entity_type}')
                    fixes_made += 1
                else:
                    validated.append(tag)
            else:
                validated.append(tag)
            
            prev_tag = validated[-1]
        
        if fixes_made > 0:
            warnings_list.append(f"Fixed {fixes_made} invalid BIO sequence(s)")
        
        return validated

class MultiModelNER:
    """
    Multi-Model Named Entity Recognition system that uses different specialized models
    for different entity types with model-specific label lists.
    """
    
    def __init__(self, global_label_list: List[str]):
        self.global_label_list = global_label_list
        self.global_id2label = {i: label for i, label in enumerate(global_label_list)}
        self.global_label2id = {label: i for i, label in enumerate(global_label_list)}
        self.ner_cache = {}  # Cache for API and causal_lm results

        
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
                'entities': ['cropSpecies', 'soilReferenceGroup', 'endTime', 'duration','startTime','soilOrganicCarbon'],
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
            self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
        
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
                self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
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
            'soilBulkDensity': 'The dry weight of soil divided by its volume. Please annotate the term “bulk density” if it is mentioned in a text',
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
        # print(f"Performing NER using DeepSeek API with text: {text}")
        """Perform NER using DeepSeek API."""
        if examples is None:
            if self.faiss_index is not None:
                examples, _ = self.get_similar_examples(text, top_k=5)
            else:
                examples = []
        # print(f"Examples: {examples}")
        system_prompt, user_prompt = self.generate_ner_prompts(text)
        
        blocks = []
        for i, (example_text, ents) in enumerate(examples):
            blocks.append(f"### Example {i} ###:\nInput Text:\n {example_text}\nOutput: {json.dumps(ents, ensure_ascii=False)}")
        blocks_str = "\n\n".join(blocks)
        system_prompt += blocks_str
        
        # print(f"system prompt is {system_prompt}")

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
        
        # print(f"system prompt is {system_prompt}")
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
        print(f"qwen response is {response}")
        
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
            # Check cache first
            cache_key = f"api_{model_name}_{text}"
            if cache_key not in self.ner_cache:
                # Only call API once and cache all entities
                response = self.perform_ner_deepseek(text)
                all_entities = []
                
                try:
                    if 'choices' in response and len(response['choices']) > 0:
                        content = response['choices'][0]['message']['content']
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            result = json.loads(json_str)
                            all_entities = result.get('entities', [])
                except Exception as e:
                    print(f"Error parsing DeepSeek response: {str(e)}")
                
                self.ner_cache[cache_key] = all_entities
            
            # Use cached entities
            all_entities = self.ner_cache[cache_key]
            bio_tags = ['O'] * len(tokens)
            
            for entity in all_entities:
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
            
            return bio_tags
        
        elif model_type == 'causal_lm':
            # Check cache first
            cache_key = f"causal_{model_name}_{text}"
            if cache_key not in self.ner_cache:
                # Only call causal_lm once and cache all entities
                all_entities_result = self.perform_ner_causal_lm_all_entities(text, tokens, model_name)
                self.ner_cache[cache_key] = all_entities_result
            
            # Use cached results and filter for target_entity
            cached_result = self.ner_cache[cache_key]
            return self.filter_entities_by_target(cached_result, target_entity, tokens)
        
        elif model_type == 'token_classification':
            # Token classification models predict one entity at a time, no caching needed
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
                    # ✅ STRICT FILTERING: Only keep target_entity predictions
                    if label != 'O':
                        # Extract entity type from BIO tag
                        entity_type = label[2:] if label.startswith(('B-', 'I-')) else None
                        
                        # Only keep if it matches target_entity
                        if entity_type == target_entity:
                            aligned_labels.append(label)
                        else:
                            # Different entity type - filter it out
                            aligned_labels.append('O')
                    else:
                        aligned_labels.append('O')
                    previous_word_id = word_id
            
            while len(aligned_labels) < len(tokens):
                aligned_labels.append('O')
            aligned_labels = aligned_labels[:len(tokens)]
            
            return aligned_labels
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    def perform_ner_causal_lm_all_entities(self, text: str, tokens: List[str], model_name: str) -> dict:
        """
        Perform NER with causal LM and return ALL entities (not filtered by target).
        Returns a dictionary mapping entity types to their BIO tags.
        """
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        examples = None
        if self.faiss_index is not None:
            examples, _ = self.get_similar_examples(text, top_k=5)
        
        response = self.perform_ner_llama(model, tokenizer, text, max_length=1500, examples=examples)
        print(f"qwen response is {response}")
        
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
        
        # Create a dictionary mapping entity types to BIO tags
        entity_dict = {}
        
        # Get all possible entity types from model config
        model_entities = self.model_configs[model_name].get('entities', [])
        
        # Initialize all entity types with 'O' tags
        for entity_type in model_entities:
            entity_dict[entity_type] = ['O'] * len(tokens)
        
        # Fill in the actual entities
        for entity in entities:
            entity_label = entity.get('label', '')
            entity_text = entity.get('text', '')
            
            if not entity_text:
                continue
            
            # Find which entity type this belongs to
            matched_entity_type = None
            for entity_type in model_entities:
                if entity_type.lower() in entity_label.lower():
                    matched_entity_type = entity_type
                    break
            
            if not matched_entity_type:
                continue
            
            # Match entity text to tokens
            entity_words = entity_text.split()
            for i in range(len(tokens) - len(entity_words) + 1):
                if tokens[i:i+len(entity_words)] == entity_words or \
                ' '.join(tokens[i:i+len(entity_words)]).lower() == entity_text.lower():
                    entity_dict[matched_entity_type][i] = f"B-{matched_entity_type}"
                    for j in range(i+1, i+len(entity_words)):
                        entity_dict[matched_entity_type][j] = f"I-{matched_entity_type}"
                    break
        
        return entity_dict


    def filter_entities_by_target(self, all_entities_result: dict, target_entity: str, tokens: List[str]) -> List[str]:
        """Filter cached causal_lm results to extract only the target entity."""
        if all_entities_result is None or not all_entities_result:
            # Return empty tags if no results
            return ['O'] * len(tokens)
        
        # Try to get the target entity
        if target_entity in all_entities_result:
            return all_entities_result[target_entity]
        
        # If target not found, return 'O' tags with same length as tokens
        return ['O'] * len(tokens)


    def clear_cache(self):
        """Clear the NER cache (call this when processing a new text)."""
        self.ner_cache.clear()
    
    def improved_merge_predictions(self, all_predictions: Dict[str, List[str]], 
                               tokens: List[str]) -> List[str]:
        """
        Replacement merge_predictions method for MultiModelNER class.
        This version respects entity assignments and detects conflicts.
        """
        # Initialize merger if needed
        if not hasattr(self, '_entity_merger'):
            self._entity_merger = EntitySpecificMerger(self.model_configs)
        
        # Build model assignments for each entity type
        model_assignments = {}
        for entity_type in all_predictions.keys():
            for model_name, config in self.model_configs.items():
                if entity_type in config.get('entities', []):
                    model_assignments[entity_type] = model_name
                    break
        
        # Perform merge
        result = self._entity_merger.merge_predictions_entity_specific(
            all_predictions, tokens, model_assignments
        )
        
        # Log if conflicts detected
        if result['conflict_count'] > 0:
            print(f"\n⚠️  {result['conflict_count']} conflicts detected during merging!")
            print("This suggests entity assignments may overlap.")
        
        return result['labels']
        
    def improved_predict(self, text: str, tokens: List[str] = None, language: str = 'en') -> Dict:
        """
        Improved predict method with sentence/document level processing.
        
        - Encoder models (token_classification): sentence-level
        - LLMs (causal_lm, api): document-level
        """
        if tokens is None:
            tokens = text.split()
        
        # Initialize multi-level processor if needed
        if not hasattr(self, '_ml_processor'):
            self._ml_processor = MultiLevelNERProcessor(self)
        
        all_predictions = {}
        processing_stats = {
            'sentence_level': [],
            'document_level': []
        }
        
        # Iterate through model configs
        for model_name, config in self.model_configs.items():
            assigned_entities = config.get('entities', [])
            
            # Check language support
            if language not in config.get('languages', []):
                print(f"Skipping {model_name}: language '{language}' not supported")
                continue
            
            # Get model and tokenizer
            if config['type'] == 'api':
                model = None
                tokenizer = None
            else:
                model = self.models.get(model_name)
                tokenizer = self.tokenizers.get(model_name)
                
                if model is None or tokenizer is None:
                    print(f"Warning: Model or tokenizer not loaded for {model_name}")
                    continue
            
            # Process each assigned entity
            for entity_type in assigned_entities:
                # Check for language overrides
                if entity_type in self.language_overrides:
                    if language in self.language_overrides[entity_type]:
                        override_model = self.language_overrides[entity_type][language]
                        if override_model != model_name:
                            print(f"Skipping {entity_type} for {model_name}: "
                                f"overridden by {override_model} for language '{language}'")
                            continue
                
                print(f"Processing {entity_type} with {model_name} ({config['type']})")
                
                try:
                    # Use appropriate granularity based on model type
                    predictions = self._ml_processor.predict_with_granularity(
                        text, tokens, model_name, config['type'], entity_type
                    )
                    all_predictions[entity_type] = predictions
                    
                    # Track processing level
                    if config['type'] == 'token_classification':
                        processing_stats['sentence_level'].append(entity_type)
                    else:
                        processing_stats['document_level'].append(entity_type)
                        
                except Exception as e:
                    print(f"Error processing {entity_type} with {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Merge predictions
        final_predictions = self.improved_merge_predictions(all_predictions, tokens)
        
        print("\n" + "-"*60)
        print("PROCESSING SUMMARY")
        print(f"Sentence-level: {', '.join(processing_stats['sentence_level']) or 'None'}")
        print(f"Document-level: {', '.join(processing_stats['document_level']) or 'None'}")
        print("-"*60)
        
        return {
            'tokens': tokens,
            'labels': final_predictions,
            'entity_predictions': all_predictions,
            'processing_stats': processing_stats
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
            
            result = self.improved_predict(text, tokens=tokens, language=language)
            
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
    ner.save_predictions(de_results, 'predictions_german_multimodel_ver3.conll')
    metrics_de = ner.calculate_f1_scores(de_results, output_file='f1_scores_german_multimodel_ver3.txt')
    
    # Load and predict on English test dataset
    print("\n" + "="*60)
    print("ENGLISH TEST DATASET EVALUATION")
    print("="*60)
    test_dataset_en = ner.load_hf_dataset(ner.test_dataset_en_path)
    en_results = ner.predict_dataset(test_dataset_en, language='en')
    ner.save_predictions(en_results, 'predictions_english_multimodel_ver3.conll')
    metrics_en = ner.calculate_f1_scores(en_results, output_file='f1_scores_english_multimodel_ver3.txt')
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"German F1 Score:  {metrics_de.get('f1_score', 0):.4f}")
    print(f"English F1 Score: {metrics_en.get('f1_score', 0):.4f}")


if __name__ == "__main__":
    main()
