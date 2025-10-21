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
import logging
from datetime import datetime

load_dotenv()

# Setup comprehensive logging
def setup_logging():
    """Setup comprehensive logging for debugging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ensemble_debug_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    # Create specific loggers for different components
    loggers = {
        'main': logging.getLogger('MultiModelNER'),
        'merger': logging.getLogger('ConfidenceBasedMerger'),
        'processor': logging.getLogger('MultiLevelNERProcessor'),
        'models': logging.getLogger('ModelOperations'),
        'confidence': logging.getLogger('ConfidenceCalculation'),
        'entity_matching': logging.getLogger('EntityMatching')
    }
    
    return loggers

# Initialize logging
LOGGERS = setup_logging()

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
        LOGGERS['processor'].debug(f"Splitting text into sentences. Text: '{text[:100]}...', Tokens: {len(tokens)}")
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
        
        result = sentences if sentences else [{'text': text, 'tokens': tokens, 'start_idx': 0, 'end_idx': len(tokens)}]
        LOGGERS['processor'].debug(f"Split into {len(result)} sentences: {[s['start_idx'] for s in result]}")
        return result
    
    def predict_sentence_level(self, text: str, tokens: List[str], model_name: str,
                              model_type: str, entity_type: str) -> Tuple[List[str], Dict]:
        """Predict entities at sentence level for encoder models."""
        LOGGERS['processor'].debug(f"Sentence-level prediction: model={model_name}, entity={entity_type}, tokens={len(tokens)}")
        sentences = self.split_into_sentences_simple(text, tokens)
        print(f"  → Processing {len(sentences)} sentences")
        LOGGERS['processor'].info(f"Processing {len(sentences)} sentences for {entity_type} with {model_name}")
        
        all_predictions = ['O'] * len(tokens)
        all_confidence_data = []
        
        for sentence in sentences:
            sent_tokens = sentence['tokens']
            start_idx = sentence['start_idx']
            
            if not sent_tokens:
                LOGGERS['processor'].debug(f"Skipping empty sentence at index {start_idx}")
                continue
            
            LOGGERS['processor'].debug(f"Processing sentence {start_idx}: '{sentence['text'][:50]}...', tokens: {len(sent_tokens)}")
            
            # Get predictions for this sentence
            sent_predictions, sent_confidence = self.ner.predict_entity_with_model(
                sentence['text'], sent_tokens, model_name, model_type, entity_type
            )
            
            LOGGERS['processor'].debug(f"Sentence {start_idx} predictions: {sent_predictions[:10]}... (showing first 10)")
            LOGGERS['processor'].debug(f"Sentence {start_idx} confidence data type: {type(sent_confidence)}")
            
            # Place predictions back into document
            # Handle both list (token classification) and dict (causal LM/API) confidence formats
            if isinstance(sent_confidence, list):
                # Token classification returns list of confidence data per token
                for i, (pred, conf) in enumerate(zip(sent_predictions, sent_confidence)):
                    doc_idx = start_idx + i
                    if doc_idx < len(all_predictions):
                        all_predictions[doc_idx] = pred
                        if doc_idx < len(all_confidence_data):
                            all_confidence_data[doc_idx] = conf
                        else:
                            all_confidence_data.extend([{}] * (doc_idx - len(all_confidence_data) + 1))
                            all_confidence_data[doc_idx] = conf
            else:
                # Causal LM/API returns single confidence dict for entire sentence
                for i, pred in enumerate(sent_predictions):
                    doc_idx = start_idx + i
                    if doc_idx < len(all_predictions):
                        all_predictions[doc_idx] = pred
                        if doc_idx < len(all_confidence_data):
                            all_confidence_data[doc_idx] = sent_confidence
                        else:
                            all_confidence_data.extend([{}] * (doc_idx - len(all_confidence_data) + 1))
                            all_confidence_data[doc_idx] = sent_confidence
        
        # Fill remaining positions with default confidence
        while len(all_confidence_data) < len(tokens):
            all_confidence_data.append({})
        
        return all_predictions, all_confidence_data
    
    def predict_document_level(self, text: str, tokens: List[str], model_name: str,
                               model_type: str, entity_type: str) -> Tuple[List[str], Dict]:
        """Predict entities at document level for LLMs."""
        print(f"  → Processing full document")
        
        predictions, confidence_data = self.ner.predict_entity_with_model(
            text, tokens, model_name, model_type, entity_type
        )
        
        return predictions, confidence_data
    
    def predict_with_granularity(self, text: str, tokens: List[str], model_name: str,
                                model_type: str, entity_type: str) -> Tuple[List[str], Dict]:
        """Predict using appropriate granularity based on model type."""
        if model_type == 'token_classification':
            return self.predict_sentence_level(text, tokens, model_name, model_type, entity_type)
        elif model_type in ['causal_lm', 'api']:
            return self.predict_document_level(text, tokens, model_name, model_type, entity_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class ConfidenceBasedMerger:
    """
    Merge predictions from models using confidence scores for conflict resolution.
    Supports both confidence-based and fixed precedence fallback.
    """
    
    def __init__(self, model_configs: Dict, model_precedence: List[str] = None, 
                 use_confidence: bool = True):
        """
        Initialize with model configurations and precedence order.
        
        Args:
            model_configs: Dictionary of model configurations with 'entities' key
            model_precedence: List of model names in order of precedence (highest first)
                            Used as fallback when confidence scores are equal
            use_confidence: Whether to use confidence scores for precedence (default: True)
        """
        self.model_configs = model_configs
        self.model_precedence = model_precedence or []
        self.use_confidence = use_confidence
        self._validate_no_overlaps()
        
        # Create precedence map for quick lookups (fallback)
        self.precedence_map = {model: idx for idx, model in enumerate(self.model_precedence)}
    
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
                f"Conflicts will be resolved using precedence order: {self.model_precedence}"
            )
    
    def get_model_precedence(self, model_name: str) -> int:
        """
        Get precedence score for a model (lower = higher priority).
        Models not in precedence list get lowest priority.
        """
        if model_name in self.precedence_map:
            return self.precedence_map[model_name]
        else:
            # Models not in precedence list get lowest priority
            return len(self.model_precedence) + 1000
    
    def calculate_confidence_score(self, prediction: str, model_name: str, 
                                 confidence_data: Dict = None) -> float:
        """
        Calculate confidence score for a prediction.
        
        Args:
            prediction: The BIO tag prediction
            model_name: Name of the model that made the prediction
            confidence_data: Additional confidence information (logits, probabilities, etc.)
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        LOGGERS['confidence'].debug(f"Calculating confidence for prediction='{prediction}', model='{model_name}', data_keys={list(confidence_data.keys()) if confidence_data else None}")
        
        if prediction == 'O':
            LOGGERS['confidence'].debug(f"O tag confidence: 0.2")
            return 0.2  # Lower confidence for O tags to further reduce suppression
        
        if not self.use_confidence or confidence_data is None:
            # Fallback to precedence-based confidence
            precedence_score = self.get_model_precedence(model_name)
            max_precedence = len(self.model_precedence)
            return 1.0 - (precedence_score / max_precedence) if max_precedence > 0 else 0.5
        
        # Extract confidence based on model type
        model_type = self.model_configs.get(model_name, {}).get('type', 'unknown')
        LOGGERS['confidence'].debug(f"Model type: {model_type}")
        
        if model_type == 'token_classification':
            conf = self._extract_token_classification_confidence(confidence_data, prediction)
            LOGGERS['confidence'].debug(f"Token classification confidence: {conf}")
            return conf
        elif model_type == 'causal_lm':
            conf = self._extract_causal_lm_confidence(confidence_data, prediction)
            LOGGERS['confidence'].debug(f"Causal LM confidence: {conf}")
            return conf
        elif model_type == 'api':
            conf = self._extract_api_confidence(confidence_data, prediction)
            LOGGERS['confidence'].debug(f"API confidence: {conf}")
            return conf
        else:
            # Default confidence based on precedence
            precedence_score = self.get_model_precedence(model_name)
            max_precedence = len(self.model_precedence)
            conf = 1.0 - (precedence_score / max_precedence) if max_precedence > 0 else 0.5
            LOGGERS['confidence'].debug(f"Default precedence confidence: {conf}")
            return conf
    
    def _extract_token_classification_confidence(self, confidence_data: Dict, prediction: str) -> float:
        """Extract confidence from token classification model outputs."""
        if 'logits' in confidence_data and 'predicted_idx' in confidence_data:
            logits = confidence_data['logits']
            predicted_idx = confidence_data['predicted_idx']
            
            # Convert logits to probabilities using softmax
            import torch
            import torch.nn.functional as F
            
            try:
                if isinstance(logits, torch.Tensor):
                    probs = F.softmax(logits, dim=-1)
                    confidence = probs[predicted_idx].item()
                else:
                    # Handle numpy arrays
                    import numpy as np
                    probs = torch.softmax(torch.tensor(logits), dim=-1)
                    confidence = probs[predicted_idx].item()
                
                return float(confidence)
            except (IndexError, ValueError, TypeError) as e:
                print(f"Warning: Error calculating confidence from logits: {e}")
                return 0.7  # Default confidence for token classification
        
        return 0.7  # Default confidence for token classification
    
    def _extract_causal_lm_confidence(self, confidence_data: Dict, prediction: str) -> float:
        """Extract confidence from causal LM outputs."""
        if 'response_quality' in confidence_data:
            # Could be based on response length, JSON validity, etc.
            quality_score = confidence_data['response_quality']
            return min(1.0, max(0.0, quality_score))
        
        if 'temperature' in confidence_data:
            # Lower temperature suggests higher confidence
            temp = confidence_data['temperature']
            return max(0.3, 1.0 - temp)
        
        return 0.6  # Default confidence for causal LM
    
    def _extract_api_confidence(self, confidence_data: Dict, prediction: str) -> float:
        """Extract confidence from API responses."""
        if 'response_quality' in confidence_data:
            return min(1.0, max(0.0, confidence_data['response_quality']))
        
        # API responses are generally less reliable than local models
        return 0.5
    
    def resolve_conflict(self, existing_model: str, new_model: str, 
                        position: int, token: str, existing_confidence: float = 0.0,
                        new_confidence: float = 0.0) -> str:
        """
        Resolve conflict between two models using confidence scores.
        
        Args:
            existing_model: Model that currently has the position
            new_model: Model trying to take the position
            position: Token position in the sequence
            token: The actual token text
            existing_confidence: Confidence score of existing prediction
            new_confidence: Confidence score of new prediction
        
        Returns:
            The model name that should win the conflict
        """
        if self.use_confidence:
            confidence_diff = new_confidence - existing_confidence
            
            # If confidence difference is significant (>0.2), use confidence
            if abs(confidence_diff) > 0.2:
                if new_confidence > existing_confidence:
                    print(f"  ✓ Position {position} ('{token}'): {new_model} overrides {existing_model} "
                          f"(confidence: {new_confidence:.3f} vs {existing_confidence:.3f})")
                    return new_model
                else:
                    print(f"  ✗ Position {position} ('{token}'): {existing_model} keeps position "
                          f"(confidence: {existing_confidence:.3f} vs {new_confidence:.3f})")
                    return existing_model
            
            # If confidence is similar, fall back to precedence
            print(f"  ~ Position {position} ('{token}'): Similar confidence, using precedence")
        
        # Fallback to precedence-based resolution
        existing_precedence = self.get_model_precedence(existing_model)
        new_precedence = self.get_model_precedence(new_model)
        
        if new_precedence < existing_precedence:
            print(f"  ✓ Position {position} ('{token}'): {new_model} overrides {existing_model} (precedence)")
            return new_model
        else:
            print(f"  ✗ Position {position} ('{token}'): {existing_model} keeps position (precedence)")
            return existing_model
    
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
                    filtered.append('O')
            else:
                filtered.append(tag)
        
        return filtered
    
    def merge_predictions_entity_specific(
        self,
        all_predictions: Dict[str, List[str]],
        tokens: List[str],
        model_assignments: Dict[str, str] = None,
        confidence_data: Dict[str, Dict] = None
    ) -> Dict:
        """
        Merge predictions where each entity type comes from one specific model.
        Uses confidence scores or precedence system to resolve conflicts.
        
        Args:
            all_predictions: Dict mapping entity_type -> BIO tag predictions
            tokens: List of tokens
            model_assignments: Dict mapping entity_type -> model_name (optional)
            confidence_data: Dict mapping entity_type -> confidence information (optional)
        
        Returns:
            Dictionary with:
                - 'labels': Final merged BIO tags
                - 'warnings': List of any warnings/issues
                - 'entity_sources': Dict showing which model provided each entity
                - 'conflict_count': Number of conflicts encountered
                - 'confidence_overrides': Number of times confidence was used
                - 'precedence_overrides': Number of times precedence was used
        """
        final_labels = ['O'] * len(tokens)
        warnings_list = []
        entity_sources = {}
        
        # Track which positions are filled and by which model
        filled_positions = {}  # position -> (entity_type, model_name, confidence)
        
        conflict_count = 0
        confidence_override_count = 0
        precedence_override_count = 0
        
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
            
            # Filter the predictions
            filtered_predictions = self.filter_predictions_by_entity(
                predictions, {entity_type}
            )
            
            # Get confidence data for this entity type
            entity_confidence_data = confidence_data.get(entity_type, {}) if confidence_data else {}
            
            # Merge into final labels
            for i, pred in enumerate(filtered_predictions):
                if pred != 'O':
                    # Calculate confidence for this prediction
                    # Handle both list (token classification) and dict (causal LM/API) formats
                    if isinstance(entity_confidence_data, list) and i < len(entity_confidence_data):
                        token_confidence_data = entity_confidence_data[i]
                    else:
                        token_confidence_data = entity_confidence_data
                    
                    pred_confidence = self.calculate_confidence_score(pred, model_name, token_confidence_data)
                    
                    # Check for conflicts
                    if final_labels[i] != 'O':
                        existing_data = filled_positions.get(i, ('unknown', 'unknown', 0.0))
                        existing_entity, existing_model, existing_confidence = existing_data
                        
                        conflict_count += 1
                        warning = (f"Conflict at position {i} (token: '{tokens[i]}'): "
                                 f"{existing_entity} from {existing_model} vs "
                                 f"{entity_type} from {model_name}")
                        warnings_list.append(warning)
                        print(f"⚠️  {warning}")
                        
                        # Use confidence or precedence to resolve conflict
                        winner = self.resolve_conflict(existing_model, model_name, i, tokens[i],
                                                     existing_confidence, pred_confidence)
                        
                        if winner == model_name:
                            # New model wins, override existing
                            final_labels[i] = pred
                            filled_positions[i] = (entity_type, model_name, pred_confidence)
                            if self.use_confidence and abs(pred_confidence - existing_confidence) > 0.1:
                                confidence_override_count += 1
                            else:
                                precedence_override_count += 1
                        # else: existing model keeps position, no change needed
                        
                        continue
                    
                    # No conflict, assign this prediction
                    final_labels[i] = pred
                    filled_positions[i] = (entity_type, model_name, pred_confidence)
                    
                    # Track source
                    if entity_type not in entity_sources:
                        entity_sources[entity_type] = model_name
        
        # Validate BIO sequence integrity
        final_labels = self._validate_bio_sequence(final_labels, warnings_list)
        
        print(f"\n📊 Merge Statistics:")
        print(f"   Conflicts: {conflict_count}")
        if self.use_confidence:
            print(f"   Confidence overrides: {confidence_override_count}")
        print(f"   Precedence overrides: {precedence_override_count}")
        
        return {
            'labels': final_labels,
            'warnings': warnings_list,
            'entity_sources': entity_sources,
            'conflict_count': conflict_count,
            'confidence_overrides': confidence_override_count,
            'precedence_overrides': precedence_override_count
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
    Multi-Model Named Entity Recognition system with precedence-based conflict resolution.
    """
    
    def __init__(self, global_label_list: List[str], model_precedence: List[str] = None,
                 enable_api: bool = True, enable_causal_lm: bool = True,
                 enable_multi_model_fusion: bool = False):
        """
        Initialize MultiModelNER with optional model precedence order.
        
        Args:
            global_label_list: List of all possible labels
            model_precedence: List of model names in order of precedence (highest to lowest)
                            Example: ['qwen_2.5_7b', 'roberta_all_specific', 'deepseek']
        """
        self.global_label_list = global_label_list
        self.global_id2label = {i: label for i, label in enumerate(global_label_list)}
        self.global_label2id = {label: i for i, label in enumerate(global_label_list)}
        self.ner_cache = {}
        self.model_precedence = model_precedence or []
        self.enable_api = enable_api
        self.enable_causal_lm = enable_causal_lm
        self.enable_multi_model_fusion = enable_multi_model_fusion
        
        print(f"\n{'='*60}")
        print("MODEL PRECEDENCE ORDER (Highest → Lowest)")
        print(f"{'='*60}")
        if self.model_precedence:
            for idx, model in enumerate(self.model_precedence, 1):
                print(f"{idx}. {model}")
        else:
            print("No precedence order set - conflicts will use first-come-first-served")
        print(f"{'='*60}\n")

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
                'Location': 'locationName',
                'Crop': 'cropSpecies',
                'TimeStatement': 'startTime'
            }
        }
        
        # Model configurations
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
                'path': None,
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
        self.model_id2label = {}
        self.model_label2id = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FAISS index and corpus
        self.embedder = None
        self.faiss_index = None
        self.corpus_texts = []
        self.corpus_entities = []
        
        # Dataset paths
        self.embeddings_dataset = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/ner_dataset_input_output.jsonl"
        self.test_dataset_en_path = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/Test_NER_dataset_English.json"
        self.test_dataset_de_path = "/home/s27mhusa_hpc/Master-Thesis/DatasetsFinalPredictions/Test_NER_dataset_German.json"
    
    def get_model_labels(self, model_name: str) -> Tuple[Dict, Dict]:
        """Get model-specific label mappings."""
        if model_name in self.model_label_lists:
            labels = self.model_label_lists[model_name]
            id2label = {i: label for i, label in enumerate(labels)}
            label2id = {label: i for i, label in enumerate(labels)}
            return id2label, label2id
        else:
            return self.global_id2label, self.global_label2id
    
    def map_entity_name(self, model_name: str, entity_name: str) -> str:
        """Map entity name from model-specific to global entity name."""
        if model_name in self.entity_mapping:
            if entity_name.startswith('B-') or entity_name.startswith('I-'):
                prefix = entity_name[:2]
                entity_type = entity_name[2:]
                
                if entity_type in self.entity_mapping[model_name]:
                    mapped_type = self.entity_mapping[model_name][entity_type]
                    return f"{prefix}{mapped_type}"
        
        return entity_name
    
    def load_model_token_classification(self, model_path: str, model_name: str):
        """Load standard token classification model with model-specific labels."""
        print(f"Loading token classification model: {model_name}")
        
        id2label, label2id = self.get_model_labels(model_name)
        self.model_id2label[model_name] = id2label
        self.model_label2id[model_name] = label2id
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
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
        """Load causal LM model."""
        print(f"Loading causal LM model from: {model_path}")
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
                    if not self.enable_api:
                        print(f"Skipping {model_name} - API disabled by flag")
                        continue
                    print(f"Skipping {model_name} - API-based model")
                    continue
                elif config['type'] == 'causal_lm':
                    if not self.enable_causal_lm:
                        print(f"Skipping {model_name} - causal LM disabled by flag")
                        continue
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
            'soilBulkDensity': 'The dry weight of soil divided by its volume. Please annotate the term "bulk density" if it is mentioned in a text: ',
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
            entity_label = entity.get('label', '').lower()
            LOGGERS['entity_matching'].debug(f"Checking entity: label='{entity_label}', target='{target_entity.lower()}'")
            
            # Use more precise matching to avoid false positives
            if not (target_entity.lower() == entity_label or 
                   f"{target_entity.lower()}" in entity_label.split() or
                   (target_entity.lower() in entity_label and 
                    (entity_label.startswith(target_entity.lower()) or 
                     entity_label.endswith(target_entity.lower())))):
                LOGGERS['entity_matching'].debug(f"Entity '{entity_label}' does not match target '{target_entity}'")
                continue
            
            LOGGERS['entity_matching'].debug(f"Entity '{entity_label}' matches target '{target_entity}'")
            
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
                          model_type: str, target_entity: str) -> Tuple[List[str], Dict]:
        """Predict entities using a specific model and return confidence data."""
        
        if model_type == 'api':
            cache_key = f"api_{model_name}_{text}"
            if cache_key not in self.ner_cache:
                response = self.perform_ner_deepseek(text)
                all_entities = []
                confidence_data = {'response_quality': 0.5}  # Default API confidence
                
                try:
                    if 'choices' in response and len(response['choices']) > 0:
                        content = response['choices'][0]['message']['content']
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            result = json.loads(json_str)
                            all_entities = result.get('entities', [])
                            confidence_data['response_quality'] = 0.8  # Good JSON response
                        else:
                            confidence_data['response_quality'] = 0.3  # Poor JSON response
                except Exception as e:
                    print(f"Error parsing DeepSeek response: {str(e)}")
                    confidence_data['response_quality'] = 0.2  # Parsing error
                
                self.ner_cache[cache_key] = (all_entities, confidence_data)
            
            all_entities, confidence_data = self.ner_cache[cache_key]
            bio_tags = ['O'] * len(tokens)
            
            for entity in all_entities:
                entity_label = entity.get('label', '').lower()
                # Use more precise matching to avoid false positives
                if not (target_entity.lower() == entity_label or 
                       f"{target_entity.lower()}" in entity_label.split() or
                       (target_entity.lower() in entity_label and 
                        (entity_label.startswith(target_entity.lower()) or 
                         entity_label.endswith(target_entity.lower())))):
                    continue
                
                entity_text = entity.get('text', '')
                entity_words = entity_text.split()
                
                for i in range(len(tokens) - len(entity_words) + 1):
                    # Use consistent matching logic with causal LM
                    if (tokens[i:i+len(entity_words)] == entity_words or 
                        ' '.join(tokens[i:i+len(entity_words)]).lower() == entity_text.lower()):
                        bio_tags[i] = f"B-{target_entity}"
                        for j in range(i+1, i+len(entity_words)):
                            bio_tags[j] = f"I-{target_entity}"
                        break
            
            return bio_tags, confidence_data
        
        elif model_type == 'causal_lm':
            cache_key = f"causal_{model_name}_{text}"
            if cache_key not in self.ner_cache:
                all_entities_result, confidence_data = self.perform_ner_causal_lm_all_entities(text, tokens, model_name)
                self.ner_cache[cache_key] = (all_entities_result, confidence_data)
            
            cached_result, confidence_data = self.ner_cache[cache_key]
            predictions = self.filter_entities_by_target(cached_result, target_entity, tokens)
            return predictions, confidence_data
        
        elif model_type == 'token_classification':
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            id2label = self.model_id2label.get(model_name, self.global_id2label)
            
            encoding = tokenizer(tokens, is_split_into_words=True, 
                            return_tensors='pt', truncation=True, 
                            padding=True, return_offsets_mapping=True)
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                logits = outputs.logits[0]  # Store logits for confidence calculation
            
            predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
            predicted_labels = [self.map_entity_name(model_name, label) for label in predicted_labels]
            
            # Store confidence data
            confidence_data = {
                'logits': logits.cpu().numpy(),
                'predicted_indices': [pred.item() for pred in predictions[0]]
            }
            
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            aligned_confidence_data = []
            previous_word_id = None
            word_logits = []
            word_pred_indices = []
            
            for i, (word_id, label) in enumerate(zip(word_ids, predicted_labels)):
                if word_id is None:
                    continue
                if word_id != previous_word_id:
                    if word_logits:
                        # Store confidence data for the previous word
                        try:
                            avg_logits = torch.stack(word_logits).mean(dim=0).cpu().numpy()
                            predicted_idx = word_pred_indices[0] if word_pred_indices else 0
                        except (IndexError, RuntimeError) as e:
                            print(f"Warning: Error processing word logits: {e}")
                            avg_logits = confidence_data['logits'][0] if 'logits' in confidence_data and len(confidence_data['logits']) > 0 else []
                            predicted_idx = 0
                        
                        aligned_confidence_data.append({
                            'logits': avg_logits,
                            'predicted_idx': predicted_idx
                        })
                        word_logits = []
                        word_pred_indices = []
                    
                    if label != 'O':
                        entity_type = label[2:] if label.startswith(('B-', 'I-')) else None
                        
                        if entity_type == target_entity:
                            aligned_labels.append(label)
                        else:
                            aligned_labels.append('O')
                    else:
                        aligned_labels.append('O')
                    previous_word_id = word_id
                
                # Collect logits and predictions for this word
                if word_id is not None:
                    word_logits.append(logits[i])
                    word_pred_indices.append(predictions[0][i].item())
            
            # Handle the last word
            if word_logits:
                try:
                    avg_logits = torch.stack(word_logits).mean(dim=0).cpu().numpy()
                    predicted_idx = word_pred_indices[0] if word_pred_indices else 0
                except (IndexError, RuntimeError) as e:
                    print(f"Warning: Error processing final word logits: {e}")
                    avg_logits = confidence_data['logits'][0] if 'logits' in confidence_data and len(confidence_data['logits']) > 0 else []
                    predicted_idx = 0
                
                aligned_confidence_data.append({
                    'logits': avg_logits,
                    'predicted_idx': predicted_idx
                })
            
            while len(aligned_labels) < len(tokens):
                aligned_labels.append('O')
                # Safe fallback for confidence data
                fallback_conf = {}
                if 'logits' in confidence_data and len(confidence_data['logits']) > 0:
                    fallback_conf = {'logits': confidence_data['logits'][0], 'predicted_idx': 0}
                aligned_confidence_data.append(fallback_conf)
            
            aligned_labels = aligned_labels[:len(tokens)]
            aligned_confidence_data = aligned_confidence_data[:len(tokens)]
            
            return aligned_labels, aligned_confidence_data
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def perform_ner_causal_lm_all_entities(self, text: str, tokens: List[str], model_name: str) -> Tuple[dict, Dict]:
        """Perform NER with causal LM and return ALL entities."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        examples = None
        if self.faiss_index is not None:
            examples, _ = self.get_similar_examples(text, top_k=5)
        
        response = self.perform_ner_llama(model, tokenizer, text, max_length=1500, examples=examples)
        print(f"qwen response is {response}")
        
        # Calculate confidence based on response quality
        confidence_data = {'response_quality': 0.5, 'temperature': 0.7}
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                entities = result.get('entities', [])
                confidence_data['response_quality'] = 0.8  # Good JSON response
            else:
                entities = []
                confidence_data['response_quality'] = 0.3  # Poor JSON response
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response from {model_name}")
            entities = []
            confidence_data['response_quality'] = 0.2  # Parsing error
        
        entity_dict = {}
        model_entities = self.model_configs[model_name].get('entities', [])
        
        for entity_type in model_entities:
            entity_dict[entity_type] = ['O'] * len(tokens)
        
        for entity in entities:
            entity_label = entity.get('label', '')
            entity_text = entity.get('text', '')
            
            if not entity_text:
                continue
            
            matched_entity_type = None
            for entity_type in model_entities:
                # Use exact match or word boundary matching to avoid false positives
                if (entity_type.lower() == entity_label.lower() or 
                    f"{entity_type.lower()}" in entity_label.lower().split() or
                    entity_type.lower() in entity_label.lower() and 
                    (entity_label.lower().startswith(entity_type.lower()) or 
                     entity_label.lower().endswith(entity_type.lower()))):
                    matched_entity_type = entity_type
                    break
            
            if not matched_entity_type:
                continue
            
            entity_words = entity_text.split()
            for i in range(len(tokens) - len(entity_words) + 1):
                if tokens[i:i+len(entity_words)] == entity_words or \
                ' '.join(tokens[i:i+len(entity_words)]).lower() == entity_text.lower():
                    entity_dict[matched_entity_type][i] = f"B-{matched_entity_type}"
                    for j in range(i+1, i+len(entity_words)):
                        entity_dict[matched_entity_type][j] = f"I-{matched_entity_type}"
                    break
        
        return entity_dict, confidence_data

    def filter_entities_by_target(self, all_entities_result: dict, target_entity: str, tokens: List[str]) -> List[str]:
        """Filter cached causal_lm results to extract only the target entity."""
        if all_entities_result is None or not all_entities_result:
            return ['O'] * len(tokens)
        
        if target_entity in all_entities_result:
            return all_entities_result[target_entity]
        
        return ['O'] * len(tokens)

    def merge_predictions_multi_entity(self, all_predictions_multi: Dict[str, Dict[str, List[str]]],
                                       tokens: List[str],
                                       all_confidence_multi: Dict[str, Dict[str, Any]]) -> List[str]:
        """Fuse multiple models per entity using confidence/precedence per-token.

        Args:
            all_predictions_multi: entity_type -> model_name -> BIO tags
            tokens: list of tokens
            all_confidence_multi: entity_type -> model_name -> confidence info (list per-token or dict)

        Returns:
            Final merged BIO tags across all entities and models.
        """
        if not hasattr(self, '_entity_merger'):
            self._entity_merger = ConfidenceBasedMerger(self.model_configs, self.model_precedence, use_confidence=False)

        final_labels = ['O'] * len(tokens)
        filled_positions = {}

        for entity_type, model_to_preds in all_predictions_multi.items():
            for model_name, preds in model_to_preds.items():
                if len(preds) != len(tokens):
                    print(f"Warning: Skipping {entity_type}/{model_name}: prediction length mismatch ({len(preds)} vs {len(tokens)})")
                    continue

                confidence_data = all_confidence_multi.get(entity_type, {}).get(model_name, {})

                for i, tag in enumerate(preds):
                    if tag == 'O':
                        continue

                    token_conf = confidence_data[i] if isinstance(confidence_data, list) and i < len(confidence_data) else confidence_data
                    new_conf = self._entity_merger.calculate_confidence_score(tag, model_name, token_conf)

                    if final_labels[i] == 'O':
                        final_labels[i] = tag
                        filled_positions[i] = (entity_type, model_name, new_conf)
                        continue

                    existing_entity, existing_model, existing_conf = filled_positions.get(i, ('', '', 0.0))
                    winner = self._entity_merger.resolve_conflict(existing_model, model_name, i, tokens[i], existing_conf, new_conf)
                    if winner == model_name:
                        final_labels[i] = tag
                        filled_positions[i] = (entity_type, model_name, new_conf)

        # BIO integrity
        final_labels = ConfidenceBasedMerger(self.model_configs, self.model_precedence, use_confidence=False)._validate_bio_sequence(final_labels, [])
        return final_labels

    def clear_cache(self):
        """Clear the NER cache."""
        self.ner_cache.clear()
    
    def improved_merge_predictions(self, all_predictions: Dict[str, List[str]], 
                               tokens: List[str], confidence_data: Dict = None, 
                               model_assignments: Dict = None) -> List[str]:
        """Merge predictions using confidence-based or precedence-based conflict resolution."""
        LOGGERS['merger'].info(f"Starting merge with {len(all_predictions)} entity types")
        LOGGERS['merger'].debug(f"Entity types: {list(all_predictions.keys())}")
        LOGGERS['merger'].debug(f"Model assignments: {model_assignments}")
        
        if not hasattr(self, '_entity_merger'):
            self._entity_merger = ConfidenceBasedMerger(self.model_configs, self.model_precedence, use_confidence=False)
            LOGGERS['merger'].info("Created new ConfidenceBasedMerger instance")
        
        # Use provided model assignments or fall back to finding them
        if model_assignments is None:
            model_assignments = {}
            for entity_type in all_predictions.keys():
                # Find the correct model for this entity type
                assigned_model = None
                for model_name, config in self.model_configs.items():
                    if entity_type in config.get('entities', []):
                        assigned_model = model_name
                        break
                
                if assigned_model:
                    model_assignments[entity_type] = assigned_model
                else:
                    print(f"Warning: No model found for entity type '{entity_type}'")
        
        result = self._entity_merger.merge_predictions_entity_specific(
            all_predictions, tokens, model_assignments, confidence_data
        )
        
        LOGGERS['merger'].info(f"Merge completed: {result['conflict_count']} conflicts, {result['confidence_overrides']} confidence overrides, {result['precedence_overrides']} precedence overrides")
        
        if result['conflict_count'] > 0:
            print(f"\n⚠️  {result['conflict_count']} conflicts detected during merging!")
            if self._entity_merger.use_confidence:
                print(f"    {result['confidence_overrides']} resolved using confidence scores")
            print(f"    {result['precedence_overrides']} resolved using precedence order")
        
        LOGGERS['merger'].debug(f"Final merged labels: {result['labels'][:10]}... (showing first 10)")
        return result['labels']
        
    def improved_predict(self, text: str, tokens: List[str] = None, language: str = 'en') -> Dict:
        """Improved predict method with sentence/document level processing."""
        LOGGERS['main'].info(f"Starting prediction for language='{language}', text_length={len(text)}")
        
        if tokens is None:
            tokens = text.split()
        
        LOGGERS['main'].debug(f"Text: '{text[:100]}...', Tokens: {len(tokens)}")
        
        if not hasattr(self, '_ml_processor'):
            self._ml_processor = MultiLevelNERProcessor(self)
        
        all_predictions = {}
        all_confidence_data = {}
        # For multi-model-per-entity fusion
        all_predictions_multi = defaultdict(dict)  # entity_type -> model_name -> labels
        all_confidence_multi = defaultdict(dict)   # entity_type -> model_name -> confidence_data
        model_assignments = {}  # Track which model produced each entity type
        processing_stats = {
            'sentence_level': [],
            'document_level': []
        }
        
        for model_name, config in self.model_configs.items():
            assigned_entities = config.get('entities', [])
            LOGGERS['main'].debug(f"Processing model {model_name}: entities={assigned_entities}, languages={config.get('languages', [])}")
            
            if language not in config.get('languages', []):
                print(f"Skipping {model_name}: language '{language}' not supported")
                LOGGERS['main'].info(f"Skipping {model_name}: language '{language}' not supported")
                continue
            
            if config['type'] == 'api':
                model = None
                tokenizer = None
            else:
                model = self.models.get(model_name)
                tokenizer = self.tokenizers.get(model_name)
                
                if model is None or tokenizer is None:
                    print(f"Warning: Model or tokenizer not loaded for {model_name}")
                    continue
            
            for entity_type in assigned_entities:
                if entity_type in self.language_overrides:
                    if language in self.language_overrides[entity_type]:
                        override_model = self.language_overrides[entity_type][language]
                        if override_model != model_name:
                            print(f"Skipping {entity_type} for {model_name}: "
                                f"overridden by {override_model} for language '{language}'")
                            LOGGERS['main'].info(f"Skipping {entity_type} for {model_name}: overridden by {override_model} for language '{language}'")
                            continue
                
                print(f"Processing {entity_type} with {model_name} ({config['type']})")
                LOGGERS['main'].info(f"Processing {entity_type} with {model_name} ({config['type']})")
                
                try:
                    predictions, confidence_data = self._ml_processor.predict_with_granularity(
                        text, tokens, model_name, config['type'], entity_type
                    )
                    
                    LOGGERS['main'].debug(f"{entity_type} predictions: {predictions[:10]}... (showing first 10)")
                    LOGGERS['main'].debug(f"{entity_type} confidence data type: {type(confidence_data)}")
                    
                    # Store single-assignment view (legacy path)
                    if entity_type not in all_predictions:
                        all_predictions[entity_type] = predictions
                        all_confidence_data[entity_type] = confidence_data
                        model_assignments[entity_type] = model_name
                    # Always store multi-model view
                    all_predictions_multi[entity_type][model_name] = predictions
                    all_confidence_multi[entity_type][model_name] = confidence_data
                    
                    if config['type'] == 'token_classification':
                        processing_stats['sentence_level'].append(entity_type)
                        LOGGERS['main'].debug(f"{entity_type} processed at sentence level")
                    else:
                        processing_stats['document_level'].append(entity_type)
                        LOGGERS['main'].debug(f"{entity_type} processed at document level")
                        
                except Exception as e:
                    print(f"Error processing {entity_type} with {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        LOGGERS['main'].info(f"Starting merge with {len(all_predictions_multi)} entity types: {list(all_predictions_multi.keys())}")
        if getattr(self, 'enable_multi_model_fusion', False):
            final_predictions = self.merge_predictions_multi_entity(all_predictions_multi, tokens, all_confidence_multi)
        else:
            LOGGERS['main'].debug(f"Model assignments: {model_assignments}")
            final_predictions = self.improved_merge_predictions(all_predictions, tokens, all_confidence_data, model_assignments)
        
        LOGGERS['main'].info(f"Final predictions length: {len(final_predictions)}")
        LOGGERS['main'].debug(f"Final predictions: {final_predictions[:10]}... (showing first 10)")
        
        print("\n" + "-"*60)
        print("PROCESSING SUMMARY")
        print(f"Sentence-level: {', '.join(processing_stats['sentence_level']) or 'None'}")
        print(f"Document-level: {', '.join(processing_stats['document_level']) or 'None'}")
        print("-"*60)
        
        # Per-entity diagnostics when multi-model fusion is enabled
        if getattr(self, 'enable_multi_model_fusion', False):
            diagnostics = {}
            for entity_type, model_to_preds in all_predictions_multi.items():
                non_o_by_model = {}
                for model_name, preds in model_to_preds.items():
                    non_o_by_model[model_name] = sum(1 for t in preds if t != 'O')
                diagnostics[entity_type] = {
                    'non_o_by_model': non_o_by_model,
                    'models': list(model_to_preds.keys())
                }
            LOGGERS['main'].info(f"Per-entity non-O counts by model: {diagnostics}")
        
        return {
            'tokens': tokens,
            'labels': final_predictions,
            'entity_predictions': all_predictions_multi if getattr(self, 'enable_multi_model_fusion', False) else all_predictions,
            'confidence_data': all_confidence_multi if getattr(self, 'enable_multi_model_fusion', False) else all_confidence_data,
            'processing_stats': processing_stats
        }
    
    def predict_dataset(self, dataset: Dataset, language: str = 'en') -> List[Dict]:
        """Perform predictions on a Hugging Face Dataset."""
        LOGGERS['main'].info(f"Starting dataset prediction: {len(dataset)} examples, language='{language}'")
        results = []
        
        for i, example in enumerate(dataset):
            print(f"\nProcessing example {i+1}/{len(dataset)}")
            LOGGERS['main'].info(f"Processing example {i+1}/{len(dataset)}")
            
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
                LOGGERS['main'].debug(f"Example {i}: gold labels length: {len(gold_tags)}")
            
            LOGGERS['main'].debug(f"Example {i}: predicted labels length: {len(result.get('labels', []))}")
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
    LOGGERS['main'].info("Starting EnsembleModels_ver5 main execution")
    
    global_label_list = ["O","B-cropSpecies","I-cropSpecies","B-cropVariety","I-cropVariety","B-locationName","I-locationName","B-startTime","I-startTime","B-endTime","I-endTime","B-duration","I-duration","B-soilDepth","I-soilDepth","B-soilReferenceGroup","I-soilReferenceGroup","B-soilOrganicCarbon","I-soilOrganicCarbon","B-soilTexture","I-soilTexture","B-soilBulkDensity","I-soilBulkDensity","B-soilAvailableNitrogen","I-soilAvailableNitrogen","B-soilPH","I-soilPH"]
    
    LOGGERS['main'].info(f"Global label list: {len(global_label_list)} labels")
    
    # Define model precedence order (highest to lowest priority)
    # Models listed first will override models listed later in case of conflicts
    model_precedence = [
        'roberta_all_specific',
        'roberta_english_specific',
        'agribert_all_specific',
        'xlm_roberta_broad',
        'deepseek',
        'qwen_2.5_7b',
    ]
    
    # Initialize multi-model NER with precedence (confidence-based system enabled by default)
    LOGGERS['main'].info("Initializing MultiModelNER with confidence-based precedence")
    ner = MultiModelNER(
        global_label_list,
        model_precedence=model_precedence,
        enable_api=False,
        enable_causal_lm=False,
        enable_multi_model_fusion=False,
    )
    
    print("\n" + "="*60)
    print("CONFIDENCE-BASED PRECEDENCE SYSTEM")
    print("="*60)
    print("✓ Confidence scoring enabled for all model types")
    print("✓ Token classification models: Using softmax probabilities from logits")
    print("✓ Causal LM models: Using response quality assessment")
    print("✓ API models: Using response parsing success rate")
    print("✓ Fallback to fixed precedence when confidence scores are similar")
    print("="*60)
    
    LOGGERS['main'].info("Confidence-based precedence system initialized")
    
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
    LOGGERS['main'].info("Starting German dataset evaluation")
    
    test_dataset_de = ner.load_hf_dataset(ner.test_dataset_de_path)
    LOGGERS['main'].info(f"Loaded German dataset: {len(test_dataset_de)} examples")
    
    de_results = ner.predict_dataset(test_dataset_de, language='de')
    LOGGERS['main'].info(f"German prediction completed: {len(de_results)} results")
    
    ner.save_predictions(de_results, 'predictions_german_multimodel_precedence.conll')
    metrics_de = ner.calculate_f1_scores(de_results, output_file='f1_scores_german_multimodel_precedence_ver5.txt')
    
    LOGGERS['main'].info(f"German F1 Score: {metrics_de.get('f1_score', 0):.4f}")
    
    # Load and predict on English test dataset
    print("\n" + "="*60)
    print("ENGLISH TEST DATASET EVALUATION")
    print("="*60)
    LOGGERS['main'].info("Starting English dataset evaluation")
    
    test_dataset_en = ner.load_hf_dataset(ner.test_dataset_en_path)
    LOGGERS['main'].info(f"Loaded English dataset: {len(test_dataset_en)} examples")
    
    en_results = ner.predict_dataset(test_dataset_en, language='en')
    LOGGERS['main'].info(f"English prediction completed: {len(en_results)} results")
    
    ner.save_predictions(en_results, 'predictions_english_multimodel_precedence.conll')
    metrics_en = ner.calculate_f1_scores(en_results, output_file='f1_scores_english_multimodel_precedence_ver5.txt')
    
    LOGGERS['main'].info(f"English F1 Score: {metrics_en.get('f1_score', 0):.4f}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"German F1 Score:  {metrics_de.get('f1_score', 0):.4f}")
    print(f"English F1 Score: {metrics_en.get('f1_score', 0):.4f}")
    
    LOGGERS['main'].info("EnsembleModels_ver5 execution completed successfully")


if __name__ == "__main__":
    main()
