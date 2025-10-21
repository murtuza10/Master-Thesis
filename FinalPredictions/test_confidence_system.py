#!/usr/bin/env python3
"""
Test script to demonstrate confidence-based precedence system.
This script shows how the new system resolves conflicts using confidence scores.
"""

import sys
import os
sys.path.append('/home/s27mhusa_hpc/Master-Thesis/FinalPredictions')

# Import from the copy file with proper module name
sys.path.insert(0, '/home/s27mhusa_hpc/Master-Thesis/FinalPredictions')
exec(open('/home/s27mhusa_hpc/Master-Thesis/FinalPredictions/EnsembleModels_ver4 copy.py').read())

def test_confidence_system():
    """Test the confidence-based precedence system with sample data."""
    
    # Sample model configurations
    model_configs = {
        'roberta_test': {
            'type': 'token_classification',
            'entities': ['cropSpecies'],
            'languages': ['en']
        },
        'qwen_test': {
            'type': 'causal_lm',
            'entities': ['cropSpecies'],
            'languages': ['en']
        }
    }
    
    # Initialize confidence-based merger
    merger = ConfidenceBasedMerger(model_configs, use_confidence=True)
    
    print("="*60)
    print("CONFIDENCE-BASED PRECEDENCE SYSTEM TEST")
    print("="*60)
    
    # Test confidence calculation for different model types
    print("\n1. Testing confidence calculation:")
    
    # Token classification confidence
    token_conf_data = {
        'logits': [0.1, 0.9, 0.05],  # High confidence for class 1
        'predicted_idx': 1
    }
    conf1 = merger.calculate_confidence_score('B-cropSpecies', 'roberta_test', token_conf_data)
    print(f"   Token classification confidence: {conf1:.3f}")
    
    # Causal LM confidence
    causal_conf_data = {
        'response_quality': 0.8,
        'temperature': 0.7
    }
    conf2 = merger.calculate_confidence_score('B-cropSpecies', 'qwen_test', causal_conf_data)
    print(f"   Causal LM confidence: {conf2:.3f}")
    
    # API confidence
    api_conf_data = {'response_quality': 0.6}
    conf3 = merger.calculate_confidence_score('B-cropSpecies', 'api_model', api_conf_data)
    print(f"   API confidence: {conf3:.3f}")
    
    print("\n2. Testing conflict resolution:")
    
    # Simulate a conflict scenario
    existing_model = 'roberta_test'
    new_model = 'qwen_test'
    position = 5
    token = 'wheat'
    
    # High confidence vs low confidence
    winner1 = merger.resolve_conflict(
        existing_model, new_model, position, token, 
        existing_confidence=0.3, new_confidence=0.8
    )
    print(f"   High confidence ({0.8:.1f}) vs Low confidence ({0.3:.1f}): {winner1} wins")
    
    # Similar confidence (should use precedence)
    winner2 = merger.resolve_conflict(
        existing_model, new_model, position, token,
        existing_confidence=0.7, new_confidence=0.75
    )
    print(f"   Similar confidence ({0.7:.1f} vs {0.75:.1f}): {winner2} wins (precedence)")
    
    print("\n3. Testing merge predictions with confidence:")
    
    # Sample predictions with confidence data
    all_predictions = {
        'cropSpecies': ['O', 'B-cropSpecies', 'I-cropSpecies', 'O', 'O']
    }
    
    confidence_data = {
        'cropSpecies': [
            {},  # O tag - neutral confidence
            {'logits': [0.1, 0.9, 0.05], 'predicted_idx': 1},  # High confidence B-tag
            {'logits': [0.05, 0.95, 0.0], 'predicted_idx': 1},  # High confidence I-tag
            {},  # O tag
            {}   # O tag
        ]
    }
    
    model_assignments = {'cropSpecies': 'roberta_test'}
    tokens = ['The', 'wheat', 'crop', 'is', 'healthy']
    
    result = merger.merge_predictions_entity_specific(
        all_predictions, tokens, model_assignments, confidence_data
    )
    
    print(f"   Final labels: {result['labels']}")
    print(f"   Conflicts resolved: {result['conflict_count']}")
    print(f"   Confidence overrides: {result['confidence_overrides']}")
    print(f"   Precedence overrides: {result['precedence_overrides']}")
    
    print("\n" + "="*60)
    print("BENEFITS OF CONFIDENCE-BASED PRECEDENCE:")
    print("="*60)
    print("✓ Dynamic decision making based on model certainty")
    print("✓ Better handling of context-dependent predictions")
    print("✓ Reduced reliance on fixed precedence order")
    print("✓ Potential for higher F1 scores through better conflict resolution")
    print("✓ Maintains fallback to precedence for similar confidence scores")
    print("="*60)

if __name__ == "__main__":
    test_confidence_system()
