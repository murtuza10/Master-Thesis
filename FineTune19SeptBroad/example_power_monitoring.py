#!/usr/bin/env python3
"""
Example script demonstrating power consumption monitoring for CPU predictions.

This script shows how to use the power monitoring tools with your trained AgriBERT model.
"""

import os
import sys
import json
from cpu_prediction_with_power_monitor import CPUPredictionWithPowerMonitor, load_sample_texts

def main():
    """Example usage of power monitoring with CPU predictions."""
    
    # Configuration
    model_path = "/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/agribert-new_final_model_regularized_saved_broad_22-3"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please update the model_path variable with the correct path to your trained model.")
        return
    
    print("="*80)
    print("CPU PREDICTION WITH POWER MONITORING - EXAMPLE")
    print("="*80)
    
    # Initialize the predictor
    print("Initializing CPU predictor...")
    predictor = CPUPredictionWithPowerMonitor(model_path)
    
    # Load the model
    print("Loading model...")
    if not predictor.load_model():
        print("Failed to load model!")
        return
    
    # Load sample texts
    print("Loading sample texts...")
    sample_texts = load_sample_texts()
    
    # You can also load from a custom file:
    # sample_texts = load_sample_texts("your_custom_texts.json")
    
    print(f"Loaded {len(sample_texts)} sample texts")
    
    # Run predictions with power monitoring
    print("\nStarting predictions with power monitoring...")
    print("This will monitor CPU power consumption during inference...")
    
    try:
        results = predictor.predict_with_power_monitoring(
            texts=sample_texts,
            monitoring_interval=0.1,  # Monitor every 0.1 seconds
            save_results=True
        )
        
        print("\n" + "="*80)
        print("PREDICTION RESULTS SUMMARY")
        print("="*80)
        
        # Show some example predictions
        print("\nExample Predictions:")
        for i, pred in enumerate(results['prediction_info']['predictions'][:3]):
            print(f"\nText {i+1}: {pred['text'][:100]}...")
            if 'predictions' in pred:
                print(f"Entities found: {len(pred['predictions'])}")
                for entity in pred['predictions'][:3]:  # Show first 3 entities
                    print(f"  - {entity['entity_group']}: {entity['word']} (confidence: {entity['score']:.3f})")
            else:
                print(f"Error: {pred.get('error', 'Unknown error')}")
        
        # Power consumption summary
        power = results['power_consumption']
        print(f"\nPower Consumption Summary:")
        print(f"  Total Energy: {power['total_energy_wh']:.4f} Wh")
        print(f"  Average Power: {power['average_power_watts']:.2f} W")
        print(f"  Duration: {power['duration_seconds']:.2f} seconds")
        
        # Carbon footprint
        carbon = results['carbon_footprint']
        print(f"\nEnvironmental Impact:")
        print(f"  CO2 emissions: {carbon['co2_g']:.2f} g")
        print(f"  Equivalent to {carbon['car_km_equivalent']:.2f} km of car travel")
        
        print("\n" + "="*80)
        print("Power monitoring completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def create_sample_texts_file():
    """Create a sample texts file for testing."""
    sample_texts = [
        {
            "text": "The wheat crop was planted in March and harvested in July in the northern region of France.",
            "expected_entities": ["wheat", "March", "July", "France"]
        },
        {
            "text": "Farmers in California are preparing for the spring planting season starting in April.",
            "expected_entities": ["California", "April"]
        },
        {
            "text": "The corn yield in Iowa increased by 15% compared to last year's harvest in September.",
            "expected_entities": ["corn", "Iowa", "September"]
        },
        {
            "text": "Organic farming practices have been implemented on the 50-acre farm in Oregon since 2020.",
            "expected_entities": ["Oregon", "2020"]
        },
        {
            "text": "The rice fields in the Sacramento Valley require irrigation during the dry summer months.",
            "expected_entities": ["rice", "Sacramento Valley"]
        }
    ]
    
    with open("sample_agriculture_texts.json", "w") as f:
        json.dump(sample_texts, f, indent=2)
    
    print("Created sample_agriculture_texts.json with example texts")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create_samples":
        create_sample_texts_file()
    else:
        main()

