#!/usr/bin/env python3
"""
CPU-based Model Prediction with Power Consumption Monitoring

This script loads a trained model and runs predictions on CPU while monitoring
power consumption. It's designed to work with the fine-tuned AgriBERT model.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from datasets import Dataset
from power_monitor import power_monitor, estimate_carbon_footprint
import argparse
from typing import List, Dict, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CPUPredictionWithPowerMonitor:
    """CPU-based prediction with power monitoring capabilities."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the prediction system.
        
        Args:
            model_path: Path to the saved model
            device: Device to run on (should be "cpu" for power monitoring)
        """
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model for CPU inference
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None  # Let PyTorch handle device placement
            )
            
            # Move model to CPU explicitly
            self.model.to(self.device)
            self.model.eval()
            
            # Create NER pipeline
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=-1  # Force CPU usage
            )
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single_text(self, text: str) -> Dict[str, Any]:
        """
        Predict entities in a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Run prediction
            predictions = self.pipeline(text)
            
            prediction_time = time.time() - start_time
            
            return {
                'text': text,
                'predictions': predictions,
                'prediction_time_seconds': prediction_time,
                'num_tokens': len(text.split()),
                'num_entities': len(predictions)
            }
            
        except Exception as e:
            return {
                'text': text,
                'error': str(e),
                'prediction_time_seconds': time.time() - start_time
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict entities in a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}...")
            result = self.predict_single_text(text)
            results.append(result)
        return results
    
    def predict_with_power_monitoring(self, texts: List[str], 
                                    monitoring_interval: float = 0.1,
                                    save_results: bool = True) -> Dict[str, Any]:
        """
        Run predictions with power consumption monitoring.
        
        Args:
            texts: List of input texts
            monitoring_interval: Power monitoring interval in seconds
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with predictions and power consumption data
        """
        print(f"Starting prediction with power monitoring for {len(texts)} texts...")
        print(f"Monitoring interval: {monitoring_interval}s")
        
        # Start power monitoring
        with power_monitor(monitoring_interval) as monitor:
            # Run predictions
            start_time = time.time()
            predictions = self.predict_batch(texts)
            total_time = time.time() - start_time
        
        # Get power monitoring results
        power_results = monitor.get_results()
        
        # Calculate carbon footprint
        carbon_footprint = estimate_carbon_footprint(power_results['total_energy_kwh'])
        
        # Compile comprehensive results
        results = {
            'model_info': {
                'model_path': self.model_path,
                'device': self.device,
                'model_type': 'AgriBERT-NER'
            },
            'prediction_info': {
                'num_texts': len(texts),
                'total_prediction_time_seconds': total_time,
                'average_prediction_time_per_text': total_time / len(texts),
                'predictions': predictions
            },
            'power_consumption': power_results,
            'carbon_footprint': carbon_footprint,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print summary
        self._print_summary(results)
        
        # Save results if requested
        if save_results:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"cpu_prediction_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("CPU PREDICTION WITH POWER MONITORING - SUMMARY")
        print("="*80)
        
        # Model info
        print(f"Model: {results['model_info']['model_path']}")
        print(f"Device: {results['model_info']['device']}")
        
        # Prediction info
        pred_info = results['prediction_info']
        print(f"Texts processed: {pred_info['num_texts']}")
        print(f"Total prediction time: {pred_info['total_prediction_time_seconds']:.2f}s")
        print(f"Average time per text: {pred_info['average_prediction_time_per_text']:.3f}s")
        
        # Power consumption
        power = results['power_consumption']
        print(f"\nPower Consumption:")
        print(f"  Duration: {power['duration_seconds']:.2f}s ({power['duration_minutes']:.2f} min)")
        print(f"  Total Energy: {power['total_energy_wh']:.4f} Wh")
        print(f"  Average Power: {power['average_power_watts']:.2f} W")
        print(f"  Peak Power: {power['peak_power_watts']:.2f} W")
        
        # Carbon footprint
        carbon = results['carbon_footprint']
        print(f"\nCarbon Footprint:")
        print(f"  CO2 emissions: {carbon['co2_g']:.2f} g")
        print(f"  Trees needed to offset: {carbon['trees_needed_for_offset']:.4f}")
        
        print("="*80)


def load_sample_texts(file_path: str = None) -> List[str]:
    """
    Load sample texts for testing.
    
    Args:
        file_path: Path to JSON file with texts, or None for default samples
        
    Returns:
        List of sample texts
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item['text'] if isinstance(item, dict) else str(item) for item in data]
            else:
                return [str(data)]
    
    # Default sample texts for agriculture NER
    sample_texts = [
        "The wheat crop was planted in March and harvested in July in the northern region of France.",
        "Farmers in California are preparing for the spring planting season starting in April.",
        "The corn yield in Iowa increased by 15% compared to last year's harvest in September.",
        "Organic farming practices have been implemented on the 50-acre farm in Oregon since 2020.",
        "The rice fields in the Sacramento Valley require irrigation during the dry summer months.",
        "Tomato production in greenhouse facilities has expanded to meet year-round demand.",
        "The dairy farm in Wisconsin processes 1000 gallons of milk daily from Holstein cows.",
        "Sustainable agriculture techniques are being adopted by farmers across the Midwest region.",
        "The apple orchard in Washington state expects a bumper crop this fall season.",
        "Precision agriculture using GPS technology is improving crop yields in Nebraska."
    ]
    
    return sample_texts


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='CPU Prediction with Power Monitoring')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model directory')
    parser.add_argument('--texts_file', type=str, default=None,
                       help='JSON file containing texts to analyze')
    parser.add_argument('--monitoring_interval', type=float, default=0.1,
                       help='Power monitoring interval in seconds (default: 0.1)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of sample texts to use (if no texts_file provided)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        return 1
    
    # Load texts
    if args.texts_file:
        texts = load_sample_texts(args.texts_file)
    else:
        texts = load_sample_texts()
        if args.num_samples:
            texts = texts[:args.num_samples]
    
    print(f"Loaded {len(texts)} texts for prediction")
    
    # Initialize predictor
    predictor = CPUPredictionWithPowerMonitor(args.model_path)
    
    # Load model
    if not predictor.load_model():
        print("Failed to load model!")
        return 1
    
    # Run predictions with power monitoring
    try:
        results = predictor.predict_with_power_monitoring(
            texts=texts,
            monitoring_interval=args.monitoring_interval,
            save_results=not args.no_save
        )
        
        print("\nPrediction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("CPU Prediction with Power Monitoring")
        print("Usage examples:")
        print("  python cpu_prediction_with_power_monitor.py --model_path /path/to/model")
        print("  python cpu_prediction_with_power_monitor.py --model_path /path/to/model --texts_file texts.json")
        print("  python cpu_prediction_with_power_monitor.py --model_path /path/to/model --num_samples 5")
        print("\nFor help: python cpu_prediction_with_power_monitor.py --help")
    else:
        sys.exit(main())

