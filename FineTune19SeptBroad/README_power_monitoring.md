# Power Consumption Monitoring for CPU Predictions

This directory contains scripts for monitoring power consumption during CPU-based model predictions with your fine-tuned AgriBERT model.

## Files Overview

- `power_monitor.py` - Core power monitoring functionality
- `cpu_prediction_with_power_monitor.py` - Standalone prediction script with power monitoring
- `example_power_monitoring.py` - Example usage script
- `README_power_monitoring.md` - This documentation

## Quick Start

### 1. Basic Usage

```bash
# Run predictions with power monitoring using your trained model
python cpu_prediction_with_power_monitor.py --model_path /path/to/your/trained/model

# Use custom texts from a JSON file
python cpu_prediction_with_power_monitor.py --model_path /path/to/your/trained/model --texts_file custom_texts.json

# Limit to 5 sample texts for quick testing
python cpu_prediction_with_power_monitor.py --model_path /path/to/your/trained/model --num_samples 5
```

### 2. Example Usage

```bash
# Run the example script
python example_power_monitoring.py

# Create sample texts file
python example_power_monitoring.py create_samples
```

## Features

### Power Monitoring
- **Real-time power consumption tracking** using system power sensors
- **Energy calculation** in Wh (Watt-hours) and kWh
- **CPU usage monitoring** alongside power consumption
- **Memory usage tracking** for comprehensive system analysis

### Carbon Footprint Estimation
- **CO2 emissions calculation** based on energy consumption
- **Environmental impact comparison** (trees needed to offset, car travel equivalent)
- **Configurable carbon intensity** for different regions

### Detailed Reporting
- **Comprehensive metrics** including average, peak, and minimum power consumption
- **JSON export** of all results for further analysis
- **Formatted console output** with summary statistics

## Usage Examples

### 1. Monitor Power During Predictions

```python
from cpu_prediction_with_power_monitor import CPUPredictionWithPowerMonitor

# Initialize predictor
predictor = CPUPredictionWithPowerMonitor("/path/to/your/model")
predictor.load_model()

# Run predictions with power monitoring
texts = ["Your agriculture text here..."]
results = predictor.predict_with_power_monitoring(texts)

# Access results
print(f"Total energy: {results['power_consumption']['total_energy_wh']:.4f} Wh")
print(f"Average power: {results['power_consumption']['average_power_watts']:.2f} W")
```

### 2. Context Manager for Custom Code

```python
from power_monitor import power_monitor

# Monitor power during any CPU-intensive operation
with power_monitor() as monitor:
    # Your CPU-intensive code here
    result = your_model.predict(data)
    
# Results are automatically available
monitor.print_results()
```

### 3. Carbon Footprint Analysis

```python
from power_monitor import estimate_carbon_footprint

# Calculate carbon footprint
energy_kwh = 0.001  # 1 Wh = 0.001 kWh
carbon_footprint = estimate_carbon_footprint(energy_kwh)

print(f"CO2 emissions: {carbon_footprint['co2_g']:.2f} g")
print(f"Trees needed: {carbon_footprint['trees_needed_for_offset']:.4f}")
```

## Output Format

The scripts generate comprehensive reports including:

### Power Consumption Metrics
- Total energy consumption (Wh, kWh)
- Average, peak, and minimum power (W)
- Duration of monitoring
- Number of measurements taken

### System Performance
- CPU usage statistics
- Memory usage statistics
- Prediction timing per text

### Environmental Impact
- CO2 emissions in grams
- Equivalent car travel distance
- Trees needed to offset emissions

## Requirements

- Python 3.7+
- torch
- transformers
- psutil
- numpy
- datasets

## Installation

```bash
pip install torch transformers psutil numpy datasets
```

## Notes

### Power Monitoring Accuracy
- The script attempts to read from RAPL (Running Average Power Limit) sensors for accurate power measurement
- Falls back to estimation based on CPU usage if hardware sensors are unavailable
- Results may vary depending on system configuration and available sensors

### CPU vs GPU Considerations
- This monitoring is specifically designed for CPU inference
- For GPU power monitoring, different tools would be needed (e.g., nvidia-smi)
- CPU inference typically has lower power consumption but longer processing times

### System Requirements
- Linux systems with RAPL support provide the most accurate measurements
- The script works on other systems but with less accurate power estimation
- Requires appropriate permissions to read system power information

## Troubleshooting

### Common Issues

1. **Permission denied errors**: Run with appropriate permissions to read system files
2. **Model loading errors**: Ensure the model path is correct and the model is compatible
3. **Power reading errors**: The script will fall back to estimation if hardware sensors are unavailable

### Debug Mode

Enable verbose output by modifying the monitoring interval:

```python
results = predictor.predict_with_power_monitoring(
    texts=texts,
    monitoring_interval=0.05,  # More frequent monitoring
    save_results=True
)
```

## Example Output

```
================================================================================
CPU PREDICTION WITH POWER MONITORING - SUMMARY
================================================================================
Model: /path/to/your/model
Device: cpu
Texts processed: 10
Total prediction time: 45.23s
Average time per text: 4.52s

Power Consumption:
  Duration: 45.23s (0.75 min)
  Total Energy: 0.1234 Wh
  Average Power: 9.82 W
  Peak Power: 15.67 W

Carbon Footprint:
  CO2 emissions: 0.06 g
  Trees needed to offset: 0.0003
================================================================================
```

This monitoring system helps you understand the computational cost and environmental impact of running your trained models on CPU hardware.

