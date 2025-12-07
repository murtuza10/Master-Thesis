#!/usr/bin/env python3
"""
Simple script to run model evaluation with power monitoring.

This script demonstrates how to use the power monitoring functionality
with your existing model evaluation workflow.
"""

import sys
import os
from evaluate_test_with_power_monitor import quick_evaluate_with_power_monitoring

def main():
    """Run evaluation with power monitoring."""
    
    print("="*80)
    print("MODEL EVALUATION WITH POWER MONITORING")
    print("="*80)
    print("This script will evaluate your model while monitoring CPU power consumption.")
    print("The evaluation will run on CPU to enable accurate power monitoring.")
    print("="*80)
    
    # Configuration
    output_dir = "/home/s27mhusa_hpc/Master-Thesis/FineTuneCalculateResults/ScibertAllNoSoilWithPower"
    monitoring_interval = 0.1  # Monitor every 0.1 seconds
    
    print(f"Output directory: {output_dir}")
    print(f"Monitoring interval: {monitoring_interval}s")
    print("\nStarting evaluation...")
    
    try:
        # Run evaluation with power monitoring
        exact_f1, partial_f1, saved_files, power_results = quick_evaluate_with_power_monitoring(
            save_results=True,
            output_dir=output_dir,
            monitoring_interval=monitoring_interval
        )
        
        # Print comprehensive results
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"   Exact F1 Score: {exact_f1:.4f}")
        print(f"   Partial F1 Score: {partial_f1:.4f}")
        
        print(f"\n⚡ POWER CONSUMPTION:")
        print(f"   Total Energy: {power_results['total_energy_wh']:.4f} Wh")
        print(f"   Average Power: {power_results['average_power_watts']:.2f} W")
        print(f"   Peak Power: {power_results['peak_power_watts']:.2f} W")
        print(f"   Duration: {power_results['duration_seconds']:.2f} seconds")
        print(f"   Average CPU Usage: {power_results['average_cpu_usage_percent']:.1f}%")
        
        # Calculate and display carbon footprint
        from power_monitor import estimate_carbon_footprint
        carbon_footprint = estimate_carbon_footprint(power_results['total_energy_kwh'])
        
        print(f"\n🌱 ENVIRONMENTAL IMPACT:")
        print(f"   CO2 Emissions: {carbon_footprint['co2_g']:.2f} g")
        print(f"   Trees needed to offset: {carbon_footprint['trees_needed_for_offset']:.4f}")
        print(f"   Equivalent car travel: {carbon_footprint['car_km_equivalent']:.2f} km")
        
        if saved_files:
            print(f"\n📁 RESULTS SAVED:")
            print(f"   Summary: {saved_files['summary_file']}")
            print(f"   Detailed JSON: {saved_files['json_file']}")
            print(f"   Predictions: {saved_files['predictions_file']}")
            print(f"   CSV Metrics: {saved_files['csv_file']}")
        
        print("\n" + "="*80)
        print("All results include both model performance and power consumption data.")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

