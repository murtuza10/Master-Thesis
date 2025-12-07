#!/usr/bin/env python3
"""
Test script for power monitoring functionality.

This script tests the power monitoring system and shows what capabilities
are available on the current system.
"""

import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from power_monitor import power_monitor, PowerMonitor, estimate_carbon_footprint

def test_power_monitoring():
    """Test the power monitoring functionality."""
    
    print("="*80)
    print("POWER MONITORING SYSTEM TEST")
    print("="*80)
    
    # Test 1: Check system capabilities
    print("\n1. Checking system power monitoring capabilities...")
    monitor = PowerMonitor()
    
    capabilities = monitor.power_capabilities
    print(f"RAPL available: {capabilities['rapl_available']}")
    print(f"Power supply available: {capabilities['power_supply_available']}")
    print(f"Estimation mode: {capabilities['estimation_mode']}")
    
    if capabilities['warnings']:
        print("\nWarnings:")
        for warning in capabilities['warnings']:
            print(f"  ⚠️  {warning}")
    
    # Test 2: Short monitoring test
    print("\n2. Running 5-second power monitoring test...")
    print("This will simulate CPU work and monitor power consumption...")
    
    with power_monitor(monitoring_interval=0.2) as monitor:
        # Simulate CPU-intensive work
        start_time = time.time()
        while time.time() - start_time < 5:
            # Simulate some CPU work
            sum(range(10000))
            time.sleep(0.1)
    
    # Test 3: Display results
    print("\n3. Power monitoring results:")
    monitor.print_results()
    
    # Test 4: Carbon footprint calculation
    results = monitor.get_results()
    if results:
        print("\n4. Environmental impact analysis:")
        carbon_footprint = estimate_carbon_footprint(results['total_energy_kwh'])
        print(f"CO2 emissions: {carbon_footprint['co2_g']:.2f} g")
        print(f"Trees needed to offset: {carbon_footprint['trees_needed_for_offset']:.4f}")
        print(f"Equivalent car travel: {carbon_footprint['car_km_equivalent']:.2f} km")
    
    # Test 5: Save results
    print("\n5. Saving test results...")
    monitor.save_results("power_monitoring_test_results.json")
    print("Results saved to power_monitoring_test_results.json")
    
    print("\n" + "="*80)
    print("POWER MONITORING TEST COMPLETED")
    print("="*80)
    
    return results

def check_system_info():
    """Check system information for power monitoring."""
    import psutil
    
    print("\nSYSTEM INFORMATION:")
    print(f"CPU count: {psutil.cpu_count()}")
    print(f"CPU frequency: {psutil.cpu_freq()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check for power monitoring files
    print("\nPOWER MONITORING FILES:")
    rapl_paths = [
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0:1/energy_uj",
    ]
    
    for path in rapl_paths:
        exists = os.path.exists(path)
        readable = os.access(path, os.R_OK) if exists else False
        print(f"  {path}: {'✅' if readable else '❌' if exists else 'N/A'}")
    
    power_paths = [
        "/sys/class/power_supply/BAT0/power_now",
        "/sys/class/power_supply/AC/power_now",
    ]
    
    for path in power_paths:
        exists = os.path.exists(path)
        readable = os.access(path, os.R_OK) if exists else False
        print(f"  {path}: {'✅' if readable else '❌' if exists else 'N/A'}")

if __name__ == "__main__":
    print("Starting power monitoring system test...")
    
    # Check system info first
    check_system_info()
    
    # Run the test
    try:
        results = test_power_monitoring()
        
        if results:
            print("\n✅ Power monitoring test completed successfully!")
            print("The system is ready for power monitoring during model evaluation.")
        else:
            print("\n❌ Power monitoring test failed!")
            
    except Exception as e:
        print(f"\n❌ Error during power monitoring test: {e}")
        import traceback
        traceback.print_exc()

