#!/usr/bin/env python3
"""
Power Consumption Monitor for CPU-based Model Predictions

This script monitors power consumption during model inference on CPU.
It provides real-time power monitoring and calculates total energy consumption.
"""

import os
import time
import psutil
import subprocess
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from contextlib import contextmanager

class PowerMonitor:
    """Monitor power consumption during model inference."""
    
    def __init__(self, monitoring_interval: float = 0.1):
        """
        Initialize power monitor.
        
        Args:
            monitoring_interval: Time interval between power measurements (seconds)
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.power_readings = []
        self.cpu_usage_readings = []
        self.memory_readings = []
        self.timestamps = []
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        
        # Check power monitoring capabilities
        self.power_capabilities = self._check_power_capabilities()
    
    def _check_power_capabilities(self) -> Dict:
        """Check what power monitoring capabilities are available on this system."""
        capabilities = {
            'rapl_available': False,
            'rapl_paths': [],
            'power_supply_available': False,
            'power_supply_paths': [],
            'estimation_mode': True,
            'warnings': []
        }
        
        # Check RAPL availability
        rapl_paths = [
            "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
            "/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj",
            "/sys/class/powercap/intel-rapl/intel-rapl:0:1/energy_uj",
        ]
        
        for path in rapl_paths:
            if os.path.exists(path):
                if os.access(path, os.R_OK):
                    capabilities['rapl_available'] = True
                    capabilities['rapl_paths'].append(path)
                else:
                    capabilities['warnings'].append(f"RAPL path exists but not readable: {path}")
        
        # Check power supply availability
        power_paths = [
            "/sys/class/power_supply/BAT0/power_now",
            "/sys/class/power_supply/AC/power_now",
        ]
        
        for path in power_paths:
            if os.path.exists(path):
                if os.access(path, os.R_OK):
                    capabilities['power_supply_available'] = True
                    capabilities['power_supply_paths'].append(path)
                else:
                    capabilities['warnings'].append(f"Power supply path exists but not readable: {path}")
        
        # Print capability summary
        if capabilities['rapl_available']:
            print("✅ RAPL power monitoring available (most accurate)")
        elif capabilities['power_supply_available']:
            print("⚠️  Using power supply monitoring (less accurate)")
        else:
            print("⚠️  Using CPU usage estimation (least accurate)")
            capabilities['warnings'].append("No hardware power sensors accessible - using estimation")
        
        return capabilities
        
    def _get_cpu_power_consumption(self) -> float:
        """Get current CPU power consumption in watts."""
        try:
            # Try to read from RAPL (Running Average Power Limit) if available
            # This is the most accurate method for Intel/AMD CPUs
            rapl_paths = [
                "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                "/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj",
                "/sys/class/powercap/intel-rapl/intel-rapl:0:1/energy_uj",
            ]
            
            for path in rapl_paths:
                if os.path.exists(path) and os.access(path, os.R_OK):
                    try:
                        with open(path, 'r') as f:
                            energy_uj = int(f.read().strip())
                        return energy_uj / 1_000_000  # Convert microjoules to joules
                    except (PermissionError, OSError, ValueError):
                        continue
            
            # Fallback: Estimate based on CPU usage and frequency
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                base_power = 15.0  # Base CPU power in watts
                dynamic_power = (cpu_percent / 100.0) * 25.0  # Dynamic power based on usage
                return base_power + dynamic_power
            else:
                # Very rough estimation
                return 20.0 + (cpu_percent / 100.0) * 30.0
                
        except Exception as e:
            # Silent fallback to avoid spam - RAPL access is often restricted on HPC systems
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return 20.0 + (cpu_percent / 100.0) * 30.0
    
    def _get_system_power_consumption(self) -> float:
        """Get total system power consumption in watts."""
        try:
            # Try to read from power supply or system power
            power_paths = [
                "/sys/class/power_supply/BAT0/power_now",  # Battery power
                "/sys/class/power_supply/AC/power_now",    # AC power
            ]
            
            for path in power_paths:
                if os.path.exists(path) and os.access(path, os.R_OK):
                    try:
                        with open(path, 'r') as f:
                            power_uw = int(f.read().strip())
                        return power_uw / 1_000_000  # Convert microwatts to watts
                    except (PermissionError, OSError, ValueError):
                        continue
            
            # Fallback: Estimate based on CPU and memory usage
            cpu_power = self._get_cpu_power_consumption()
            memory = psutil.virtual_memory()
            memory_power = (memory.percent / 100.0) * 5.0  # Estimate memory power
            return cpu_power + memory_power + 10.0  # Add base system power
            
        except Exception as e:
            # Silent fallback - power monitoring is often restricted on HPC systems
            return self._get_cpu_power_consumption() + 15.0
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring:
            try:
                current_time = time.time()
                power = self._get_system_power_consumption()
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                self.power_readings.append(power)
                self.cpu_usage_readings.append(cpu_usage)
                self.memory_readings.append(memory.percent)
                self.timestamps.append(current_time)
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break
    
    def start_monitoring(self):
        """Start power monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Power monitoring started...")
    
    def stop_monitoring(self):
        """Stop power monitoring and return results."""
        if not self.monitoring:
            return None
        
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """Get monitoring results and statistics."""
        if not self.power_readings:
            return {}
        
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate energy consumption (power * time)
        if len(self.power_readings) > 1:
            # Use trapezoidal integration for more accurate energy calculation
            energy_joules = np.trapz(self.power_readings, self.timestamps)
        else:
            energy_joules = self.power_readings[0] * duration if self.power_readings else 0
        
        # Convert to more common units
        energy_kwh = energy_joules / (1000 * 3600)  # Joules to kWh
        energy_wh = energy_joules / 3600  # Joules to Wh
        
        results = {
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'duration_hours': duration / 3600,
            'total_energy_joules': energy_joules,
            'total_energy_wh': energy_wh,
            'total_energy_kwh': energy_kwh,
            'average_power_watts': np.mean(self.power_readings),
            'peak_power_watts': np.max(self.power_readings),
            'min_power_watts': np.min(self.power_readings),
            'average_cpu_usage_percent': np.mean(self.cpu_usage_readings),
            'peak_cpu_usage_percent': np.max(self.cpu_usage_readings),
            'average_memory_usage_percent': np.mean(self.memory_readings),
            'peak_memory_usage_percent': np.max(self.memory_readings),
            'measurement_count': len(self.power_readings),
            'measurement_interval': self.monitoring_interval,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'monitoring_capabilities': self.power_capabilities,
        }
        
        return results
    
    def print_results(self):
        """Print monitoring results in a formatted way."""
        results = self.get_results()
        if not results:
            print("No monitoring data available.")
            return
        
        print("\n" + "="*60)
        print("POWER CONSUMPTION MONITORING RESULTS")
        print("="*60)
        
        # Show monitoring method used
        capabilities = results.get('monitoring_capabilities', {})
        if capabilities.get('rapl_available'):
            print("📊 Monitoring Method: RAPL sensors (most accurate)")
        elif capabilities.get('power_supply_available'):
            print("📊 Monitoring Method: Power supply sensors")
        else:
            print("📊 Monitoring Method: CPU usage estimation")
            if capabilities.get('warnings'):
                print("⚠️  Note: Hardware power sensors not accessible")
        
        print(f"Duration: {results['duration_seconds']:.2f} seconds ({results['duration_minutes']:.2f} minutes)")
        print(f"Total Energy: {results['total_energy_wh']:.4f} Wh ({results['total_energy_kwh']:.6f} kWh)")
        print(f"Average Power: {results['average_power_watts']:.2f} W")
        print(f"Peak Power: {results['peak_power_watts']:.2f} W")
        print(f"Min Power: {results['min_power_watts']:.2f} W")
        print(f"Average CPU Usage: {results['average_cpu_usage_percent']:.1f}%")
        print(f"Peak CPU Usage: {results['peak_cpu_usage_percent']:.1f}%")
        print(f"Average Memory Usage: {results['average_memory_usage_percent']:.1f}%")
        print(f"Peak Memory Usage: {results['peak_memory_usage_percent']:.1f}%")
        print(f"Measurements: {results['measurement_count']} samples")
        print("="*60)
    
    def save_results(self, filename: str):
        """Save monitoring results to a JSON file."""
        results = self.get_results()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")


@contextmanager
def power_monitor(monitoring_interval: float = 0.1):
    """
    Context manager for power monitoring.
    
    Usage:
        with power_monitor() as monitor:
            # Your code here
            pass
        # Results are automatically available
    """
    monitor = PowerMonitor(monitoring_interval)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def estimate_carbon_footprint(energy_kwh: float, carbon_intensity: float = 0.5) -> Dict:
    """
    Estimate carbon footprint based on energy consumption.
    
    Args:
        energy_kwh: Energy consumption in kWh
        carbon_intensity: Carbon intensity in kg CO2 per kWh (default: 0.5 kg CO2/kWh)
    
    Returns:
        Dictionary with carbon footprint estimates
    """
    co2_kg = energy_kwh * carbon_intensity
    co2_g = co2_kg * 1000
    
    # Some context for comparison
    # Average tree absorbs ~22 kg CO2 per year
    # Average car emits ~4.6 tons CO2 per year
    trees_equivalent = co2_kg / 22.0
    car_km_equivalent = co2_kg / 0.12  # 120g CO2 per km
    
    return {
        'co2_kg': co2_kg,
        'co2_g': co2_g,
        'trees_needed_for_offset': trees_equivalent,
        'car_km_equivalent': car_km_equivalent,
        'carbon_intensity_used': carbon_intensity
    }


if __name__ == "__main__":
    # Example usage
    print("Power Monitor Test")
    print("This will monitor power consumption for 10 seconds...")
    
    with power_monitor(monitoring_interval=0.5) as monitor:
        # Simulate some CPU-intensive work
        start = time.time()
        while time.time() - start < 10:
            # Simulate CPU work
            sum(range(100000))
            time.sleep(0.1)
    
    # Print results
    monitor.print_results()
    
    # Save results
    monitor.save_results("power_monitoring_test.json")
    
    # Calculate carbon footprint
    results = monitor.get_results()
    if results:
        carbon_footprint = estimate_carbon_footprint(results['total_energy_kwh'])
        print(f"\nCarbon Footprint Estimate:")
        print(f"CO2 emissions: {carbon_footprint['co2_g']:.2f} g")
        print(f"Trees needed to offset: {carbon_footprint['trees_needed_for_offset']:.4f}")
