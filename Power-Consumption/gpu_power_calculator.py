import csv
from datetime import datetime
import os

# Constants
CARBON_INTENSITY = 0.5  # kg CO2 per kWh - adjust based on your location's energy mix
SECONDS_TO_HOURS = 1 / 3600
WATTS_TO_KILOWATTS = 1 / 1000

def parse_log_file(file_path):
    """Parse the log file and return a list of entries."""
    entries = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 6:  # Ensure we have all columns
                entries.append({
                    'timestamp': row[0].strip(),
                    'index': int(row[1].strip()),
                    'power': float(row[5].strip().replace(' W', ''))
                })
    return entries

def calculate_power_consumption(entries):
    """Calculate total power consumption in kWh."""
    if not entries:
        return 0.0
    
    # Sort entries by timestamp
    entries.sort(key=lambda x: x['timestamp'])
    
    total_power = 0.0  # in watt-seconds
    prev_time = None
    prev_power = {}
    
    for entry in entries:
        current_time = datetime.strptime(entry['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
        gpu_index = entry['index']
        current_power = entry['power']
        
        if prev_time is not None and gpu_index in prev_power:
            # Calculate time difference in seconds
            time_diff = (current_time - prev_time).total_seconds()
            
            # Add power consumption for this interval (average power * time)
            avg_power = (prev_power[gpu_index] + current_power) / 2
            total_power += avg_power * time_diff
        
        prev_power[gpu_index] = current_power
        prev_time = current_time
    
    # Convert to kWh
    return total_power * WATTS_TO_KILOWATTS * SECONDS_TO_HOURS

def calculate_carbon_footprint(power_kWh, carbon_intensity=CARBON_INTENSITY):
    """Calculate carbon footprint in kg CO2."""
    return power_kWh * carbon_intensity

def analyze_log_file(file_path):
    """Analyze a single log file and return power and carbon footprint."""
    entries = parse_log_file(file_path)
    power_kWh = calculate_power_consumption(entries)
    carbon_kg = calculate_carbon_footprint(power_kWh)
    return power_kWh, carbon_kg

def analyze_log_directory(directory_path):
    """Analyze all log files in a directory."""
    total_power = 0.0
    total_carbon = 0.0
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.log') or filename.endswith('.csv'):  # adjust extensions as needed
            file_path = os.path.join(directory_path, filename)
            power_kWh, carbon_kg = analyze_log_file(file_path)
            total_power += power_kWh
            total_carbon += carbon_kg
            print(f"File: {filename}")
            print(f"  Power consumption: {power_kWh:.3f} kWh")
            print(f"  Carbon footprint: {carbon_kg:.3f} kg CO2")
            print()
    
    print("=== Summary ===")
    print(f"Total power consumption: {total_power:.3f} kWh")
    print(f"Total carbon footprint: {total_carbon:.3f} kg CO2")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python gpu_power_calculator.py <log_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        power_kWh, carbon_kg = analyze_log_file(path)
        print(f"Power consumption: {power_kWh:.3f} kWh")
        print(f"Carbon footprint: {carbon_kg:.3f} kg CO2")
    elif os.path.isdir(path):
        analyze_log_directory(path)
    else:
        print(f"Error: Path '{path}' not found")
        sys.exit(1)