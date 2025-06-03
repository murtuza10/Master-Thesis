import json

# Load the original JSON data
with open("/home/s27mhusa_hpc/Master-Thesis/countries_de.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("/home/s27mhusa_hpc/Master-Thesis/countries_en.json", "r", encoding="utf-8") as f:
    data_en = json.load(f)

# Extract only the 'name' fields
country_names = [entry["name"] for entry in data]
country_names.extend([entry["name"] for entry in data_en])

# Save to a new JSON file
with open("/home/s27mhusa_hpc/Master-Thesis/countries_list.json", "w+", encoding="utf-8") as f:
    json.dump(country_names, f, ensure_ascii=False, indent=2)

print("Country names saved to countries_list.json")