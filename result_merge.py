import json
from pathlib import Path

# Initialize empty dictionary to hold merged results
merged_data = {}

# Loop through output-tip3p-0, 1, 2
for i in range(3):
    file_path = Path(f"output-tip3p-{i}/rep-1/openff-2.2.1/results.json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        merged_data.update(data)  # Overwrites duplicate keys, if any

# Save merged result
output_file = "merged_results.json"
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"Merged JSON saved to {output_file}")
