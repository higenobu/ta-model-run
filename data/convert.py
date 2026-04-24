import os
import glob
import json

# Directory containing the JSON files
json_directory = '/home/alkalinemoe/psych_model_scripts/data/old_cp'

# Function to convert a single JSON file to JSON Lines format
def convert_json_to_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert each dictionary in the list to a separate line in JSON Lines format
    jsonl_content = '\n'.join(json.dumps(record, ensure_ascii=False) for record in data)

    # Write the JSON Lines content back to the file
    with open(file_path, 'w') as file:
        # ensure_ascii=False to write the file in UTF-8 format
        file.write(jsonl_content)

# List all JSON files in the directory
json_files = glob.glob(os.path.join(json_directory, '*.json'))

# Convert each file to JSON Lines format
for file in json_files:
    convert_json_to_jsonl(file)

# Return the list of processed files
json_files