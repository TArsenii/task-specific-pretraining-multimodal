import os
import json


def remove_keys_from_json(root_dir, key_pattern="average_precision"):
    # Walk through all files in the directory and subdirectories
    for root, _, files in os.walk(root_dir):
        for file in files:
            # Process only .json files
            print(file)
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Load JSON data
                with open(file_path, "r") as json_file:
                    try:
                        data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON file: {file_path}")
                        continue

                if isinstance(data, list):
                    data = [{k: v for k, v in d.items() if key_pattern not in k} for d in data]
                else:
                    # Remove keys containing the key_pattern
                    data = {k: v for k, v in data.items() if key_pattern not in k}

                # Write the updated JSON back to the file
                with open(file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"Updated file: {file_path}")


root_directory = "./experiments_output/AVMNIST_Multimodal_Training"
remove_keys_from_json(root_directory)
