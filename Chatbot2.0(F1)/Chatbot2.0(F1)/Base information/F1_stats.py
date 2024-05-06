import json
import os

def load_json(filename):
    """ Load data from a JSON file. """
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    """ Save data to a JSON file. """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def reformat_data(original_data):
    """ Reformat the original JSON structure to the desired format. """
    new_format = {"F1_data": {}}
    for driver, driver_info in original_data["drivers"].items():
        for win in driver_info["wins"]:
            year = str(win["year"])
            if year not in new_format["F1_data"]:
                new_format["F1_data"][year] = {"drivers": {}}
            if driver not in new_format["F1_data"][year]["drivers"]:
                new_format["F1_data"][year]["drivers"][driver] = {"wins": []}
            new_format["F1_data"][year]["drivers"][driver]["wins"].append({
                "race": win["race"],
                "track": win["track"],
                "date": win["date"],
                "lap_time": win["lap_time"],
                "average_speed": win["average_speed"]
            })
    return new_format

def process_files(input_files, output_file):
    """ Process multiple JSON files and combine their data into a single output file. """
    combined_data = {"F1_data": {}}
    for file_name in input_files:
        data = load_json(file_name)
        reformatted = reformat_data(data)
        for year, content in reformatted["F1_data"].items():
            if year not in combined_data["F1_data"]:
                combined_data["F1_data"][year] = content
            else:
                for driver, wins in content["drivers"].items():
                    if driver not in combined_data["F1_data"][year]["drivers"]:
                        combined_data["F1_data"][year]["drivers"][driver] = wins
                    else:
                        combined_data["F1_data"][year]["drivers"][driver]["wins"].extend(wins["wins"])
    save_json(combined_data, output_file)

# Define file paths
base_path = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot2.0(F1)\F1 Database"
output_path = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot2.0(F1)\CHAT\F1_stats.json"
files = [os.path.join(base_path, f"f1_knowledge_base_V{version}.json") for version in range(1, 3)]

# Process the files and save the new structure
process_files(files, output_path)

print("Data has been reformatted and saved successfully.")
