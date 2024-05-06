import json
import os

def load_data(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            print(f"Data loaded from {file_name}: {data}")  # Debugging output
            return data
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading {file_name}: {e}")
        return {}
def update_knowledgebase(main_file, data_block):
    try:
        with open(main_file, 'r+') as file:
            knowledgebase = json.load(file)
            print(f"Original knowledge base data: {knowledgebase}")  # Debugging output
            
            # Update the "F1_data" section specifically
            if "F1_data" in knowledgebase:
                knowledgebase["F1_data"].update(data_block)
            else:
                knowledgebase["F1_data"] = data_block

            file.seek(0)
            file.truncate()
            json.dump(knowledgebase, file, indent=4)
            print(f"Updated knowledge base data: {knowledgebase}")  # Debugging output
    except FileNotFoundError:
        print(f"File not found: {main_file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {main_file}")
    except Exception as e:
        print(f"Unexpected error updating {main_file}: {e}")
base_path = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot2.0(F1)\F1 Database"
knowledge_base_path = os.path.join(base_path, "knowledge_base.json")

# Load data from each source file
data_v1 = load_data(os.path.join(base_path, "f1_knowledge_base_V1.json"))
data_v2 = load_data(os.path.join(base_path, "f1_knowledge_base_V2.json"))

# Combine data from V1 and V2 into a single dictionary before updating (if necessary)
combined_data = {**data_v1, **data_v2}  # This merges V2 into V1, with V2's data taking precedence if keys overlap

# Update the main knowledge base file
update_knowledgebase(knowledge_base_path, combined_data)
