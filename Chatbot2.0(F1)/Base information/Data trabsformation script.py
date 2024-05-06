import json
import os

# Directory where your JSON files are stored
directory = 'C:\\Users\\Joseph N Nimyel\\OneDrive\\Documents\\Chatbot\\Chatbot2.0(F1)\\Database'

# New structure setup
new_data = {
    "drivers": {},
    "races": {},
    "tracks": {}
}

# Function to process each race data entry
def process_race_data(race, new_data):
    race_name = race['raceName']
    race_date = race['date']
    track_name = race['Circuit']['circuitName']
    for result in race['Results']:
        driver_id = result['Driver']['driverId']
        if driver_id not in new_data['drivers']:
            new_data['drivers'][driver_id] = {
                "team": result['Constructor']['name'],
                "wins": []
            }
        new_data['drivers'][driver_id]['wins'].append({
            "race": race_name,
            "year": int(race['season']),
            "track": track_name,
            "date": race_date,
            "lap_time": result['FastestLap']['Time']['time'],
            "average_speed": result['FastestLap']['AverageSpeed']['speed'] + " kph"
        })
        if race['season'] not in new_data['races']:
            new_data['races'][race['season']] = []
        new_data['races'][race['season']].append({
            "name": race_name,
            "date": race_date,
            "winners": [driver_id],
            "track": track_name
        })
        if track_name not in new_data['tracks']:
            new_data['tracks'][track_name] = {
                "location": race['Circuit']['Location']['country'],
                "best_time_lap": result['FastestLap']['Time']['time'],
                "record_holder": driver_id
            }

def load_and_process_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        for year in data.keys():
            print(f"Processing data for the year: {year}")
            mr_data = data[year]['MRData']
            if 'RaceTable' in mr_data and 'Races' in mr_data['RaceTable']:
                races = mr_data['RaceTable']['Races']
                for race in races:
                    process_race_data(race, new_data)
            else:
                print(f"No race data available for the year {year}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"An error occurred with file: {os.path.basename(file_path)}. Error: {str(e)}")

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):  # Processing only JSON files
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {filename}")
        load_and_process_data(file_path)
    else:
        print(f"Skipped non-JSON file: {filename}")

# Save the transformed data to a new JSON file
with open('C:\\Users\\Joseph N Nimyel\\OneDrive\\Documents\\Chatbot\\Chatbot2.0(F1)\\f1_knowledge_base.json', 'w') as file:
    json.dump(new_data, file, indent=4)

print("Data transformation complete.")
