import requests
import json
import os

def fetch_f1_data(year):
    base_url = f"http://ergast.com/api/f1/{year}/results/1.json"
    response = requests.get(base_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get data for {year}")
        return None

def save_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

def process_file(json_file):
    # Read the JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Process the data here if needed
    print(f"Processed {json_file}")

def main():
    directory = 'database'  # Directory to store JSON files
    start_year = 2020
    end_year = 2024  # Adjust the range as needed
    
    for year in range(start_year, end_year + 1):
        year_data = fetch_f1_data(year)
        if year_data:
            filename = f"{directory}/first_place_finishers_{year}.json"
            save_data(year_data, filename)
            process_file(filename)
        else:
            print(f"No data available for {year}")

if __name__ == "__main__":
    main()
