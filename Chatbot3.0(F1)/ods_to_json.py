import pandas as pd
import json
import os
import re
from collections import OrderedDict
from datetime import datetime, time

def clean_and_convert_to_json(input_file, output_path):
    """
    Convert ODS file with F1 data to clean JSON format
    
    Args:
        input_file (str): Path to the ODS file
        output_path (str): Path to save the JSON output
    """
    # Dict to hold all race data
    all_races = []
    
    print(f"Processing {input_file}...")
    
    # Read the ODS file - each sheet is in a separate DataFrame
    xls = pd.ExcelFile(input_file, engine='odf')
    
    # Process each sheet
    for sheet_name in xls.sheet_names:
        # Read the sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Skip completely empty sheets
        if df.empty:
            continue
        
        # Clean up the data and extract information
        race_data = process_sheet(df, sheet_name)
        
        if race_data:
            all_races.append(race_data)
    
    # Write the JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_races, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion complete. JSON saved to {output_path}")

def process_sheet(df, sheet_name):
    """
    Process a single sheet and extract the three tables
    
    Args:
        df (pandas.DataFrame): DataFrame containing the sheet data
        sheet_name (str): Name of the sheet
        
    Returns:
        dict: Structured race data
    """
    # Remove NaN values for better processing
    df = df.fillna('')
    
    # Convert the DataFrame to a list of lists for easier manipulation
    data_rows = df.values.tolist()
    
    # Find the race name (usually in the second row)
    race_name = None
    for row in data_rows:
        if isinstance(row[0], str) and 'Grand Prix' in row[0]:
            race_name = row[0].strip()
            break
    
    if not race_name:
        # Try to extract race name from sheet name
        race_name = sheet_name
    
    # Find the qualifying results section
    quali_start = -1
    quali_end = -1
    race_start = -1
    race_end = -1
    standings_start = -1
    standings_end = -1
    
    for i, row in enumerate(data_rows):
        row_str = ' '.join([str(cell) for cell in row]).lower()
        
        # Find qualifying table
        if 'position' in row_str and 'driver' in row_str and 'q1' in row_str:
            quali_start = i
        elif quali_start != -1 and quali_end == -1 and all(cell == '' for cell in row):
            quali_end = i
        
        # Find race results table
        if 'driver' in row_str and 'team' in row_str and 'starting position' in row_str:
            race_start = i
        elif race_start != -1 and race_end == -1 and all(cell == '' for cell in row):
            race_end = i
        
        # Find standings table
        if 'pos' in row_str and 'pts' in row_str and 'wins' in row_str and 'driver' in row_str:
            standings_start = i
        elif standings_start != -1 and standings_end == -1 and i == len(data_rows) - 1:
            standings_end = i + 1
        elif standings_start != -1 and standings_end == -1 and all(cell == '' for cell in row):
            standings_end = i
    
    # Process qualifying data
    qualifying_results = []
    if quali_start != -1 and quali_end != -1:
        # Add a section header
        qualifying_header = {"section": "qualifying_results"}
        qualifying_results.append(qualifying_header)
        
        quali_headers = [str(h).lower().strip() for h in data_rows[quali_start]]
        
        # Define the columns we want to keep for qualifying
        quali_keep_columns = ['position', 'driver', 'constructor', 'q1', 'q2', 'q3']
        
        # Find indices of the columns we want to keep
        quali_keep_indices = []
        quali_keep_headers = []
        
        for col in quali_keep_columns:
            try:
                idx = quali_headers.index(col)
                quali_keep_indices.append(idx)
                quali_keep_headers.append(col)
            except ValueError:
                # Column might not exist
                pass
        
        for i in range(quali_start + 1, quali_end):
            row = data_rows[i]
            if not any(row):  # Skip empty rows
                continue
                
            driver_data = {}
            for idx, header in zip(quali_keep_indices, quali_keep_headers):
                if idx < len(row):
                    driver_data[header] = row[idx]
            
            if driver_data:
                qualifying_results.append(driver_data)
    
    # Process race results
    race_results = []
    if race_start != -1 and race_end != -1:
        # Add a section header
        race_header = {"section": "race_results"}
        race_results.append(race_header)
        
        race_headers = [str(h).lower().strip() for h in data_rows[race_start]]
        
        # Define the columns we want to keep for race results
        race_keep_columns = [
            'driver', 'team', 'starting position', 'finish position', 
            'ergast laps', 'points', 'fastest lap time', 'dnf', 
            'tire compounds', 'rain during race'
        ]
        
        # Find indices of the columns we want to keep
        race_keep_indices = []
        race_keep_headers = []
        
        for col in race_keep_columns:
            try:
                idx = race_headers.index(col)
                race_keep_indices.append(idx)
                race_keep_headers.append(col)
            except ValueError:
                # Column might not exist
                pass
        
        for i in range(race_start + 1, race_end):
            row = data_rows[i]
            if not any(row):  # Skip empty rows
                continue
            
            driver_data = {}
            for idx, header in zip(race_keep_indices, race_keep_headers):
                if idx < len(row):
                    cell_value = row[idx]
                    # If the cell value is a time object, convert it to a string
                    if isinstance(cell_value, time):
                        cell_value = cell_value.strftime("%H:%M:%S")
                    # Optionally, you could do a similar check for other types like timedelta if needed.
                    driver_data[header] = cell_value
                    
            if driver_data:
                race_results.append(driver_data)
    
    # Process standings
    standings = []
    if standings_start != -1 and standings_end != -1:
        # Add a section header
        standings_header = {"section": "championship_standings"}
        standings.append(standings_header)
        
        standings_headers = [str(h).lower().strip() for h in data_rows[standings_start]]
        
        # Calculate driver age from DOB if available
        current_year = datetime.now().year
        
        for i in range(standings_start + 1, standings_end):
            row = data_rows[i]
            if not any(row):  # Skip empty rows
                continue
                
            driver_data = {}
            
            # Process each cell in the row
            for j, header in enumerate(standings_headers):
                cell_value = row[j] if j < len(row) else ""
                
                # Handle "Pos" column
                if header.startswith('pos'):
                    driver_data['pos'] = cell_value
                
                # Handle "Pts" column
                elif header.startswith('pts'):
                    driver_data['pts'] = cell_value
                
                # Handle "Wins" column
                elif header.startswith('wins'):
                    driver_data['wins'] = cell_value
                
                # Handle "Driver" column
                elif header.startswith('driver') and '/' in str(cell_value):
                    parts = str(cell_value).split('/')
                    if len(parts) >= 3:
                        driver_data['driver_name'] = parts[0].strip()
                        driver_data['driver_nationality'] = parts[1].strip()
                        
                        # Calculate age if DOB is available
                        dob = parts[2].strip()
                        if dob and '-' in dob:
                            try:
                                birth_year = int(dob.split('-')[0])
                                driver_data['driver_age'] = current_year - birth_year
                            except:
                                driver_data['driver_age'] = ""
                
                # Handle "Team" column
                elif header.startswith('team') and '/' in str(cell_value):
                    parts = str(cell_value).split('/')
                    if len(parts) >= 2:
                        driver_data['team_name'] = parts[0].strip()
                        driver_data['team_nationality'] = parts[1].strip()
            
            if driver_data:
                standings.append(driver_data)
    
    # Create the final race data structure
    race_data = {
        'race_name': race_name,
        'data': [
            *qualifying_results,
            *race_results,
            *standings
        ]
    }
    
    return race_data

# Your specific file paths
input_file = r'C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot3.0(F1)\F1_Seasons_Cleaned_(2024).ods'
output_file = r'C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\DataBase\F1_Seasons_Cleaned_2024.json'

# Run the conversion
if __name__ == "__main__":
    clean_and_convert_to_json(input_file, output_file)
