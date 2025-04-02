import os
import time
import requests
import pandas as pd
import fastf1
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from datetime import datetime, time as dt_time


# Configure caching for 2023
CACHE_DIR = os.path.join('cache', '2023')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# API Configuration (same as 2024)
OPENWEATHER_API_KEY = '07e9b69a901753d144df46077c7198f9'
ERGAST_API_BASE = "http://ergast.com/api/f1"
CIRCUIT_CACHE_FILE = 'circuit_coordinates_2023.json'

# Define all 2023 races (year, round number, race name)
RACES_2023 = [
    (2023, 1,  "Bahrain Grand Prix"),
    (2023, 2,  "Saudi Arabian Grand Prix"),
    (2023, 3,  "Australian Grand Prix"),
    (2023, 4,  "Azerbaijan Grand Prix"),
    (2023, 5,  "Miami Grand Prix"),
    (2023, 6,  "Monaco Grand Prix"),
    (2023, 7,  "Spanish Grand Prix"),
    (2023, 8,  "Canadian Grand Prix"),
    (2023, 9,  "Austrian Grand Prix"),
    (2023, 10, "British Grand Prix"),
    (2023, 11, "Hungarian Grand Prix"),
    (2023, 12, "Belgian Grand Prix"),
    (2023, 13, "Dutch Grand Prix"),
    (2023, 14, "Italian Grand Prix"),
    (2023, 15, "Singapore Grand Prix"),
    (2023, 16, "Japanese Grand Prix"),
    (2023, 17, "Qatar Grand Prix"),
    (2023, 18, "United States Grand Prix"),
    (2023, 19, "Mexico City Grand Prix"),
    (2023, 20, "São Paulo Grand Prix"),
    (2023, 21, "Las Vegas Grand Prix"),
    (2023, 22, "Abu Dhabi Grand Prix")
]

# Load or create circuit coordinate cache
if os.path.exists(CIRCUIT_CACHE_FILE):
    with open(CIRCUIT_CACHE_FILE, 'r') as f:
        circuit_cache = json.load(f)
else:
    circuit_cache = {}

def save_circuit_cache():
    with open(CIRCUIT_CACHE_FILE, 'w') as f:
        json.dump(circuit_cache, f)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_weather(api_key, lat, lon, timestamp):
    """Fetch weather data from OpenWeather API"""
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={api_key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ergast_race_results(year, round_number):
    """Fetch Ergast race results (Race Classification)"""
    url = f"{ERGAST_API_BASE}/{year}/{round_number}/results.json?limit=100"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()['MRData']['RaceTable']['Races'][0]['Results']

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_qualifying_results(year, round_number):
    """Fetch Ergast qualifying results"""
    url = f"{ERGAST_API_BASE}/{year}/{round_number}/qualifying.json?limit=100"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    races = response.json()['MRData']['RaceTable']['Races']
    if races:
        return races[0]['QualifyingResults']
    return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_driver_standings(year, round_number):
    """Fetch Ergast driver standings after the race"""
    url = f"{ERGAST_API_BASE}/{year}/{round_number}/driverStandings.json?limit=100"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    standings = response.json()['MRData']['StandingsTable']['StandingsLists']
    if standings:
        return standings[0]['DriverStandings']
    return []

def get_circuit_coordinates(session):
    """Retrieve circuit coordinates using FastF1 (with caching)"""
    cache_key = f"{session.event['EventName']}_{session.date.year}"
    if cache_key in circuit_cache:
        return circuit_cache[cache_key]
    
    circuit_info = session.get_circuit_info()
    coords = (circuit_info.location["lat"], circuit_info.location["long"])
    circuit_cache[cache_key] = coords
    save_circuit_cache()
    return coords

def process_driver_combined(session, ergast_result, start_time, end_time):
    """
    Merge Ergast primary data with supplementary FastF1 data for a driver.
    If FastF1 fails, use Ergast data only.
    """
    ergast_driver_id = ergast_result['Driver']['driverId']
    laps = None
    fastest_lap = None
    driver_info = None

    try:
        if session and ergast_driver_id in session.drivers:
            laps = session.laps.pick_driver(ergast_driver_id)
            fastest_lap = laps.pick_fastest() if not laps.empty else None
            driver_info = session.get_driver(ergast_driver_id)
        else:
            print(f"Driver {ergast_driver_id} not found in FastF1; skipping supplementary info.")
    except Exception as e:
        print(f"FastF1 error for driver {ergast_driver_id}: {e}")
        laps = None
        fastest_lap = None
        driver_info = None

    try:
        lat, lon = get_circuit_coordinates(session) if session else (None, None)
    except Exception as e:
        print(f"Circuit coordinate error: {e}")
        lat, lon = (None, None)
    
    rain_status = 'No'
    try:
        if lat and lon:
            weather_data = None
            try:
                weather_data = fetch_weather(OPENWEATHER_API_KEY, lat, lon, start_time)
            except Exception as e:
                print(f"OpenWeather API failed for driver {ergast_driver_id}: {e}")
            
            if not weather_data and session and hasattr(session, 'weather_data'):
                try:
                    weather_data = session.weather_data
                except Exception as e:
                    print(f"FastF1 weather data fetch error: {e}")
            
            if weather_data:
                if 'hourly' in weather_data:
                    rain_status = 'Yes' if any(hour.get('rain') for hour in weather_data.get('hourly', [])
                                                 if start_time <= hour['dt'] <= end_time) else 'No'
                elif isinstance(weather_data, list):
                    rain_status = 'Yes' if any(record.get('rain', False) for record in weather_data) else 'No'
    except Exception as e:
        print(f"Weather fetch error for driver {ergast_driver_id}: {e}")

    race_data = {
        'Driver': (driver_info['FullName'] if driver_info else ergast_result['Driver']['familyName']),
        'Team': ergast_result['Constructor']['name'],
        'Starting Position': int(ergast_result['grid']),
        'Finish Position': int(ergast_result['position']),
        'Ergast Laps': int(ergast_result['laps']),
        'Points': float(ergast_result['points']),
        'Fastest Lap Time': fastest_lap['LapTime'] if fastest_lap is not None and not fastest_lap.empty else pd.NaT,
        'FastF1 Laps Completed': len(laps) if laps is not None else 0,
        'Overtakes': (len(laps) - int(ergast_result['grid'])) if laps is not None else 0,
        'DNF': 'Yes' if ergast_result['status'] != 'Finished' else 'No',
        'Tire Compounds': ', '.join(laps['Compound'].unique()) if laps is not None and not laps.empty else '',
        'Rain During Race': rain_status
    }
    return race_data

def process_race_detailed(year, round_number, provided_race_name):
    """
    Process a single race by obtaining:
      - Qualifying Classification (from Ergast)
      - Race Classification (combined Ergast + FastF1, if available)
      - Championship Standings After the Race (from Ergast)
    If FastF1 fails to load, rely solely on Ergast data.
    """
    race_data = {}
    try:
        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load()
            time.sleep(2)
        except Exception as fe:
            print(f"FastF1 session load failed for {year} R{round_number}: {fe}")
            session = None

        if session:
            start_time = int(session.date.timestamp())
        else:
            start_time = int(time.time())
        end_time = start_time + 7200

        ergast_results = fetch_ergast_race_results(year, round_number)
        if session:
            race_classification = [process_driver_combined(session, res, start_time, end_time) for res in ergast_results]
        else:
            race_classification = []
            for res in ergast_results:
                record = {
                    'Driver': res['Driver']['familyName'],
                    'Team': res['Constructor']['name'],
                    'Starting Position': int(res['grid']),
                    'Finish Position': int(res['position']),
                    'Ergast Laps': int(res['laps']),
                    'Points': float(res['points']),
                    'Fastest Lap Time': pd.NaT,
                    'FastF1 Laps Completed': 0,
                    'Overtakes': 0,
                    'DNF': 'Yes' if res['status'] != 'Finished' else 'No',
                    'Tire Compounds': '',
                    'Rain During Race': 'Not Available'
                }
                race_classification.append(record)
        race_classification_df = pd.DataFrame(race_classification)

        qual_results = fetch_qualifying_results(year, round_number)
        qualifying_df = pd.DataFrame(qual_results) if qual_results else pd.DataFrame()

        standings_results = fetch_driver_standings(year, round_number)
        standings_df = pd.DataFrame(standings_results) if standings_results else pd.DataFrame()

        race_metadata = {}
        try:
            # Use Ergast's race name if available, otherwise fallback to the provided name
            ergast_race_name = ergast_results[0].get('raceName') if ergast_results and 'raceName' in ergast_results[0] else provided_race_name
            race_metadata = {
                'Race': ergast_race_name,
                'Date': pd.to_datetime(ergast_results[0]['date']) if ergast_results and 'date' in ergast_results[0] else None
            }
        except Exception as e:
            print(f"Error extracting race metadata: {e}")
            race_metadata = {'Race': provided_race_name, 'Date': None}

        race_data['metadata'] = race_metadata
        race_data['qualifying'] = qualifying_df
        race_data['race'] = race_classification_df
        race_data['standings'] = standings_df

    except Exception as e:
        print(f"Error processing detailed race data for {year} R{round_number}: {e}")
        race_data = {}
    return race_data

def process_all_races(races):
    """Process detailed data for all races and return a dictionary keyed by race identifier."""
    all_races_data = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Each race tuple now has (year, round, race_name)
        futures = {executor.submit(process_race_detailed, year, rnd, race_name): (year, rnd, race_name) for year, rnd, race_name in races}
        for future in futures:
            yr, rnd, race_name = futures[future]
            try:
                race_data = future.result()
                if race_data:
                    key = f"{yr}_R{rnd}"
                    all_races_data[key] = race_data
            except Exception as e:
                print(f"Error processing race {yr} R{rnd}: {e}")
    return all_races_data

def save_to_excel_detailed(races_data, template_path):
    """
    Save the detailed race data to an Excel file.
    Each race gets its own sheet with the following layout:
      Race Metadata (top)
      Qualifying Classification
      (1 blank row)
      Race Classification
      (1 blank row)
      Championship Standings After the Race
    """
    try:
        with pd.ExcelWriter(template_path, engine='openpyxl', mode='w') as writer:
            for race_key, race_info in races_data.items():
                # Prepend "2023_" to differentiate if needed.
                sheet_name = f"2023_{race_key}"
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                startrow = 0
                meta_df = pd.DataFrame([race_info.get('metadata', {})])
                meta_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(meta_df) + 1
                race_info['qualifying'].to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(race_info['qualifying']) + 1
                race_info['race'].to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(race_info['race']) + 1
                race_info['standings'].to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
        print(f"2023 data saved successfully to {template_path}")
    except Exception as e:
        print(f"Excel save error: {e}")


if __name__ == "__main__":
    start_time = time.time()
    all_races = process_all_races(RACES_2023)
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Ensure the directory exists before saving the Excel file
    TEMPLATE_DIR = r'C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot3.0(F1)'
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR)
    TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, 'F1_Seasons_2023.xlsx')
    
    save_to_excel_detailed(all_races, TEMPLATE_PATH)

