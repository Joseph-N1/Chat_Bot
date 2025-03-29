import os
import time
import requests
import pandas as pd
import fastf1
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import json

# Configure caching
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# API Configuration
OPENWEATHER_API_KEY = '07e9b69a901753d144df46077c7198f9'
ERGAST_API_BASE = "http://ergast.com/api/f1"
CIRCUIT_CACHE_FILE = 'circuit_coordinates.json'

# Define race rounds for 2024
RACES_2024 = [
    (2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5),
    (2024, 6), (2024, 7), (2024, 8), (2024, 9), (2024, 10),
    (2024, 11), (2024, 12), (2024, 13), (2024, 14), (2024, 15),
    (2024, 16), (2024, 17), (2024, 18), (2024, 19), (2024, 20),
    (2024, 21), (2024, 22), (2024, 23), (2024, 24)
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
    
    # Determine rain status using OpenWeather API first; if not available, fallback to FastF1 weather data
    rain_status = 'No'
    try:
        if lat and lon:
            weather_data = None
            # First, try OpenWeather API
            try:
                weather_data = fetch_weather(OPENWEATHER_API_KEY, lat, lon, start_time)
            except Exception as e:
                print(f"OpenWeather API failed for driver {ergast_driver_id}: {e}")
            
            # If OpenWeather data is not available and session has weather info, try FastF1 weather data
            if not weather_data and session and hasattr(session, 'weather_data'):
                try:
                    weather_data = session.weather_data
                except Exception as e:
                    print(f"FastF1 weather data fetch error: {e}")
            
            if weather_data:
                # Check for hourly data from OpenWeather API
                if 'hourly' in weather_data:
                    rain_status = 'Yes' if any(hour.get('rain') for hour in weather_data.get('hourly', [])
                                                 if start_time <= hour['dt'] <= end_time) else 'No'
                # Otherwise, if using FastF1 weather data, assume similar structure (modify as needed)
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

def process_race_detailed(year, round_number):
    """
    Process a single race by obtaining:
      - Qualifying Classification (from Ergast)
      - Race Classification (combined Ergast + FastF1, if available)
      - Championship Standings After the Race (from Ergast)
    If FastF1 fails to load, rely solely on Ergast data.
    """
    race_data = {}
    try:
        # Try to load FastF1 session; if it fails, set session to None
        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load()
            time.sleep(2)
        except Exception as fe:
            print(f"FastF1 session load failed for {year} R{round_number}: {fe}")
            session = None

        # Define weather time window
        if session:
            start_time = int(session.date.timestamp())
        else:
            start_time = int(time.time())
        end_time = start_time + 7200

        # Fetch primary Ergast race results
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

        # Fetch Qualifying Classification from Ergast
        qual_results = fetch_qualifying_results(year, round_number)
        qualifying_df = pd.DataFrame(qual_results) if qual_results else pd.DataFrame()

        # Fetch Championship Standings after the race from Ergast
        standings_results = fetch_driver_standings(year, round_number)
        standings_df = pd.DataFrame(standings_results) if standings_results else pd.DataFrame()

        # Race Metadata (e.g., Race Name, Date) from Ergast
        race_metadata = {}
        try:
            race_metadata = {
                'Race': ergast_results[0]['raceName'] if ergast_results and 'raceName' in ergast_results[0] else f"Round {round_number}",
                'Date': pd.to_datetime(ergast_results[0]['date']) if ergast_results and 'date' in ergast_results[0] else None
            }
        except Exception as e:
            print(f"Error extracting race metadata: {e}")

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
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_race_detailed, year, rnd): (year, rnd) for year, rnd in races}
        for future in futures:
            yr, rnd = futures[future]
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
                sheet_name = race_key if len(race_key) <= 31 else race_key[:31]
                startrow = 0
                # Write Race Metadata
                meta_df = pd.DataFrame([race_info.get('metadata', {})])
                meta_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(meta_df) + 1
                
                # Write Qualifying Classification
                race_info['qualifying'].to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(race_info['qualifying']) + 1

                # Write Race Classification
                race_info['race'].to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
                startrow += len(race_info['race']) + 1

                # Write Championship Standings
                race_info['standings'].to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False)
        print(f"Data saved successfully to {template_path}")
    except Exception as e:
        print(f"Excel save error: {e}")

if __name__ == "__main__":
    start_time = time.time()
    all_races = process_all_races(RACES_2024)
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    TEMPLATE_PATH = r'C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot3.0(F1)\F1 Seasons.xlsx'
    save_to_excel_detailed(all_races, TEMPLATE_PATH)
