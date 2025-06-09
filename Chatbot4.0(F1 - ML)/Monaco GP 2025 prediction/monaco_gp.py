import os
import json
import glob
import argparse
from collections import defaultdict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRanker
import numpy as np

# ------------------------
# Section: Configuration
# ------------------------
# Local database path
DB_DIR = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\DataBase"
YEARS = [2024]
PRE_MONACO_COUNT = 7
MONACO_HISTORY_YEARS = [2021, 2022, 2023]
OUTPUT_FILE = "monaco_prediction_2024.json"

# Driver codes mapping (will be populated from championship data)
DRIVER_CODES = {}

def generate_driver_codes(championship_data):
    """Extract driver codes from championship standings"""
    codes = {}
    for entry in championship_data:
        if isinstance(entry, dict) and 'driver_name' in entry:
            full_name = entry['driver_name']
            surname = full_name.split()[-1]  # Get last name
            # Special cases for drivers with same surname initials
            if surname == 'Alonso': code = 'ALO'
            elif surname == 'Bottas': code = 'BOT'
            elif surname == 'Gasly': code = 'GAS'
            elif surname == 'Hamilton': code = 'HAM'
            elif surname == 'Hülkenberg': code = 'HUL'
            elif surname == 'Leclerc': code = 'LEC'
            elif surname == 'Magnussen': code = 'MAG'
            elif surname == 'Norris': code = 'NOR'
            elif surname == 'Ocon': code = 'OCO'
            elif surname == 'Pérez': code = 'PER'
            elif surname == 'Piastri': code = 'PIA'
            elif surname == 'Ricciardo': code = 'RIC'
            elif surname == 'Russell': code = 'RUS'
            elif surname == 'Sainz': code = 'SAI'
            elif surname == 'Stroll': code = 'STR'
            elif surname == 'Tsunoda': code = 'TSU'
            elif surname == 'Verstappen': code = 'VER'
            elif surname == 'Zhou': code = 'ZHO'
            elif surname == 'Albon': code = 'ALB'
            elif surname == 'Bearman': code = 'BEA'
            elif surname == 'Colapinto': code = 'COL'
            elif surname == 'Doohan': code = 'DOO'
            elif surname == 'Lawson': code = 'LAW'
            elif surname == 'Sargeant': code = 'SAR'
            elif surname == 'Guanyu': code = 'ZHO'  # Zhou Guanyu special case
            else:
                code = surname[:3].upper()  # Default first 3 letters of surname
            codes[full_name] = code
            codes[surname] = code  # Also map surname for race results
    return codes

def get_driver_code(name):
    """Get standardized 3-letter driver code"""
    return DRIVER_CODES.get(name, name[:3].upper())

# ------------------------
# Section: Data Loading
# ------------------------
def load_season(year):
    """Load season data from JSON file"""
    path = os.path.join(DB_DIR, f"F1_Seasons_Cleaned_{year}.json")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Initialize driver codes from championship data
            global DRIVER_CODES
            for race in data:
                if isinstance(race, dict) and 'data' in race:
                    for item in race['data']:
                        if isinstance(item, dict) and item.get('section') == 'championship_standings':
                            # Find championship data after the section marker
                            race_data = race['data']
                            section_idx = race_data.index(item)
                            champ_data = race_data[section_idx+1:]
                            DRIVER_CODES = generate_driver_codes(champ_data)
                            break
                    if DRIVER_CODES:
                        break
            
            return data
    except FileNotFoundError:
        print(f"Error: Could not find file {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {path}")
        return []

# ------------------------
# Section: Feature Engineering
# ------------------------
def extract_race_features(races, up_to_race):
    """Process race data from the loaded JSON structure"""
    records = []
    
    for race in races[:up_to_race]:
        if not isinstance(race, dict) or 'data' not in race:
            continue
            
        # Find race_results section
        race_data = race['data']
        results_section_idx = None
        
        for i, item in enumerate(race_data):
            if isinstance(item, dict) and item.get('section') == 'race_results':
                results_section_idx = i
                break
        
        if results_section_idx is None:
            continue
            
        # Process all entries after race_results marker
        for item in race_data[results_section_idx+1:]:
            if not isinstance(item, dict) or 'driver' not in item:
                continue
                
            # Skip if we hit another section
            if 'section' in item:
                break
                
            try:
                records.append({
                    'driver': item['driver'],
                    'team': item['team'],
                    'start_pos': int(item.get('starting position', 20)),
                    'finish_pos': int(item.get('finish position', 20)),
                    'points': int(item.get('points', 0)),
                    'fastest_lap': 1 if item.get('fastest lap time') and item.get('fastest lap time').strip() else 0,
                    'dnf': 1 if str(item.get('dnf', 'No')).lower() == 'yes' else 0
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping malformed driver data - {e}")
                continue
                
    return pd.DataFrame(records)

# Hot streak detection
def detect_hot_streak(df, driver_name):
    """Detect if driver is on a hot streak based on recent performance"""
    driver_data = df[df['driver'] == driver_name]
    if len(driver_data) < 3:
        return 0
        
    # Sort by index to get chronological order
    driver_data = driver_data.sort_index()
    last3_points = driver_data.tail(3)['points'].sum()
    avg_points = driver_data['points'].mean()
    
    return 1 if (avg_points > 0 and last3_points > 1.5 * len(driver_data) * avg_points / len(driver_data)) else 0

# Monaco history features
def load_monaco_history():
    """Load Monaco GP historical results"""
    podiums = []
    
    for year in MONACO_HISTORY_YEARS:
        try:
            season_data = load_season(year)
            
            # Find Monaco GP
            for race in season_data:
                if isinstance(race, dict) and 'race_name' in race:
                    if 'Monaco' in race['race_name']:
                        # Find race results section
                        race_data = race['data']
                        results_section_idx = None
                        
                        for i, item in enumerate(race_data):
                            if isinstance(item, dict) and item.get('section') == 'race_results':
                                results_section_idx = i
                                break
                        
                        if results_section_idx is not None:
                            # Get top 3 finishers
                            race_results = []
                            for item in race_data[results_section_idx+1:]:
                                if isinstance(item, dict) and 'driver' in item and 'section' not in item:
                                    race_results.append(item)
                                elif 'section' in item:
                                    break
                            
                            # Sort by finish position and get top 3
                            race_results.sort(key=lambda x: int(x.get('finish position', 999)))
                            for pos, result in enumerate(race_results[:3], 1):
                                podiums.append({
                                    'driver': result['driver'],
                                    'year': year,
                                    'pos': pos,
                                    'team': result['team']
                                })
                        break
        except Exception as e:
            print(f"Warning: Could not load Monaco history for {year}: {e}")
            continue
    
    return pd.DataFrame(podiums)

# ------------------------
# Section: Model Training
# ------------------------
def train_models(X, y_class, group):
    """Train both classification and ranking models"""
    # RandomForest for podium classification (top3 vs others)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_class)
    
    # XGBoost ranker for full grid ordering
    xgb = XGBRanker(objective='rank:pairwise', random_state=42)
    xgb.fit(X, y_class, group=group)
    
    return rf, xgb

# ------------------------
# Section: Prediction & Output
# ------------------------
def predict_and_output(drivers_df, rf, xgb):
    """Make predictions and output results"""
    X_pred = drivers_df.drop(columns=['driver', 'team'])
    
    # Get probabilities and rankings
    probs = rf.predict_proba(X_pred)[:, 1] if rf.predict_proba(X_pred).shape[1] > 1 else rf.predict_proba(X_pred)[:, 0]
    drivers_df['podium_prob'] = probs
    drivers_df['rank_score'] = xgb.predict(X_pred)
    
    # Sort by combined score
    drivers_df = drivers_df.sort_values(['podium_prob', 'rank_score'], ascending=False)
    
    # Convert to driver codes
    top3 = [get_driver_code(d) for d in drivers_df.head(3)['driver'].tolist()]
    
    confidence = float(100 * drivers_df.head(3)['podium_prob'].mean())
    
    # Generate key factors
    key_factors = {}
    for _, row in drivers_df.head(3).iterrows():
        factors = []
        if row['hot_streak']: factors.append("On hot streak")
        if row['monaco_podiums'] > 0: factors.append(f"{int(row['monaco_podiums'])} past Monaco podiums")
        if row['avg_pos_change'] > 0: factors.append(f"Average +{row['avg_pos_change']:.1f} positions gained")
        if row['fastest_lap_count'] > 0: factors.append(f"{int(row['fastest_lap_count'])} fastest laps")
        key_factors[get_driver_code(row['driver'])] = ", ".join(factors) if factors else "Consistent performer"
    
    # Output results
    output = {
        "predicted_podium": top3,
        "confidence_level": round(confidence, 1),
        "key_factors": key_factors
    }
    
    print(json.dumps(output, indent=2))
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Prediction saved to {OUTPUT_FILE}")

# ------------------------
# Main Execution
# ------------------------
def main():
    try:
        print("Loading 2024 season data...")
        season_data = load_season(2024)
        
        if not season_data:
            print("[ERROR] No season data loaded")
            return
        
        print(f"Found {len(season_data)} races in season")
        
        # Count races before Monaco
        races_before_monaco = 0
        monaco_found = False
        
        for race in season_data:
            if isinstance(race, dict) and 'race_name' in race:
                if 'Monaco' in race['race_name']:
                    monaco_found = True
                    break
                races_before_monaco += 1
        
        if not monaco_found:
            print("[WARNING] Monaco GP not found in data, using all available races")
            races_before_monaco = len(season_data)
        
        use_races = min(races_before_monaco, PRE_MONACO_COUNT)
        print(f"Using {use_races} races for training")
        
        # Extract features from races before Monaco
        hist_df = extract_race_features(season_data, use_races)
        
        if hist_df.empty:
            print("[ERROR] No race data extracted")
            return
        
        print(f"Extracted {len(hist_df)} driver records")
        
        # Build feature matrix
        drivers = hist_df['driver'].unique()
        features = []
        
        for driver in drivers:
            driver_data = hist_df[hist_df['driver'] == driver]
            
            features.append({
                'driver': driver,
                'team': driver_data['team'].mode()[0] if not driver_data['team'].mode().empty else "Unknown",
                'avg_pos_change': (driver_data['start_pos'] - driver_data['finish_pos']).mean(),
                'dnf_rate': driver_data['dnf'].mean(),
                'fastest_lap_count': driver_data['fastest_lap'].sum(),
                'hot_streak': detect_hot_streak(hist_df, driver)
            })
        
        drivers_df = pd.DataFrame(features)
        print(f"Created features for {len(drivers_df)} drivers")
        
        # Add Monaco history
        try:
            monaco_history = load_monaco_history()
            if not monaco_history.empty:
                monaco_counts = monaco_history.groupby('driver').size().to_dict()
                drivers_df['monaco_podiums'] = drivers_df['driver'].map(monaco_counts).fillna(0)
                print(f"Added Monaco history for {len(monaco_counts)} drivers")
            else:
                drivers_df['monaco_podiums'] = 0
                print("No Monaco history found")
        except Exception as e:
            print(f"[WARNING] Could not load Monaco history: {e}")
            drivers_df['monaco_podiums'] = 0
        
        # Prepare training data
        X = drivers_df[['avg_pos_change', 'dnf_rate', 'fastest_lap_count', 'hot_streak', 'monaco_podiums']]
        
        # Create labels (1 for podium finishers, 0 for others)
        y = []
        for _, row in hist_df.iterrows():
            y.append(1 if row['finish_pos'] <= 3 else 0)
        
        # Create groups for XGBoost (number of records per race)
        race_counts = hist_df.groupby(hist_df.index // 20).size().tolist()  # Assuming ~20 drivers per race
        group = race_counts if race_counts else [len(drivers)]
        
        print(f"Training models with {len(X)} samples...")
        
        # Train models
        rf, xgb = train_models(X, y, group)
        
        # Make predictions
        predict_and_output(drivers_df, rf, xgb)
        
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

# ------------------------
# Entry Point
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Monaco GP podium finishers')
    parser.add_argument('--db_dir', type=str, help='Path to directory containing JSON season data')
    parser.add_argument('--pre_monaco_count', type=int, default=7, help='Number of races before Monaco to use')
    parser.add_argument('--output_file', type=str, default='monaco_prediction.json', help='Output file path')
    parser.add_argument('--monaco_history_years', nargs='+', type=int, default=[2021, 2022, 2023], help='Years for Monaco history')
    
    args = parser.parse_args()
    
    # Update global variables if provided
    if args.db_dir:
        DB_DIR = args.db_dir
    if args.pre_monaco_count:
        PRE_MONACO_COUNT = args.pre_monaco_count
    if args.output_file:
        OUTPUT_FILE = args.output_file
    if args.monaco_history_years:
        MONACO_HISTORY_YEARS = args.monaco_history_years
    
    main()