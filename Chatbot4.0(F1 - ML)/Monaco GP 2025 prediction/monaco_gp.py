import os
import json
import glob
import argparse
from collections import defaultdict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRanker

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
            else:
                code = surname[:3].upper()  # Default first 3 letters of surname
            codes[full_name] = code
    return codes

def get_driver_code(full_name):
    """Get standardized 3-letter driver code"""
    return DRIVER_CODES.get(full_name, full_name.split()[-1][:3].upper())

# ------------------------
# Section: Data Loading
# ------------------------
def load_season(year):
    path = os.path.join(DB_DIR, f"F1_Seasons_Cleaned_{year}.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Initialize driver codes if we have championship data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'section' in item and item['section'] == 'championship_standings':
                    global DRIVER_CODES
                    DRIVER_CODES = generate_driver_codes(data[data.index(item)+1:])
                    break
        return data

# ------------------------
# Section: Feature Engineering
# ------------------------
def extract_race_features(races, up_to_race):
    """Process race data from the loaded JSON structure"""
    records = []
    
    for race in races[:up_to_race]:
        if not isinstance(race, dict) or 'data' not in race:
            continue
            
        # Find all race_results sections (there might be multiple)
        results_sections = [
            i for i, item in enumerate(race['data']) 
            if isinstance(item, dict) and item.get('section') == 'race_results'
        ]
        
        for section_idx in results_sections:
            # Process all entries after race_results marker
            for item in race['data'][section_idx+1:]:
                if not isinstance(item, dict) or 'driver' not in item:
                    continue  # Skip non-driver entries
                    
                try:
                    records.append({
                        'driver': item['driver'],
                        'team': item['team'],
                        'start_pos': int(item.get('starting position', 20)),
                        'finish_pos': int(item.get('finish position', 20)),
                        'points': int(item.get('points', 0)),
                        'fastest_lap': 1 if item.get('fastest lap time') else 0,
                        'dnf': 1 if str(item.get('dnf', 'No')).lower() == 'yes' else 0
                    })
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed driver data - {e}")
                    continue
                    
    return pd.DataFrame(records)

# Hot streak detection
def detect_hot_streak(df, driver_name):
    driver_points = df[df['driver'] == driver_name].sort_index()
    last3 = driver_points.tail(3)['points'].sum()
    avg = driver_points['points'].mean()
    return 1 if (avg and last3 > 1.5 * avg) else 0

# Monaco history features
def load_monaco_history():
    podiums = []
    for y in MONACO_HISTORY_YEARS:
        season = load_season(y)
        # find Monaco gp
        for race in season['data']:
            if 'Monaco Grand Prix' in race.get('race_name', ''):
                results = [d for d in race['data'] if d.get('section') == 'race_results'][1:]
                top3 = results[:3]
                for pos, d in enumerate(top3, start=1):
                    podiums.append({'driver': d['driver'], 'year': y, 'pos': pos, 'team': d['team']})
    return pd.DataFrame(podiums)

# ------------------------
# Section: Model Training
# ------------------------
def train_models(X, y_class, group):
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
    X_pred = drivers_df.drop(columns=['driver', 'team'])
    probs = rf.predict_proba(X_pred)[:, 1]
    drivers_df['podium_prob'] = probs
    drivers_df['rank_score'] = xgb.predict(X_pred)
    drivers_df = drivers_df.sort_values(['podium_prob', 'rank_score'], ascending=False)
    
    # Convert to driver codes
    top3 = [get_driver_code(d) for d in drivers_df.head(3)['driver'].tolist()]
    
    confidence = float(100 * drivers_df.head(3)['podium_prob'].mean())
    
    # Improved key factors
    key_factors = {}
    for _, row in drivers_df.head(3).iterrows():
        factors = []
        if row['hot_streak']: factors.append("On hot streak")
        if row['monaco_podiums']: factors.append(f"{row['monaco_podiums']} past Monaco podiums")
        if row['avg_pos_change'] > 0: factors.append(f"Average +{row['avg_pos_change']:.1f} positions gained")
        key_factors[get_driver_code(row['driver'])] = ", ".join(factors) or "Consistent performer"
    
    out = {
        "predicted_podium": top3,
        "confidence_level": round(confidence, 1),
        "key_factors": key_factors
    }
    print(json.dumps(out, indent=2))
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(out, f, indent=2)

# ------------------------
# Main Execution
# ------------------------
def main():
    try:
        season_data = load_season(2024)
        
        # Process races - looking for race_name entries
        races = []
        current_race = None
        
        for item in season_data:
            if isinstance(item, dict) and 'race_name' in item:
                if current_race:
                    races.append(current_race)
                current_race = {
                    'race_name': item['race_name'],
                    'data': []
                }
            elif current_race:
                current_race['data'].append(item)
        
        if current_race:
            races.append(current_race)
            
        print(f"Found {len(races)} races")
        if races:
            print(f"First race has {len(races[0]['data'])} data items")
            print("Sample data items:", [x for x in races[0]['data'] if isinstance(x, dict)][:3])

        # Historical features
        hist_df = extract_race_features(races, PRE_MONACO_COUNT)
        
        if hist_df.empty:
            print("[ERROR] No race data extracted. Possible issues:")
            print("- Check JSON structure matches expected format")
            print("- Verify race results contain proper driver data")
            if races:
                print(f"- First race structure: {races[0]}")
            return

        # Build feature matrix
        drivers = hist_df['driver'].unique()
        feats = []

        for d in drivers:
            sub = hist_df[hist_df['driver'] == d]
            feats.append({
                'driver': d,
                'team': sub['team'].mode()[0] if not sub['team'].mode().empty else "Unknown",
                'avg_pos_change': (sub['start_pos'] - sub['finish_pos']).mean(),
                'dnf_rate': sub['dnf'].mean(),
                'fastest_lap_count': sub['fastest_lap'].sum(),
                'hot_streak': detect_hot_streak(hist_df, d)
            })

        drivers_df = pd.DataFrame(feats)

        # Monaco history counts
        try:
            mdf = load_monaco_history()
            counts = mdf.groupby('driver').size().to_dict()
            drivers_df['monaco_podiums'] = drivers_df['driver'].map(counts).fillna(0)
        except Exception as e:
            print(f"[WARNING] Could not load Monaco history: {e}")
            drivers_df['monaco_podiums'] = 0

        # Prepare labels and group for training
        extracted_records = extract_race_features(races, PRE_MONACO_COUNT).to_dict('records')
        if not extracted_records:
            print("[ERROR] No extracted records for labeling.")
            return
            
        labels = [1 if r['finish_pos'] <= 3 else 0 for r in extracted_records]
        
        try:
            X = hist_df.drop(columns=['driver','team','points']).reset_index(drop=True)
        except KeyError as e:
            print(f"[ERROR] Missing expected column in hist_df: {e}")
            return
            
        y = labels

        # Grouping logic for XGBoost ranking
        try:
            races_per_driver = hist_df.groupby('driver').size()
            if len(races_per_driver.unique()) == 1:
                group = [races_per_driver[0]] * PRE_MONACO_COUNT
            else:
                group = list(races_per_driver)
        except Exception as e:
            print(f"[ERROR] Group calculation failed: {e}")
            group = [len(drivers)] * PRE_MONACO_COUNT

                # Train models
        try:
            rf, xgb = train_models(X, y, group)
        except Exception as e:
            print(f"[ERROR] Model training failed: {e}")
            return

        # Prediction and output
        try:
            predict_and_output(drivers_df, rf, xgb)
        except Exception as e:
            print(f"[ERROR] Prediction or output failed: {e}")
            return

    except Exception as e:
        print(f"[FATAL ERROR] Failed during main execution: {e}")

# ------------------------
# Constants and Entry Point
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_dir', type=str, required=True, help='Path to directory containing JSON season data')
    parser.add_argument('--pre_monaco_count', type=int, default=6, help='Number of races before Monaco to use for training')
    parser.add_argument('--output_file', type=str, default='monaco_prediction.json', help='Path to output prediction JSON')
    parser.add_argument('--monaco_history_years', nargs='+', type=int, default=[2019, 2021, 2022, 2023], help='Years to include for Monaco GP history')

    args = parser.parse_args()
    # Assign global constants
    global DB_DIR, PRE_MONACO_COUNT, OUTPUT_FILE, MONACO_HISTORY_YEARS
    DB_DIR = args.db_dir
    PRE_MONACO_COUNT = args.pre_monaco_count
    OUTPUT_FILE = args.output_file
    MONACO_HISTORY_YEARS = args.monaco_history_years

    main()