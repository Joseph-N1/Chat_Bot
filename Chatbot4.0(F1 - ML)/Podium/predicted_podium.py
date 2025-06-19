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
# Updated paths to reflect new folder structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory of this script

# Try multiple possible locations for the DataBase folder
def find_database_dir():
    possible_db_dirs = [
        os.path.join(os.path.dirname(SCRIPT_DIR), "DataBase"),  # Original location
        r"c:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\DataBase",  # Found location
        os.path.join(SCRIPT_DIR, "DataBase"),
        os.path.join(os.getcwd(), "DataBase"),
    ]
    
    for possible_dir in possible_db_dirs:
        test_file = os.path.join(possible_dir, "F1_Seasons_Cleaned_2024.json")
        if os.path.exists(test_file):
            print(f"✅ Found F1 data directory: {possible_dir}")
            return possible_dir
    
    # Fallback to original
    return os.path.join(os.path.dirname(SCRIPT_DIR), "DataBase")

DB_DIR = find_database_dir()
OUTPUT_DIR = SCRIPT_DIR  # Save outputs in the same Podium folder as the script

YEARS = [2024]
HISTORY_YEARS = [2021, 2022, 2023]
OUTPUT_FILE = "race_prediction_2024.json"

# Race name mapping for historical data matching
RACE_NAME_MAPPING = {
    "Monaco": ["Monaco Grand Prix", "Monaco"],
    "Canadian": ["Canadian Grand Prix", "Canada"],
    "Spanish": ["Spanish Grand Prix", "Spain"],
    "Austrian": ["Austrian Grand Prix", "Austria"],
    "British": ["British Grand Prix", "Great Britain", "Britain"],
    "Hungarian": ["Hungarian Grand Prix", "Hungary"],
    "Belgian": ["Belgian Grand Prix", "Belgium"],
    "Dutch": ["Dutch Grand Prix", "Netherlands"],
    "Italian": ["Italian Grand Prix", "Italy"],
    "Singapore": ["Singapore Grand Prix"],
    "Japanese": ["Japanese Grand Prix", "Japan"],
    "Qatar": ["Qatar Grand Prix"],
    "United States": ["United States Grand Prix", "USA", "US"],
    "Mexican": ["Mexican Grand Prix", "Mexico"],
    "Brazilian": ["Brazilian Grand Prix", "Brazil"],
    "Las Vegas": ["Las Vegas Grand Prix"],
    "Abu Dhabi": ["Abu Dhabi Grand Prix"]
}

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

def get_user_input_race():
    """Get target race from user input with helpful prompts"""
    print("\n" + "="*60)
    print("🏁 F1 RACE PREDICTION SYSTEM")
    print("="*60)
    
    print("\nSelect the race you want to predict:")
    print("-" * 40)
    print("Examples:")
    print("• 'Monaco' or 'Monaco Grand Prix'")
    print("• 'Canadian' or 'Canadian Grand Prix'") 
    print("• 'Spanish', 'British', 'Hungarian', etc.")
    print("• 'Singapore', 'Japanese', 'Brazilian', etc.")
    
    while True:
        race_input = input("\n🎯 Enter race name: ").strip()
        
        if not race_input:
            print("❌ Please enter a race name.")
            continue
            
        # Find matching race key
        race_key = None
        for key, aliases in RACE_NAME_MAPPING.items():
            if any(alias.lower() in race_input.lower() or race_input.lower() in alias.lower() 
                   for alias in aliases):
                race_key = key
                break
        
        if race_key:
            print(f"✅ Selected: {race_key} Grand Prix")
            return race_key
        else:
            print(f"❌ Race '{race_input}' not recognized.")
            print("💡 Try: Monaco, Canadian, Spanish, British, Hungarian, etc.")

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

def find_target_race_position(season_data, target_race_key):
    """Find the position of target race in the season"""
    for i, race in enumerate(season_data):
        if isinstance(race, dict) and 'race_name' in race:
            race_name = race['race_name']
            # Check if any of the target race aliases match
            if any(alias.lower() in race_name.lower() 
                   for alias in RACE_NAME_MAPPING.get(target_race_key, [])):
                return i, race_name
    return None, None

# ------------------------
# Section: Enhanced Feature Engineering
# ------------------------
def extract_qualifying_data(race_data):
    """Extract qualifying results from race data with enhanced position tracking"""
    qualifying_results = {}
    
    # Find qualifying section
    qualifying_section_idx = None
    for i, item in enumerate(race_data):
        if isinstance(item, dict) and item.get('section') == 'qualifying_results':
            qualifying_section_idx = i
            break
    
    if qualifying_section_idx is not None:
        # Process qualifying data
        for item in race_data[qualifying_section_idx+1:]:
            if not isinstance(item, dict) or 'driver' not in item:
                continue
            if 'section' in item:  # Hit another section
                break
            
            try:
                # Extract position - this is crucial for the model
                position = int(item.get('position', 20))
                qualifying_results[item['driver']] = {
                    'quali_pos': position,
                    'q1_time': item.get('q1', ''),
                    'q2_time': item.get('q2', ''),
                    'q3_time': item.get('q3', ''),
                    # Add normalized qualifying score (higher is better)
                    'quali_score': max(0, (21 - position) / 20)  # Normalize 0-1, position 1 = 1.0
                }
            except (ValueError, KeyError):
                continue
                
    return qualifying_results

def extract_race_features(races, up_to_race):
    """Process race data from the loaded JSON structure with enhanced features"""
    records = []
    
    for race in races[:up_to_race]:
        if not isinstance(race, dict) or 'data' not in race:
            continue
            
        race_data = race['data']
        
        # Extract qualifying data with enhanced position tracking
        qualifying_data = extract_qualifying_data(race_data)
        
        # Find race_results section
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
                driver_name = item['driver']
                quali_info = qualifying_data.get(driver_name, {
                    'quali_pos': 20, 
                    'quali_score': 0.05  # Default low score
                })
                
                records.append({
                    'driver': driver_name,
                    'team': item['team'],
                    'start_pos': int(item.get('starting position', 20)),
                    'finish_pos': int(item.get('finish position', 20)),
                    'quali_pos': quali_info['quali_pos'],
                    'quali_score': quali_info['quali_score'],  # New enhanced field
                    'points': int(item.get('points', 0)),
                    'fastest_lap': 1 if item.get('fastest lap time') and item.get('fastest lap time').strip() else 0,
                    'dnf': 1 if str(item.get('dnf', 'No')).lower() == 'yes' else 0
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping malformed driver data - {e}")
                continue
                
    return pd.DataFrame(records)

def calculate_recent_form(df, driver_name, window=3):
    """Calculate recent form based on last N races"""
    driver_data = df[df['driver'] == driver_name]
    if len(driver_data) < window:
        window = len(driver_data)
    
    if window == 0:
        return 0.0
    
    # Sort by index to get chronological order
    driver_data = driver_data.sort_index()
    recent_data = driver_data.tail(window)
    
    # Calculate weighted recent form score
    form_score = 0.0
    total_weight = 0.0
    
    for i, (_, race) in enumerate(recent_data.iterrows()):
        # More recent races have higher weight
        weight = (i + 1) / window
        
        # Score based on finish position (lower is better)
        position_score = max(0, (21 - race['finish_pos']) / 20)  # Normalized 0-1
        
        # Bonus for points
        points_bonus = min(race['points'] / 25, 1.0)  # Normalized, max 25 points
        
        # Much smaller penalty for DNF
        dnf_penalty = 0.1 if race['dnf'] else 0  # Reduced from 0.5 to 0.1
        
        race_score = (position_score + points_bonus - dnf_penalty) * weight
        form_score += race_score
        total_weight += weight
    
    return form_score / total_weight if total_weight > 0 else 0.0

def calculate_average_starting_position(df, driver_name):
    """Calculate average starting position for a driver"""
    driver_data = df[df['driver'] == driver_name]
    
    if len(driver_data) == 0:
        return 15.0  # Default middle-grid position if no data
    
    # Calculate average starting position
    avg_start_pos = driver_data['start_pos'].mean()
    
    return avg_start_pos

# Hot streak detection (enhanced)
def detect_hot_streak(df, driver_name):
    """Enhanced hot streak detection based on recent performance"""
    driver_data = df[df['driver'] == driver_name]
    if len(driver_data) < 3:
        return 0
        
    # Sort by index to get chronological order
    driver_data = driver_data.sort_index()
    
    # Check last 3-4 races for consistent good performance
    recent_races = driver_data.tail(4)
    
    # Count podiums and points finishes in recent races
    recent_podiums = (recent_races['finish_pos'] <= 3).sum()
    recent_points = (recent_races['points'] > 0).sum()
    recent_avg_pos = recent_races['finish_pos'].mean()
    
    # Hot streak criteria (more sophisticated)
    is_hot = (
        recent_podiums >= 2 or  # 2+ podiums in last 4 races
        (recent_points >= 3 and recent_avg_pos <= 8) or  # Consistent points with good avg
        recent_races.tail(3)['points'].sum() >= recent_races['points'].mean() * 4  # Recent points surge
    )
    
    return 1 if is_hot else 0

# Track-specific history features (enhanced)
def load_track_history(target_race_key):
    """Load historical results for the target track with more detail"""
    track_results = []
    
    for year in HISTORY_YEARS:
        try:
            season_data = load_season(year)
            
            # Find target race
            for race in season_data:
                if isinstance(race, dict) and 'race_name' in race:
                    race_name = race['race_name']
                    # Check if any of the target race aliases match
                    if any(alias.lower() in race_name.lower() 
                           for alias in RACE_NAME_MAPPING.get(target_race_key, [])):
                        
                        race_data = race['data']
                        
                        # Get qualifying data
                        qualifying_data = extract_qualifying_data(race_data)
                        
                        # Find race results section
                        results_section_idx = None
                        for i, item in enumerate(race_data):
                            if isinstance(item, dict) and item.get('section') == 'race_results':
                                results_section_idx = i
                                break
                        
                        if results_section_idx is not None:
                            # Get all race results
                            for item in race_data[results_section_idx+1:]:
                                if isinstance(item, dict) and 'driver' in item and 'section' not in item:
                                    driver_name = item['driver']
                                    quali_info = qualifying_data.get(driver_name, {
                                        'quali_pos': 20,
                                        'quali_score': 0.05
                                    })
                                    
                                    track_results.append({
                                        'driver': driver_name,
                                        'year': year,
                                        'finish_pos': int(item.get('finish position', 20)),
                                        'start_pos': int(item.get('starting position', 20)),
                                        'quali_pos': quali_info['quali_pos'],
                                        'quali_score': quali_info['quali_score'],
                                        'points': int(item.get('points', 0)),
                                        'team': item['team']
                                    })
                                elif 'section' in item:
                                    break
                        break
        except Exception as e:
            print(f"Warning: Could not load {target_race_key} history for {year}: {e}")
            continue
    
    return pd.DataFrame(track_results)

# ------------------------
# Section: Model Training
# ------------------------
def train_models(X, y_class):
    """Train classification model with balanced feature importance"""
    # RandomForest with parameters to balance features properly
    rf = RandomForestClassifier(
        n_estimators=300,  # More trees for stability
        max_depth=6,       # Reduce depth to prevent overfitting
        min_samples_split=10,  # Higher minimum to prevent overfitting
        min_samples_leaf=5,   # Higher minimum to prevent overfitting
        max_features=0.7,     # Use 70% of features to balance influence
        random_state=42,
        class_weight='balanced'  # Balance classes
    )
    rf.fit(X, y_class)
    
    # Print feature importance
    feature_names = X.columns
    importances = rf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Feature Importance:")
    for feature, importance in feature_importance:
        print(f"   {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    return rf

# ------------------------
# Section: Prediction & Output
# ------------------------
def predict_and_output(drivers_df, rf, target_race_name, races_used):
    """Make predictions and output results with enhanced analysis"""
    X_pred = drivers_df.drop(columns=['driver', 'team'])
    
    # Get probabilities
    probs = rf.predict_proba(X_pred)
    # Handle both binary and multiclass cases
    if probs.shape[1] > 1:
        drivers_df['podium_prob'] = probs[:, 1]
    else:
        drivers_df['podium_prob'] = probs[:, 0]
    
    # Sort by podium probability
    drivers_df = drivers_df.sort_values('podium_prob', ascending=False)
    
    # Convert to driver codes
    top3 = [get_driver_code(d) for d in drivers_df.head(3)['driver'].tolist()]
    
    confidence = float(100 * drivers_df.head(3)['podium_prob'].mean())
    
    # Generate enhanced key factors
    key_factors = {}
    for _, row in drivers_df.head(3).iterrows():
        factors = []
        if row['hot_streak']: factors.append("On hot streak")
        if row['track_podiums'] > 0: factors.append(f"{int(row['track_podiums'])} past podiums at this track")
        if row['avg_pos_change'] > 0: factors.append(f"Average +{row['avg_pos_change']:.1f} positions gained")
        if row['fastest_lap_count'] > 0: factors.append(f"{int(row['fastest_lap_count'])} fastest laps")
        if row['recent_form'] > 0.6: factors.append(f"Strong recent form ({row['recent_form']:.2f})")
        if row['avg_starting_position'] <= 5: factors.append(f"Good qualifier (avg P{row['avg_starting_position']:.1f})")
        if row['avg_quali_improvement'] > 2: factors.append(f"Race day improver (+{row['avg_quali_improvement']:.1f} avg)")
        
        key_factors[get_driver_code(row['driver'])] = ", ".join(factors) if factors else "Consistent performer"
    
    # Output results
    output = {
        "race_predicted": target_race_name,
        "races_used_for_training": races_used,
        "predicted_podium": top3,
        "confidence_level": round(confidence, 1),
        "key_factors": key_factors,
        "detailed_predictions": [
            {
                "driver": get_driver_code(row['driver']),
                "full_name": row['driver'],
                "team": row['team'],
                "podium_probability": round(row['podium_prob'] * 100, 1),
                "recent_form": round(row['recent_form'], 3),
                "avg_starting_position": round(row['avg_starting_position'], 1)
            }
            for _, row in drivers_df.head(10).iterrows()
        ]
    }
    
    print("\n" + "="*60)
    print("🏆 ENHANCED RACE PREDICTION RESULTS")
    print("="*60)
    print(json.dumps(output, indent=2))
    
    # Save to file in the Podium folder
    race_filename = target_race_name.replace(" ", "_").lower() + "_enhanced_prediction_2024.json"
    output_path = os.path.join(OUTPUT_DIR, race_filename)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Enhanced prediction saved to {output_path}")

# ------------------------
# Main Execution
# ------------------------
def main():
    try:
        # Get target race from user
        target_race_key = get_user_input_race()
        
        print(f"\n🔄 Loading 2024 season data from {DB_DIR}...")
        season_data = load_season(2024)
        
        if not season_data:
            print("[ERROR] No season data loaded")
            return
        
        print(f"Found {len(season_data)} races in season")
        
        # Find target race position
        target_race_idx, target_race_name = find_target_race_position(season_data, target_race_key)
        
        if target_race_idx is None:
            print(f"[ERROR] {target_race_key} Grand Prix not found in 2024 season data")
            print("Available races:")
            for i, race in enumerate(season_data):
                if isinstance(race, dict) and 'race_name' in race:
                    print(f"  {i+1}. {race['race_name']}")
            return
        
        print(f"🎯 Target race: {target_race_name} (Round {target_race_idx + 1}) (Position {target_race_idx + 1})")
        
        # Use all races before the target race
        races_before_target = target_race_idx
        print(f"📊 Using {races_before_target} races for training")
        
        if races_before_target == 0:
            print("[ERROR] Cannot predict first race of season - no previous data available")
            return
        
        # Extract enhanced features from races before target
        hist_df = extract_race_features(season_data, races_before_target)
        
        if hist_df.empty:
            print("[ERROR] No race data extracted")
            return
        
        print(f"✅ Extracted {len(hist_df)} driver records with starting position data")
        
        # Build enhanced feature matrix
        drivers = hist_df['driver'].unique()
        features = []
        labels = []
        
        for driver in drivers:
            driver_data = hist_df[hist_df['driver'] == driver]
            
            # Calculate enhanced aggregated features
            avg_pos_change = (driver_data['start_pos'] - driver_data['finish_pos']).mean()
            
            # MAJOR CHANGE: Drastically reduce DNF impact to less than 5%
            raw_dnf_rate = driver_data['dnf'].mean()
            dnf_rate = min(raw_dnf_rate * 0.02, 0.03)  # Max 3% influence, usually 2%
            
            fastest_lap_count = driver_data['fastest_lap'].sum()
            hot_streak = detect_hot_streak(hist_df, driver)
            recent_form = calculate_recent_form(hist_df, driver)
            
            # NEW: Replace quali_performance with average starting position
            avg_starting_position = calculate_average_starting_position(hist_df, driver)
            
            # Calculate qualifying vs race position improvement rate
            quali_improvements = []
            for _, race in driver_data.iterrows():
                if race['start_pos'] > 0 and race['finish_pos'] > 0:
                    improvement = race['start_pos'] - race['finish_pos']  # Positive = gained positions
                    quali_improvements.append(improvement)
            
            avg_quali_improvement = np.mean(quali_improvements) if quali_improvements else 0
            avg_quali_improvement = abs(avg_quali_improvement) * 2.0  # Double the improvement factor
            
            # Calculate podium rate as target
            podium_finishes = (driver_data['finish_pos'] <= 3).sum()
            total_races = len(driver_data)
            podium_rate = podium_finishes / total_races if total_races > 0 else 0
            
            # Enhanced classification: high podium performers vs others
            is_podium_contender = 1 if (podium_rate > 0.15 or podium_finishes > 1 or recent_form > 0.7 or avg_starting_position <= 6) else 0
            
            features.append({
                'driver': driver,
                'team': driver_data['team'].mode()[0] if not driver_data['team'].mode().empty else "Unknown",
                'avg_pos_change': avg_pos_change,
                'dnf_rate': dnf_rate,  # Now heavily weighted down to 2-3%
                'fastest_lap_count': fastest_lap_count,
                'hot_streak': hot_streak,
                'recent_form': recent_form,
                'avg_starting_position': avg_starting_position,  # NEW: Replace quali_performance
                'avg_quali_improvement': avg_quali_improvement  # Still 2x boosted
            })
            
            labels.append(is_podium_contender)
        
        drivers_df = pd.DataFrame(features)
        print(f"✅ Created enhanced features for {len(drivers_df)} drivers")
        
        # Add enhanced track-specific history
        try:
            track_history = load_track_history(target_race_key)
            if not track_history.empty:
                # Calculate track metrics
                track_podiums = track_history.groupby('driver')['finish_pos'].apply(lambda x: (x <= 3).sum()).to_dict()
                track_avg_pos = track_history.groupby('driver')['finish_pos'].mean().to_dict()
                track_start_avg = track_history.groupby('driver')['start_pos'].mean().to_dict()
                
                # Add track metrics to drivers dataframe
                drivers_df['track_podiums'] = drivers_df['driver'].map(track_podiums).fillna(0)
                drivers_df['track_avg_pos'] = drivers_df['driver'].map(track_avg_pos).fillna(15.0)
                drivers_df['track_start_avg'] = drivers_df['driver'].map(track_start_avg).fillna(15.0)
                
                # Normalize track_avg_pos to be a positive feature (lower position = higher score)
                drivers_df['track_avg_pos'] = drivers_df['track_avg_pos'].apply(lambda x: max(0, (21 - x) / 20))
                
                print(f"✅ Added enhanced {target_race_key} track history with starting position data")
            else:
                drivers_df['track_podiums'] = 0
                drivers_df['track_avg_pos'] = 0.25
                drivers_df['track_start_avg'] = 15.0
                print(f"⚠️  No {target_race_key} track history found")
        except Exception as e:
            print(f"[WARNING] Could not load {target_race_key} track history: {e}")
            drivers_df['track_podiums'] = 0
            drivers_df['track_avg_pos'] = 0.25
            drivers_df['track_start_avg'] = 15.0
        
        # Prepare enhanced training data
        feature_columns = [
            'avg_pos_change', 'dnf_rate', 'fastest_lap_count', 'hot_streak', 
            'recent_form', 'avg_starting_position', 'avg_quali_improvement',
            'track_podiums', 'track_avg_pos', 'track_start_avg'
        ]
        
        X = drivers_df[feature_columns]
        y = labels
        
        print(f"🔧 Training enhanced model with {len(X)} samples and {len(y)} labels...")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Features: {list(X.columns)}")
        
        # Train enhanced model
        rf = train_models(X, y)
        
        # Make enhanced predictions
        predict_and_output(drivers_df, rf, target_race_name, races_before_target)
        
    except KeyboardInterrupt:
        print("\n❌ Prediction cancelled by user")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

# ------------------------
# Entry Point
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced F1 race podium prediction')
    parser.add_argument('--db_dir', type=str, help='Path to directory containing JSON season data')
    parser.add_argument('--race', type=str, help='Target race name (e.g., Monaco, Canadian, Spanish)')
    parser.add_argument('--output_file', type=str, help='Output file path')
    parser.add_argument('--history_years', nargs='+', type=int, default=[2021, 2022, 2023], help='Years for track history')
    
    args = parser.parse_args()
    
    # Update global variables if provided
    if args.db_dir:
        DB_DIR = args.db_dir
    if args.output_file:
        OUTPUT_FILE = args.output_file
    if args.history_years:
        HISTORY_YEARS = args.history_years
    
    # If race provided via command line, skip user input
    if args.race:
        # Find matching race key
        target_race_key = None
        for key, aliases in RACE_NAME_MAPPING.items():
            if any(alias.lower() in args.race.lower() or args.race.lower() in alias.lower() 
                   for alias in aliases):
                target_race_key = key
                break
        
        if target_race_key:
            print(f"🎯 Command line race selected: {target_race_key} Grand Prix")
            # Temporarily override the input function
            original_input = get_user_input_race
            get_user_input_race = lambda: target_race_key
            main()
            get_user_input_race = original_input
        else:
            print(f"❌ Race '{args.race}' not recognized in command line argument")
            main()
    else:
        main()