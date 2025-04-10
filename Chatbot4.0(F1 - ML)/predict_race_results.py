import json
import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#########################################
# UTILITY FUNCTIONS
#########################################
def time_to_seconds(time_str):
    """Convert time string 'M:SS.mmm' to total seconds."""
    if not time_str or not isinstance(time_str, str):
        return np.nan
    match = re.match(r"(\d+):(\d+\.\d+)", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return np.nan

def extract_qualifying_results(data):
    """
    Extract qualifying results from race data.
    Returns list of dicts with driver, constructor, grid_position, and Q times.
    """
    results = []
    in_qualifying = False
    for item in data:
        if "section" in item:
            in_qualifying = (item["section"] == "qualifying_results")
            continue
        if in_qualifying and "position" in item:
            record = {
                "driver": item.get("driver"),
                "constructor": item.get("constructor"),
                "grid_position": int(item.get("position", 0)),
                "q1": time_to_seconds(item.get("q1", "")),
                "q2": time_to_seconds(item.get("q2", "")),
                "q3": time_to_seconds(item.get("q3", ""))
            }
            results.append(record)
    return results

def extract_championship_standings(data):
    """
    Extract championship standings from a race data.
    Returns a list of dictionaries with driver_name, pos, pts, and wins.
    """
    standings = []
    for item in data:
        if "section" in item and item["section"] == "championship_standings":
            continue
        # Assume standing items have 'driver_name'
        if "driver_name" in item:
            standings.append({
                "driver": item["driver_name"],
                "champ_pos": item.get("pos", np.nan),
                "champ_pts": item.get("pts", np.nan),
                "champ_wins": item.get("wins", np.nan)
            })
    return standings

def extract_race_results(data):
    """
    Extract race results from a race data.
    Returns a list of dictionaries with driver, finish_position, and dnf indicator.
    """
    results = []
    for item in data:
        if "section" in item and item["section"] == "race_results":
            continue
        if "driver" in item and "finish position" in item:
            finish_pos = item.get("finish position")
            # Sometimes finish position might be numeric or string; cast to float.
            try:
                finish_pos = float(finish_pos)
            except:
                finish_pos = np.nan
            dnf = 1 if str(item.get("dnf", "No")).lower() == "yes" else 0
            results.append({
                "driver": item.get("driver"),
                "finish_position": finish_pos,
                "dnf": dnf
            })
    return results

def load_data_with_season(file_path):
    """
    Load a JSON file and extract race-level data.
    Returns a DataFrame that includes qualifying results and also retains the raw race data.
    """
    season = "Unknown"
    match = re.search(r"(\d{4})", os.path.basename(file_path))
    if match:
        season = match.group(1)
    all_races = []
    with open(file_path, 'r', encoding='utf-8') as f:
        races = json.load(f)
        for race in races:
            race_name = race.get("race_name", "Unknown Race")
            race["race_name"] = race_name  # ensure field is present
            race["season"] = season
            all_races.append(race)
    return all_races  # return list of race objects

def load_dataset(file_paths):
    """Load multiple JSON files and return list of race objects."""
    all_races = []
    for path in file_paths:
        if os.path.exists(path):
            races = load_data_with_season(path)
            all_races.extend(races)
        else:
            print(f"File not found: {path}")
    return all_races

#########################################
# FEATURE PREPARATION FUNCTIONS (for grid model)
#########################################
def prepare_features(df):
    """
    Prepare feature matrix X and target y from race data (for grid model).
    Uses qualifying times and derived metrics.
    """
    X = df[['driver', 'q1', 'q2', 'q3']].copy()
    X['total_q_time'] = X['q1'] + X['q2'] + X['q3']
    X['q1_q2_diff'] = X['q1'] - X['q2']
    X['q2_q3_diff'] = X['q2'] - X['q3']
    X['q1_q3_diff'] = X['q1'] - X['q3']
    X['q_std'] = X[['q1', 'q2', 'q3']].std(axis=1)
    X['qual_stage'] = X.apply(lambda row: 3 if pd.notna(row['q3']) 
                                         else (2 if pd.notna(row['q2']) else 1), axis=1)
    y = df['grid_position']
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    return X.loc[valid_idx], y.loc[valid_idx]

def prepare_new_race_features(df):
    """
    Prepare features for new race predictions.
    """
    X = df[['driver', 'q1', 'q2', 'q3']].copy()
    X['total_q_time'] = X['q1'] + X['q2'] + X['q3']
    X['q1_q2_diff'] = X['q1'] - X['q2']
    X['q2_q3_diff'] = X['q2'] - X['q3']
    X['q1_q3_diff'] = X['q1'] - X['q3']
    X['q_std'] = X[['q1', 'q2', 'q3']].std(axis=1)
    X['qual_stage'] = X.apply(lambda row: 3 if pd.notna(row['q3']) 
                                         else (2 if pd.notna(row['q2']) else 1), axis=1)
    return X

#########################################
# COMPOSITE PREDICTION FUNCTION
#########################################
def composite_prediction(features):
    """
    Compute composite finishing position prediction using a weighted sum:
      - 50%: current race grid prediction (grid_pred)
      - 10%: championship position (champ_pos)
      - 10%: championship points (champ_pts)
      - 2.5%: championship wins (champ_wins)
      - 2.5%: DNF penalty (dnf_count * 5)
      - 15%: historical average finish (hist_avg)
      - 10%: pattern score (pattern_score)
    """
    dnf_penalty = features['dnf_count'] * 5
    finish_pred = (
        0.5 * features['grid_pred'] +
        0.1 * features['champ_pos'] +
        0.1 * features['champ_pts'] +
        0.025 * features['champ_wins'] +
        0.025 * dnf_penalty +
        0.15 * features['hist_avg'] +
        0.1 * features['pattern_score']
    )
    return finish_pred

#########################################
# EXTRACTION FUNCTIONS FOR 2024 DATA
#########################################
def extract_championship_data(races, selected_round):
    """
    From the list of race objects from 2024, extract championship standings from
    all races with round number less than selected_round.
    Returns a dictionary mapping driver names to averaged champ stats.
    """
    champ_stats = {}
    count = {}
    for race in races:
        m = re.search(r"Round\s*(\d+)", race.get("race_name", ""))
        if not m:
            continue
        round_num = int(m.group(1))
        if round_num >= selected_round:
            continue
        # Find championship_standings section:
        for item in race.get("data", []):
            if item.get("section") == "championship_standings":
                for stat in race.get("data", []):
                    if "driver_name" in stat:
                        driver = stat["driver_name"]
                        champ_stats.setdefault(driver, {"champ_pos": 0, "champ_pts": 0, "champ_wins": 0})
                        count.setdefault(driver, 0)
                        champ_stats[driver]["champ_pos"] += stat.get("pos", 0)
                        champ_stats[driver]["champ_pts"] += stat.get("pts", 0)
                        champ_stats[driver]["champ_wins"] += stat.get("wins", 0)
                        count[driver] += 1
                break
    # Average the stats
    for driver in champ_stats:
        if count[driver]:
            champ_stats[driver]["champ_pos"] /= count[driver]
            champ_stats[driver]["champ_pts"] /= count[driver]
            champ_stats[driver]["champ_wins"] /= count[driver]
    return champ_stats

def extract_dnf_data(races, selected_round):
    """
    From races in 2024 with round < selected_round, extract DNF counts per driver.
    """
    dnf_counts = {}
    for race in races:
        m = re.search(r"Round\s*(\d+)", race.get("race_name", ""))
        if not m:
            continue
        round_num = int(m.group(1))
        if round_num >= selected_round:
            continue
        for item in race.get("data", []):
            if item.get("section") == "race_results":
                # Process race_results items
                for res in race.get("data", []):
                    if "driver" in res and "dnf" in res:
                        driver = res["driver"]
                        dnf = 1 if str(res.get("dnf", "No")).lower() == "yes" else 0
                        dnf_counts[driver] = dnf_counts.get(driver, 0) + dnf
                break
    return dnf_counts

def extract_hist_data(training_races, race_keyword):
    """
    From the training dataset (2020-2023), filter races that match the race_keyword (e.g., 'British Grand Prix')
    and compute the average finishing position per driver.
    Assumes each race object has a race_results section with a 'finish position' field.
    """
    hist_finishes = {}
    count = {}
    for race in training_races:
        if race_keyword.lower() in race.get("race_name", "").lower():
            # Look for race_results section.
            for item in race.get("data", []):
                if item.get("section") == "race_results":
                    for res in race.get("data", []):
                        if "driver" in res and "finish position" in res:
                            driver = res["driver"]
                            try:
                                finish = float(res.get("finish position"))
                            except:
                                finish = np.nan
                            if np.isnan(finish):
                                continue
                            hist_finishes[driver] = hist_finishes.get(driver, 0) + finish
                            count[driver] = count.get(driver, 0) + 1
                    break
    # Compute average finish
    for driver in hist_finishes:
        if count[driver]:
            hist_finishes[driver] /= count[driver]
    return hist_finishes

def extract_pattern_data(training_races, race_keyword):
    """
    From training data, for races that match race_keyword, compute a simple pattern score.
    For each driver, if they finished in the top 3 in at least 50% of races, assign -1 (bonus), else 0.
    """
    pattern = {}
    occurrences = {}
    for race in training_races:
        if race_keyword.lower() in race.get("race_name", "").lower():
            for item in race.get("data", []):
                if item.get("section") == "race_results":
                    for res in race.get("data", []):
                        if "driver" in res and "finish position" in res:
                            driver = res["driver"]
                            try:
                                finish = float(res.get("finish position"))
                            except:
                                continue
                            occurrences[driver] = occurrences.get(driver, 0) + 1
                            if finish <= 3:
                                pattern[driver] = pattern.get(driver, 0) + 1
                    break
    pattern_score = {}
    for driver in occurrences:
        frac = pattern.get(driver, 0) / occurrences[driver]
        pattern_score[driver] = -1 if frac >= 0.5 else 0
    return pattern_score

#########################################
# MAIN SCRIPT
#########################################
import pandas as pd  # Ensure pandas is imported
import math

# File paths
training_files = [
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
]
test_file = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Test\F1_Seasons_Cleaned_2024.json"

# Load training races (list of race objects) and also test races.
training_races = load_dataset(training_files)
test_races = load_data_with_season(test_file)

# --- TRAIN GRID MODEL ---
# For grid model training, we need to extract qualifying results from training races.
qual_results = []
for race in training_races:
    # For each race, get qualifying section and merge with race name and season.
    q_results = extract_qualifying_results(race.get("data", []))
    for rec in q_results:
        rec["race_name"] = race.get("race_name", "")
        rec["season"] = race.get("season", "")
    qual_results.extend(q_results)
df_train_qual = pd.DataFrame(qual_results)
X_train, y_train = prepare_features(df_train_qual)
print("Training set for grid model:", X_train.shape, y_train.shape)

# Historical average of total qualifying time from training data.
driver_avg_grid = X_train.groupby('driver')['total_q_time'].mean().to_dict()
overall_avg_grid = np.mean(list(driver_avg_grid.values()))
X_train['driver_avg_q_time'] = X_train['driver'].map(driver_avg_grid)
X_train_model = X_train.drop(columns=['driver'])
grid_model = RandomForestRegressor(n_estimators=100, random_state=42)
grid_model.fit(X_train_model, y_train)

# --- SELECT A 2024 RACE ---
# Get all race names from 2024 test races.
race_names = []
for race in test_races:
    if race.get("race_name") not in race_names:
        race_names.append(race.get("race_name"))
print("\nRaces available in 2024:")
for idx, r in enumerate(race_names, 1):
    print(f"{idx}. {r}")
    
while True:
    try:
        choice = int(input("\nEnter the number corresponding to the race to predict finishing positions: "))
        if 1 <= choice <= len(race_names):
            selected_race_name = race_names[choice - 1]
            break
        else:
            print("Invalid choice.")
    except ValueError:
        print("Please enter a valid number.")

# Gather all race objects for the selected race.
selected_race_objs = [race for race in test_races if race.get("race_name") == selected_race_name]
if not selected_race_objs:
    print("No race data found for the selected race.")
    sys.exit(1)

# For simplicity, we assume the round number is in the race name as "Round X"
m = re.search(r"Round\s*(\d+)", selected_race_name)
selected_round = int(m.group(1)) if m else 999

# From the test races, extract drivers for the selected race using qualifying results.
qual_results_selected = []
for race in selected_race_objs:
    q_results = extract_qualifying_results(race.get("data", []))
    for rec in q_results:
        rec["race_name"] = race.get("race_name")
        rec["season"] = race.get("season")
        qual_results_selected.append(rec)
df_selected = pd.DataFrame(qual_results_selected)

if df_selected.empty:
    print("No qualifying data for the selected race.")
    sys.exit(1)

# Display drivers and teams for confirmation.
print(f"\nDrivers and teams for '{selected_race_name}':")
for idx, row in df_selected.iterrows():
    print(f"{row['driver']} ({row['constructor']}) - Q1: {row['q1']:.2f}, Q2: {row['q2']:.2f}, Q3: {row['q3']:.2f}")
confirm = input("\nIs this list correct? (Y/N): ").strip().lower()
if confirm != "y":
    print("Please update your data manually. Exiting.")
    sys.exit(1)

# Prepare features for the new race.
X_new = prepare_new_race_features(df_selected)
X_new['driver_avg_q_time'] = X_new['driver'].map(driver_avg_grid).fillna(overall_avg_grid)
X_new_model = X_new.drop(columns=['driver'])
X_new_model = X_new_model.reindex(columns=X_train_model.columns, fill_value=0)
grid_predictions = grid_model.predict(X_new_model)
df_selected = df_selected.reset_index(drop=True)

# --- EXTRACT EXTRA FEATURES FROM 2024 DATA ---
# Championship data: from all races in 2024 with round < selected_round.
champ_data = extract_championship_data(test_races, selected_round)
dnf_data = extract_dnf_data(test_races, selected_round)

# Historical data: from training races that match the selected race keyword (e.g., "British Grand Prix")
race_keyword = selected_race_name.split('(')[0].strip()  # e.g., "British Grand Prix"
hist_data = extract_hist_data(training_races, race_keyword)
pattern_data = extract_pattern_data(training_races, race_keyword)

#########################################
# COMPOSITE PREDICTION: FINISHING POSITIONS
#########################################
print(f"\nPredicted Finishing Positions for '{selected_race_name}':")
for i in range(len(df_selected)):
    driver = df_selected.loc[i, 'driver']
    # Current grid prediction from Q times.
    grid_pred = round(grid_predictions[i])
    
    # Championship features (if available)
    champ_feats = champ_data.get(driver, {"champ_pos": 10, "champ_pts": 50, "champ_wins": 0})
    # DNF count from previous races in 2024 (default 0)
    dnf_count = dnf_data.get(driver, 0)
    # Historical average finishing position from training data (default 10)
    hist_avg = hist_data.get(driver, 10)
    # Pattern score (default 0)
    pat_score = pattern_data.get(driver, 0)
    
    features = {
        "grid_pred": grid_pred,
        "champ_pos": champ_feats["champ_pos"],
        "champ_pts": champ_feats["champ_pts"],
        "champ_wins": champ_feats["champ_wins"],
        "dnf_count": dnf_count,
        "hist_avg": hist_avg,
        "pattern_score": pat_score
    }
    finish_pred = composite_prediction(features)
    print(f"{driver} ({df_selected.loc[i, 'constructor']}): Predicted finishing position {round(finish_pred)}")
