import json
import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import shap




#########################################
# UTILITY FUNCTIONS
#########################################
CONSTRUCTOR_MAPPING = {
    "AlphaTauri": "RB F1 Team",
    "Toro Rosso": "RB F1 Team",
    "Sauber": "Stake F1 Team",
    # Add other mappings as needed
}

def map_constructor(old_name):
    """Normalize constructor/team names across seasons"""
    return CONSTRUCTOR_MAPPING.get(old_name, old_name)

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

def validate_race_data(race):
    """Validate that race data has required structure."""
    if not race.get("race_name"):
        print("Warning: Missing race name in race data")
        return False
    
    # Check for qualifying data
    has_qualifying = False
    for item in race.get("data", []):
        if isinstance(item, dict) and item.get("section") == "qualifying_results":
            has_qualifying = True
            break
    
    if not has_qualifying:
        print(f"Warning: No qualifying data found for {race.get('race_name')}")
        return False
    
    return True

def extract_round_number(race_name):
    """Extract round number from race name."""
    match = re.search(r"Round\s*(\d+)", race_name)
    if match:
        return int(match.group(1))
    return None

def get_race_keyword(race_name):
    """Extract the main keyword from race name, e.g., 'British Grand Prix'."""
    if "Grand Prix" in race_name:
        return race_name.split('(')[0].strip()
    return race_name.split()[0]  # Fallback to first word

#########################################
# DATA LOADING FUNCTIONS
#########################################
def load_data_with_season(file_path):
    """
    Load a JSON file and extract race-level data.
    Returns a list of race objects with season information.
    """
    season = "Unknown"
    match = re.search(r"(\d{4})", os.path.basename(file_path))
    if match:
        season = match.group(1)
    
    all_races = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            races = json.load(f)
            for race in races:
                race_name = race.get("race_name", "Unknown Race")
                race["race_name"] = race_name  # ensure field is present
                race["season"] = season
                
                # Add round number if available
                round_num = extract_round_number(race_name)
                if round_num:
                    race["round"] = round_num
                    
                if validate_race_data(race):
                    all_races.append(race)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
        
    return all_races

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


def load_track_characteristics(track_data_path=None):
    """
    Load circuit-specific data. If path is None, returns default characteristics.
    """
    # Default track data if file doesn't exist
    default_tracks = {
        "Monaco Grand Prix": {
            "circuit_type": "street",
            "lap_distance": 3.337,
            "avg_overtakes": 8,
            "high_degradation": False,
            "weather_variability": "low"
        },
        "Monza Grand Prix": {
            "circuit_type": "permanent",
            "lap_distance": 5.793,
            "avg_overtakes": 45,
            "high_degradation": True,
            "weather_variability": "medium"
        }
    }
    
    if not track_data_path or not os.path.exists(track_data_path):
        print("Using default track characteristics")
        return default_tracks
    
    try:
        with open(track_data_path, 'r') as f:
            tracks = json.load(f)
            
        track_info = {}
        for track in tracks:
            name = track.get("name")
            track_info[name] = {
                "circuit_type": track.get("circuit_type", "unknown"),
                "lap_distance": track.get("lap_distance", 5.0),
                "avg_overtakes": track.get("avg_overtakes", 20),
                "high_degradation": track.get("high_degradation", False),
                "weather_variability": track.get("weather_variability", "low")
            }
        return track_info
    except Exception as e:
        print(f"Error loading track data: {e}. Using defaults.")
        return default_tracks

#########################################
# DATA EXTRACTION FUNCTIONS
#########################################
def extract_qualifying_results(data):
    """
    Extract qualifying results from race data.
    Returns list of dicts with driver, constructor, grid_position, and Q times.
    """
    results = []
    in_qualifying = False
    qualifying_weather = "dry"  # default weather
    
    for item in data:
        if isinstance(item, dict) and "section" in item:
            in_qualifying = (item["section"] == "qualifying_results")
            # Check for weather info
            if in_qualifying and "weather" in item:
                qualifying_weather = item["weather"]
            continue
        
        if in_qualifying and isinstance(item, dict) and "position" in item:
            grid_position = item.get("position", "0")
            # Handle grid position as string or int
            try:
                grid_position = int(grid_position)
            except ValueError:
                grid_position = 20  # Default to back of grid if invalid
                
            record = {
                "driver": item.get("driver"),
                "constructor":  map_constructor(item.get("constructor")),
                "grid_position": grid_position,
                "q1": time_to_seconds(item.get("q1", "")),
                "q2": time_to_seconds(item.get("q2", "")),
                "q3": time_to_seconds(item.get("q3", "")),
                "qualifying_weather": qualifying_weather
            }
            results.append(record)
    
    return results

def extract_race_results(data):
    """
    Extract race results from race data.
    Returns a list of dictionaries with driver, finish_position, and dnf indicator.
    """
    results = []
    in_results = False
    race_weather = "dry"  # default weather
    
    for item in data:
        if isinstance(item, dict) and "section" in item:
            in_results = (item["section"] == "race_results")
            # Check for weather info
            if in_results and "weather" in item:
                race_weather = item["weather"]
            continue
        
        if in_results and isinstance(item, dict) and "driver" in item:
            # Process finish position using the JSON key "finish position"
            finish_pos = item.get("finish position")
            try:
                finish_pos = float(finish_pos)
            except (ValueError, TypeError):
                finish_pos = np.nan
                
            # Process DNF status
            dnf_status = item.get("dnf", "No")
            dnf = 1 if isinstance(dnf_status, str) and dnf_status.lower() == "yes" else 0
            
            # Extract DNF cause if available
            dnf_cause = item.get("dnf_cause", "unknown")
            mechanical_dnf = 1 if dnf and "mechanical" in str(dnf_cause).lower() else 0
            crash_dnf = 1 if dnf and "crash" in str(dnf_cause).lower() else 0
            
            # Map the JSON key "finish position" to "finish_position" for consistency downstream
            results.append({
                "driver": item.get("driver"),
                "finish_position": finish_pos,
                "team": map_constructor(item.get("team")),
                "dnf": dnf,
                "mechanical_dnf": mechanical_dnf,
                "crash_dnf": crash_dnf,
                "race_weather": race_weather
            })
    
    return results


def extract_championship_standings(data):
    """
    Extract championship standings from race data.
    Returns list of dictionaries with driver, position, points, and wins.
    """
    standings = []
    in_standings = False
    
    for item in data:
        if isinstance(item, dict) and "section" in item:
            in_standings = (item["section"] == "championship_standings")
            continue
        
        if in_standings and isinstance(item, dict) and "driver_name" in item:
            champ_pos = item.get("pos")
            champ_pts = item.get("pts")
            
            try:
                champ_pos = int(champ_pos)
            except (ValueError, TypeError):
                champ_pos = np.nan
                
            try:
                champ_pts = float(champ_pts)
            except (ValueError, TypeError):
                champ_pts = 0.0
                
            standings.append({
                "driver": item["driver_name"],
                "champ_pos": champ_pos,
                "champ_pts": champ_pts,
                "champ_wins": int(item.get("wins", 0))
            })
    
    return standings

def get_latest_championship_standings(races, selected_round):
    """
    Get championship standings from the round immediately before selected_round.
    """
    # Filter races by round number
    previous_races = [r for r in races if r.get("round", 0) < selected_round]
    
    if not previous_races:
        print("No previous races found for championship data")
        return {}
    
    # Sort by round and get the most recent
    previous_races.sort(key=lambda x: x.get("round", 0), reverse=True)
    latest_race = previous_races[0]
    
    # Extract championship standings from this race
    standings = {}
    standings_data = extract_championship_standings(latest_race.get("data", []))
    
    for item in standings_data:
        driver = item["driver"]
        standings[driver] = {
            "champ_pos": item.get("champ_pos", 20),
            "champ_pts": item.get("champ_pts", 0),
            "champ_wins": item.get("champ_wins", 0)
        }
    
    return standings

#########################################
# FEATURE ENGINEERING FUNCTIONS
#########################################
def calculate_rolling_dnf_rate(driver, races, window=5):
    """
    Calculate rolling DNF rate from last N races.
    """
    # Extract all race results
    driver_races = []
    for race in races:
        race_results = extract_race_results(race.get("data", []))
        for result in race_results:
            if result.get("driver") == driver:
                result["season"] = race.get("season", "0")
                result["round"] = race.get("round", 0)
                driver_races.append(result)
    
    # Sort by season and round
    driver_races.sort(key=lambda x: (x.get("season", "0"), x.get("round", 0)))
    
    # Get the most recent races within window
    recent_races = driver_races[-window:] if len(driver_races) >= window else driver_races
    
    if not recent_races:
        return 0.0
    
    dnf_count = sum(1 for race in recent_races if race.get("dnf", 0) == 1)
    return dnf_count / len(recent_races)

def calculate_weighted_finish_position(driver, training_races, current_year=2024, race_keyword=None):
    """
    Calculate weighted average finish position with exponential decay for older seasons.
    Optionally filter by race_keyword to get track-specific performance.
    """
    finishes = []
    weights = []
    
    for race in training_races:
        if race_keyword and race_keyword.lower() not in race.get("race_name", "").lower():
            continue
            
        race_results = extract_race_results(race.get("data", []))
        for result in race_results:
            if result.get("driver") == driver and not np.isnan(result.get("finish_position", np.nan)):
                year = int(race.get("season", current_year))
                # Apply exponential decay weight
                weight = 0.5 ** (int(current_year) - year + 1)
                finishes.append(result["finish_position"])
                weights.append(weight)
    
    if not finishes:
        return np.nan
    
    # Calculate weighted average
    weighted_sum = sum(pos * weight for pos, weight in zip(finishes, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else np.nan

def identify_track_specialists(training_races, race_keyword, n_clusters=3):
    """
    Group drivers into performance tiers at specific tracks using K-means.
    Returns a dict mapping drivers to cluster labels (0, 1, 2).
    Lower cluster number = better performance.
    """
    track_performances = {}
    
    # Collect historical performance data
    for race in training_races:
        if race_keyword.lower() in race.get("race_name", "").lower():
            race_results = extract_race_results(race.get("data", []))
            for result in race_results:
                driver = result.get("driver")
                if driver and not np.isnan(result.get("finish_position", np.nan)):
                    track_performances.setdefault(driver, []).append(result["finish_position"])
    
    # Calculate average performance
    avg_performances = {
        driver: np.mean(finishes) for driver, finishes in track_performances.items()
        if len(finishes) >= 2  # Require at least 2 races at this track
    }
    
    if len(avg_performances) < n_clusters:
        # Not enough data for clustering
        return {driver: 0 for driver in avg_performances}
    
    # Perform K-means clustering
    X = np.array(list(avg_performances.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Sort clusters by performance (cluster 0 = best performers)
    cluster_means = {}
    for i in range(n_clusters):
        mask = (clusters == i)
        if np.any(mask):
            cluster_means[i] = np.mean(X[mask])
    
    # Sort clusters by mean value (ascending)
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
    cluster_map = {old: new for new, (old, _) in enumerate(sorted_clusters)}
    
    # Map drivers to clusters
    driver_clusters = {}
    for i, driver in enumerate(avg_performances.keys()):
        original_cluster = int(clusters[i])
        driver_clusters[driver] = cluster_map[original_cluster]
    
    return driver_clusters

def calculate_position_change_trend(driver, training_races, race_keyword):
    """
    Calculate average positions gained/lost per driver at a specific track.
    Positive value = tends to gain positions during race
    Negative value = tends to lose positions
    """
    position_changes = []
    
    for race in training_races:
        if race_keyword.lower() not in race.get("race_name", "").lower():
            continue
            
        # Get qualifying and race results
        qual_results = extract_qualifying_results(race.get("data", []))
        race_results = extract_race_results(race.get("data", []))
        
        # Find driver's grid and finish positions
        grid_pos = None
        for q in qual_results:
            if q.get("driver") == driver:
                grid_pos = q.get("grid_position")
                break
                
        finish_pos = None
        for r in race_results:
            if r.get("driver") == driver:
                finish_pos = r.get("finish_position")
                break
        
        if grid_pos is not None and finish_pos is not None and not np.isnan(finish_pos):
            # Calculate positions gained (negative = gained, positive = lost)
            # Subtract finish from grid so positive values mean improvement
            position_changes.append(grid_pos - finish_pos)
    
    if not position_changes:
        return 0.0
        
    return np.mean(position_changes)

def handle_new_driver(driver, constructor, df_selected, training_races, current_season_races, race_keyword):
    """
    Handle drivers with no historical data by:
    1. Using teammate's performance if available
    2. Using constructor's historical performance
    3. Fallback to midfield default
    """
    # Find teammate from the current race entry
    teammate = None
    for _, row in df_selected.iterrows():
        if row["driver"] != driver and row["constructor"] == constructor:
            teammate = row["driver"]
            break
    
    # Calculate features for substitute driver
    features = {
        "hist_avg": 10.0,  # Default to midfield
        "track_specialist": 1,  # Middle tier
        "position_change": 0.0,  # Neutral
    }
    
    # 1. Check if teammate has historical data at this track
    if teammate:
        teammate_hist = calculate_weighted_finish_position(teammate, training_races, race_keyword=race_keyword)
        if not np.isnan(teammate_hist):
            # Use teammate data with slight penalty
            features["hist_avg"] = teammate_hist * 1.1
            
            # Get teammate's track specialist rating
            track_specialists = identify_track_specialists(training_races, race_keyword)
            if teammate in track_specialists:
                features["track_specialist"] = track_specialists[teammate]
                
            # Get teammate's position change trend
            pos_change = calculate_position_change_trend(teammate, training_races, race_keyword)
            features["position_change"] = pos_change * 0.8  # Slightly reduced

    # 2. Use constructor historical performance (if teammate approach failed)
    if features["hist_avg"] == 10.0:
        # Find all drivers for this constructor in current season
        constructor_drivers = []
        for race in current_season_races:
            qual_data = extract_qualifying_results(race.get("data", []))
            for q in qual_data:
                if q.get("constructor") == constructor and q.get("driver") not in constructor_drivers:
                    constructor_drivers.append(q.get("driver"))
        
        # Calculate average historical performance for those drivers
        constructor_hist = []
        for d in constructor_drivers:
            hist = calculate_weighted_finish_position(d, training_races, race_keyword=race_keyword)
            if not np.isnan(hist):
                constructor_hist.append(hist)
        
        if constructor_hist:
            features["hist_avg"] = np.mean(constructor_hist)
    
    return features

def normalize_features(features_df):
    """Normalize numerical features to 0-1 range"""
    normalized_df = features_df.copy()
    
    for col in ['champ_pts', 'grid_position', 'hist_avg', 'best_q_time']:
        if col in features_df.columns:
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            if max_val > min_val:  # Avoid division by zero
                normalized_df[col + '_norm'] = (features_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col + '_norm'] = 0.5
    
    return normalized_df

def prepare_quali_features(df):
    """Prepare qualifying-based features using best session time."""
    X = df.copy()
    
    # Determine best qualifying time (Q3 > Q2 > Q1)
    X['best_q_time'] = X.apply(
        lambda row: row['q3'] if pd.notna(row['q3']) else 
                    row['q2'] if pd.notna(row['q2']) else 
                    row['q1'],
        axis=1
    )
    
    
    # Calculate stage reached
    X['qual_stage'] = X.apply(
        lambda row: 3 if pd.notna(row['q3']) else 
                   2 if pd.notna(row['q2']) else 
                   1,
        axis=1
    )
    
    # Drop individual Q times if not needed elsewhere
    # X = X.drop(['q1', 'q2', 'q3'], axis=1)
    
    # Convert weather to numerical feature (existing code)
    weather_mapping = {"dry": 0.0, "damp": 0.5, "wet": 1.0}
    X['quali_weather_num'] = X['qualifying_weather'].map(
        lambda x: weather_mapping.get(str(x).lower(), 0.0)
    )

    # Add these new calculations:
    X['q1_q2_diff'] = X['q2'] - X['q1']  # Time improvement Q1→Q2
    X['q2_q3_diff'] = X['q3'] - X['q2']  # Time improvement Q2→Q3 
    X['q_std'] = X[['q1','q2','q3']].std(axis=1)  # Consistency metric
    
    
    return X

#########################################
# MODEL BUILDING FUNCTIONS
#########################################
def build_training_dataset(training_races, current_year=2024):
    """
    Build a comprehensive training dataset from historical races.
    Combines qualifying, race results, and derived features.
    """
    all_data = []
    
    for race in training_races:
        race_name = race.get("race_name", "")
        season = int(race.get("season", 0))
        race_keyword = get_race_keyword(race_name)
        
        # Get qualifying results
        qual_results = extract_qualifying_results(race.get("data", []))
        
        # Get race results
        race_results = extract_race_results(race.get("data", []))
        
        # Get championship standings
        champ_data = extract_championship_standings(race.get("data", []))
        champ_dict = {item["driver"]: item for item in champ_data}
        
        # For each driver with qualifying data
        for qual in qual_results:
            driver_full = qual.get("driver")
            # Extract surname from qualifying name (e.g. "George Russell" -> "Russell")
            driver_surname = driver_full.split()[-1] if " " in driver_full else driver_full
            
            # Find race result using surname match
            race_result = next(
                (r for r in race_results 
                 if (r.get("driver", "").split()[-1] == driver_surname)),
                None
            )
            
            if not race_result:
                continue  # Skip if no matching race result
            
            # Get championship data (using original full name)
            champ_info = champ_dict.get(driver_full, {})
            
            # Create record combining all data
            record = {
                "race_name": race_name,
                "race_keyword": race_keyword,
                "season": season,
                "driver": driver_full,  # Keep original qualifying name
                "constructor": qual.get("constructor"),
                "grid_position": qual.get("grid_position"),
                "finish_position": race_result.get("finish_position"),
                "q_time_total": qual.get("best_q_time"),
                "dnf": race_result.get("dnf", 0),
                "mechanical_dnf": race_result.get("mechanical_dnf", 0),
                "crash_dnf": race_result.get("crash_dnf", 0),
                "q1": qual.get("q1"),
                "q2": qual.get("q2"),
                "q3": qual.get("q3"),
                "qualifying_weather": qual.get("qualifying_weather", "dry"),
                "race_weather": race_result.get("race_weather", "dry"),
                "champ_pos": champ_info.get("champ_pos", np.nan),
                "champ_pts": champ_info.get("champ_pts", 0),
                "champ_wins": champ_info.get("champ_wins", 0)
            }
            
            all_data.append(record)
    
    # Rest of the function remains unchanged
    df = pd.DataFrame(all_data)
    
    
    # Drop rows with missing finish position
    df = df.dropna(subset=['finish_position'])
    
    # Add qualifying-derived features
    df = prepare_quali_features(df)
    
    # Add historical performance features
    df['weighted_hist_finish'] = df.apply(
        lambda row: calculate_weighted_finish_position(
            row['driver'], 
            [r for r in training_races if int(r.get("season", 0)) < row['season']], 
            row['season'],
            row['race_keyword']
        ), 
        axis=1
    )
    
    # Fill missing historical values with current grid position
    df['weighted_hist_finish'] = df['weighted_hist_finish'].fillna(df['grid_position'])
    
    # Calculate DNF rate (requires list of previous races for each driver/season)
    def get_dnf_rate(row):
        prev_races = [
            r for r in training_races 
            if int(r.get("season", 0)) <= row['season'] and 
               get_race_keyword(r.get("race_name", "")) != row['race_keyword']
        ]
        return calculate_rolling_dnf_rate(row['driver'], prev_races)
    
    df['dnf_rate'] = df.apply(get_dnf_rate, axis=1)
    
    # Drop rows with missing key features
    df = df.dropna(subset=['grid_position', 'q_time_total'])
    
    return df

def train_unified_model(df_train):
    """
    Train an XGBoost model for race finish prediction.
    Returns the trained model and feature list.
    """
    # Prepare feature matrix
    features = [
        'grid_position', 'champ_pos', 'champ_pts', 'champ_wins',
        'best_q_time', 'q1_q2_diff', 'q2_q3_diff', 'q_std', 'qual_stage',
        'quali_weather_num', 'weighted_hist_finish', 'dnf_rate'
    ]
    
    # Drop rows with NaN in any feature
    X = df_train[features].copy()
    X = X.dropna()
    y = df_train.loc[X.index, 'finish_position']
    
    # Train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X, y)
    
    # Save feature names for inference
    model.feature_names_ = features
    
    return model

def evaluate_model(model, df_test):
    """Evaluate model performance using MAE and Top-3 precision."""
    # Prepare feature matrix
    X_test = df_test[model.feature_names_].copy()
    X_test = X_test.dropna()
    y_true = df_test.loc[X_test.index, 'finish_position']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Top-3 precision
    true_top3_idx = y_true[y_true <= 3].index
    pred_top3_idx = pd.Series(y_pred, index=X_test.index).sort_values().head(3).index
    
    true_top3 = set(df_test.loc[true_top3_idx, 'driver'])
    pred_top3 = set(df_test.loc[pred_top3_idx, 'driver'])
    
    if len(true_top3) > 0:
        precision = len(true_top3.intersection(pred_top3)) / len(true_top3)
    else:
        precision = 0.0
    
    # Calculate positions gained/lost accuracy
    true_pos_change = df_test['grid_position'] - df_test['finish_position']
    pred_pos_change = df_test['grid_position'] - pd.Series(y_pred, index=X_test.index)
    
    # Check if direction matches (improvement or decline)
    direction_match = (true_pos_change * pred_pos_change > 0).mean()
    
    return {
        'mae': mae,
        'top3_precision': precision,
        'direction_accuracy': direction_match
    }

def cross_validate_model(df, n_splits=5):
    """Perform time-based cross-validation."""
    # Sort by season
    df = df.sort_values('season')
    
    # Create time series split (each split represents one season)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    for train_idx, test_idx in tscv.split(df):
        # Get train/test sets
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
        # Train model
        model = train_unified_model(df_train)
        
        # Evaluate
        metrics = evaluate_model(model, df_test)
        results.append(metrics)
    
    # Average results
    avg_results = {
        'mae': np.mean([r['mae'] for r in results]),
        'top3_precision': np.mean([r['top3_precision'] for r in results]),
        'direction_accuracy': np.mean([r['direction_accuracy'] for r in results])
    }
    
    return avg_results

def analyze_feature_importance(model, X):
    """Analyze feature importance using SHAP values."""
    # Calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    # Plot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    plt.close()
    
    # Return feature importance as dictionary
    importance_dict = {}
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    for i, name in enumerate(X.columns):
        importance_dict[name] = mean_abs_shap[i]
    
    return importance_dict

#########################################
# PREDICTION FEATURE PREPARATION
#########################################
def prepare_prediction_features(df_selected, model, training_races, test_races, selected_round, race_keyword):
    """
    Prepare feature matrix for unified model prediction.
    This function combines qualifying features, latest championship data,
    historical performance, rolling DNF rates, and applies normalization.
    """
    # Prepare qualifying features
    X_new = prepare_quali_features(df_selected)
    
    # Get latest championship standings from test races before the selected round
    champ_data = get_latest_championship_standings(test_races, selected_round)
    
    # Calculate rolling DNF rates for each driver in the selected race
    dnf_rates = {}
    for _, row in df_selected.iterrows():
        driver = row['driver']
        dnf_rates[driver] = calculate_rolling_dnf_rate(driver, test_races, window=5)
    
    # Calculate historical performance using exponential decay weighting.
    hist_performance = {}
    for _, row in df_selected.iterrows():
        driver = row['driver']
        # Only consider training races that match the race keyword
        hist_avg = calculate_weighted_finish_position(driver, training_races, current_year=int(row.get("season", 2024)), race_keyword=race_keyword)
        # If no historical data, fallback to the current grid position
        hist_performance[driver] = hist_avg if not np.isnan(hist_avg) else row['grid_position']

    X_new['champ_pos'] = X_new['driver'].map(lambda d: champ_data.get(d, {}).get('champ_pos', 20))
    X_new['champ_pts'] = X_new['driver'].map(lambda d: champ_data.get(d, {}).get('champ_pts', 0))
    X_new['champ_wins'] = X_new['driver'].map(lambda d: champ_data.get(d, {}).get('champ_wins', 0))
    
    # Add the calculated features to the prediction dataframe.
    X_new['dnf_rate'] = X_new['driver'].apply(lambda d: dnf_rates.get(d, 0.0))
    X_new['weighted_hist_finish'] = X_new['driver'].apply(lambda d: hist_performance.get(d, 10.0))
    
    # For drivers with no historical data, use new driver handling.
    for idx, row in X_new.iterrows():
        driver = row['driver']
        # Here, we assume that if weighted_hist_finish is not set or equals a default value,
        # then the driver is new.
        if pd.isna(hist_performance.get(driver)):
            new_feats = handle_new_driver(driver, row.get('constructor', ''), df_selected, training_races, test_races, race_keyword)
            X_new.at[idx, 'weighted_hist_finish'] = new_feats['hist_avg']
    
    # Normalize numerical features to ensure they are on a similar scale.
    X_new = normalize_features(X_new)
    
    # Ensure the prediction dataframe contains the same feature columns that the model was trained on.
    # (model.feature_names_ was saved during training.)
    X_new_model = X_new[model.feature_names_]
    return X_new_model

#########################################
# VISUALIZATION FUNCTION
#########################################
def visualize_predictions(predictions, race_name):
    """Plot predicted positions using a horizontal bar chart."""
    drivers = [f"{p['driver']} ({p['constructor']})" for p in predictions]
    positions = [p['predicted_position'] for p in predictions]

    plt.figure(figsize=(10, 6))
    plt.barh(drivers, positions, color='skyblue')
    plt.gca().invert_yaxis()  # Top driver at the top
    plt.title(f"Predicted Finishing Positions: {race_name}")
    plt.xlabel("Position")
    plt.tight_layout()
    plt.show()

#########################################
# MAIN EXECUTION BLOCK
#########################################
if __name__ == "__main__":
    # File paths for training and test data
    training_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
    ]
    test_file = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Test\F1_Seasons_Cleaned_2024.json"

    # Load historical (training) and 2024 (test) races data
    training_races = load_dataset(training_files)
    test_races = load_data_with_season(test_file)

    # Build the training dataset using combined features and then train the unified XGBoost model.
    df_train = build_training_dataset(training_races)
    model = train_unified_model(df_train)

    # Let the user select a race from the available 2024 races.
    race_names = []
    for race in test_races:
        if race.get("race_name") not in race_names:
            race_names.append(race.get("race_name"))
    print("\nRaces available in 2024:")
    for idx, name in enumerate(race_names, 1):
        print(f"{idx}. {name}")

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

    # Collect all race objects for the selected race
    selected_race_objs = [race for race in test_races if race.get("race_name") == selected_race_name]
    if not selected_race_objs:
        print("No race data found for the selected race.")
        sys.exit(1)

    # Extract qualifying results for the selected race into a DataFrame.
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

    # (Optional) Confirm the list of drivers and constructors with the user.
    print(f"\nDrivers and teams for '{selected_race_name}':")
    for idx, row in df_selected.iterrows():
        print(f"{row['driver']} ({row['constructor']}) - Q1: {row['q1']:.2f}, Q2: {row['q2']:.2f}, Q3: {row['q3']:.2f}")
    confirm = input("\nIs this list correct? (Y/N): ").strip().lower()
    if confirm != "y":
        print("Please update your data manually. Exiting.")
        sys.exit(1)

    # Extract the selected round from the race name, assuming format "Round X".
    m = re.search(r"Round\s*(\d+)", selected_race_name)
    selected_round = int(m.group(1)) if m else 999
    race_keyword = get_race_keyword(selected_race_name)
    
    # Prepare features for prediction from the selected race data.
    X_pred = prepare_prediction_features(df_selected, model, training_races, test_races, selected_round, race_keyword)
    predictions = model.predict(X_pred)

    # Combine predictions with driver names for output.
    prediction_results = []
    for i, row in df_selected.iterrows():
        prediction_results.append({
            "driver": row['driver'],
            "constructor": row.get('constructor', ''),
            "predicted_position": round(predictions[i])
        })

        # Display predicted finishing positions.
    print(f"\nPredicted Finishing Positions for '{selected_race_name}':")
    for result in prediction_results:
        print(f"{result['driver']} ({result['constructor']}): Predicted Position {result['predicted_position']}")

# Visualize predictions using a horizontal bar chart.
visualize_predictions(prediction_results, selected_race_name)

