import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
    Returns list of dictionaries with driver, constructor, position,
    and qualifying times (converted to seconds).
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

def load_data_with_season(file_path):
    """
    Load a JSON file and extract qualifying results.
    Also extract season from the file name (expects a 4-digit year).
    """
    season = "Unknown"
    match = re.search(r'(\d{4})', os.path.basename(file_path))
    if match:
        season = match.group(1)
        
    all_results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        races = json.load(f)
        for race in races:
            race_name = race.get("race_name", "Unknown Race")
            q_results = extract_qualifying_results(race.get("data", []))
            for rec in q_results:
                rec["race_name"] = race_name
                rec["season"] = season
            all_results.extend(q_results)
    return pd.DataFrame(all_results)

def load_dataset(file_paths):
    """Load multiple season files and combine into one DataFrame."""
    df_list = []
    for path in file_paths:
        if os.path.exists(path):
            df = load_data_with_season(path)
            df_list.append(df)
        else:
            print(f"File not found: {path}")
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def prepare_features(df):
    """
    Prepare the feature matrix X and target variable y.
    Features include:
      - q1, q2, q3 times
      - total_q_time: sum of q1, q2, and q3
      - Improvement metrics: q1_q2_diff, q2_q3_diff, q1_q3_diff
      - Variability: standard deviation (q_std) of q1, q2, and q3
    The 'driver' column is retained so that we can merge in driver historical averages.
    Target:
      - grid_position (as a proxy for performance)
    """
    # Retain driver column along with qualifying times.
    X = df[['driver', 'q1', 'q2', 'q3']].copy()
    
    # Create aggregated and improvement features.
    X['total_q_time'] = X['q1'] + X['q2'] + X['q3']
    X['q1_q2_diff'] = X['q1'] - X['q2']
    X['q2_q3_diff'] = X['q2'] - X['q3']
    X['q1_q3_diff'] = X['q1'] - X['q3']
    X['q_std'] = X[['q1', 'q2', 'q3']].std(axis=1)
    
    # Target: grid_position (as a proxy for performance)
    y = df['grid_position']
    
    # Remove rows with NaN values in features or target.
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    return X.loc[valid_idx], y.loc[valid_idx]

if __name__ == "__main__":
    # File paths for training data (2020-2023) and test data (2024)
    training_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
    ]
    test_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Test\F1_Seasons_Cleaned_2024.json"
    ]
    
    # Load full data (df_train and df_test include all columns)
    df_train = load_dataset(training_files)
    df_test = load_dataset(test_files)
    
    # Prepare features (retaining the 'driver' column for merging)
    X_train, y_train = prepare_features(df_train)
    X_test, y_test = prepare_features(df_test)
    
    print("Training set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)
    
    # --- Historical Average Feature: Compute driver average total qualifying time ---
    # Compute driver averages from training data:
    driver_avg = X_train.groupby('driver')['total_q_time'].mean().to_dict()
    
    # Add historical average as a new feature to training set.
    X_train['driver_avg_q_time'] = X_train['driver'].map(driver_avg)
    
    # For the test set, merge the historical average from training.
    # If a driver in the test set is not found in training, fill with overall mean.
    overall_avg = np.mean(list(driver_avg.values()))
    X_test['driver_avg_q_time'] = X_test['driver'].map(driver_avg).fillna(overall_avg)
    
    # Drop the 'driver' column as it's no longer needed for modeling.
    X_train = X_train.drop(columns=['driver'])
    X_test = X_test.drop(columns=['driver'])
    
    # Align columns between training and test sets if needed.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Initialize and train the model.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data.
    predictions = model.predict(X_test)
    
    # Evaluate model performance.
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\nModel Performance on Test Set:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")