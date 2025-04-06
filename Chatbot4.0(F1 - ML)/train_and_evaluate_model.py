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
    Here, we predict 'grid_position' (or you might choose finishing position if available)
    using q1, q2, and q3 times. Adjust this according to your target.
    """
    # Use qualifying session times as features.
    X = df[['q1', 'q2', 'q3']].copy()
    # Target: grid_position (as a proxy for performance; replace with finishing position if available)
    y = df['grid_position']
    # Remove rows with NaN values in features or target
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
    
    # Load and prepare training data
    df_train = load_dataset(training_files)
    X_train, y_train = prepare_features(df_train)
    print("Training set:", X_train.shape, y_train.shape)
    
    # Load and prepare test data
    df_test = load_dataset(test_files)
    X_test, y_test = prepare_features(df_test)
    print("Test set:", X_test.shape, y_test.shape)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data
    predictions = model.predict(X_test)
    
    # Evaluate model performance
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\nModel Performance on Test Set:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    # Optionally, save the model or further analyze errors
