import json
import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#########################################
# Utility Functions
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
    Returns list of dicts with driver, constructor, grid_position, and qualifying times.
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
    Also extracts season from the file name (expects a 4-digit year).
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
    """Load multiple JSON files and combine into one DataFrame."""
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
    Prepare feature matrix X and target y from training/test data.
    Retains the 'driver' column for merging historical averages.
    Computes:
      - total_q_time: q1 + q2 + q3
      - Improvement metrics: q1_q2_diff, q2_q3_diff, q1_q3_diff
      - Variability: q_std (std-dev of q1, q2, q3)
    """
    X = df[['driver', 'q1', 'q2', 'q3']].copy()
    X['total_q_time'] = X['q1'] + X['q2'] + X['q3']
    X['q1_q2_diff'] = X['q1'] - X['q2']
    X['q2_q3_diff'] = X['q2'] - X['q3']
    X['q1_q3_diff'] = X['q1'] - X['q3']
    X['q_std'] = X[['q1', 'q2', 'q3']].std(axis=1)
    y = df['grid_position']
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    return X.loc[valid_idx], y.loc[valid_idx]

def prepare_new_race_features(df):
    """
    Prepare feature matrix for new race predictions.
    Assumes df has columns: driver, q1, q2, q3.
    """
    X = df[['driver', 'q1', 'q2', 'q3']].copy()
    X['total_q_time'] = X['q1'] + X['q2'] + X['q3']
    X['q1_q2_diff'] = X['q1'] - X['q2']
    X['q2_q3_diff'] = X['q2'] - X['q3']
    X['q1_q3_diff'] = X['q1'] - X['q3']
    X['q_std'] = X[['q1', 'q2', 'q3']].std(axis=1)
    return X

#########################################
# Model Training Section
#########################################
# Paths for training data (2020-2023) and test data (2024)
training_files = [
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
    r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
]
test_file = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Test\F1_Seasons_Cleaned_2024.json"

# Load training data.
df_train = load_dataset(training_files)
X_train, y_train = prepare_features(df_train)
print("Training set:", X_train.shape, y_train.shape)

# Historical Average Feature: Compute average total qualifying time per driver from training.
driver_avg = X_train.groupby('driver')['total_q_time'].mean().to_dict()
overall_avg = np.mean(list(driver_avg.values()))
X_train['driver_avg_q_time'] = X_train['driver'].map(driver_avg)

# Drop 'driver' column for modeling.
X_train_model = X_train.drop(columns=['driver'])

# Train the model.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_model, y_train)

#########################################
# Interactive Prediction for Selected 2024 Race
#########################################
print("\n--- Interactive Prediction for a 2024 Race ---")

# Load test data and get list of races.
df_test = load_data_with_season(test_file)
race_names = df_test['race_name'].unique()
print("\nRaces available in 2024:")
for idx, rname in enumerate(race_names, start=1):
    print(f"{idx}. {rname}")

# Ask user to select a race.
while True:
    try:
        choice = int(input("\nEnter the number corresponding to the race you want to predict: "))
        if 1 <= choice <= len(race_names):
            selected_race = race_names[choice - 1]
            break
        else:
            print("Invalid choice. Please choose a number from the list.")
    except ValueError:
        print("Please enter a valid number.")

# Filter the test data for the selected race.
df_selected = df_test[df_test['race_name'] == selected_race].copy()
if df_selected.empty:
    print("No data found for the selected race.")
    exit()

# Display the list of drivers and teams.
print(f"\nDrivers and teams for '{selected_race}':")
for idx, row in df_selected.iterrows():
    print(f"{row['driver']} ({row['constructor']}) - Q1: {row['q1']:.2f}, Q2: {row['q2']:.2f}, Q3: {row['q3']:.2f}")

confirm = input("\nIs this list correct? (Y/N): ").strip().lower()
if confirm != "y":
    print("Please edit the JSON data or manually create a new list. Exiting.")
    exit()

# Prepare features for the new race.
X_new = prepare_new_race_features(df_selected)
# Add historical average for each driver.
X_new['driver_avg_q_time'] = X_new['driver'].map(driver_avg).fillna(overall_avg)
# Drop driver column for modeling.
X_new_model = X_new.drop(columns=['driver'])
# Align new race features with training set.
X_new_model = X_new_model.reindex(columns=X_train_model.columns, fill_value=0)

# Predict grid positions using the trained model.
predictions_new = model.predict(X_new_model)

# Reset index of df_selected for safe iteration
df_selected_reset = df_selected.reset_index(drop=True)

# Output predictions.
print(f"\nPredicted Starting Grid Positions for '{selected_race}':")
for i in range(len(predictions_new)):
    pred_grid = round(predictions_new[i])
    driver = df_selected_reset.loc[i, 'driver']
    constructor = df_selected_reset.loc[i, 'constructor']
    print(f"{driver} ({constructor}): Predicted grid position {pred_grid}")
