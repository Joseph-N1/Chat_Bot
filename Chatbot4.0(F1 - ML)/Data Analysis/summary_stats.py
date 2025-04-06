import json
import glob
import pandas as pd
import numpy as np
import os
import re

def time_to_seconds(time_str):
    """
    Convert time string in format M:SS.mmm to total seconds (float).
    Returns np.nan if the string is empty or invalid.
    """
    if not time_str or not isinstance(time_str, str):
        return np.nan
    # Regular expression to capture minutes, seconds and milliseconds
    match = re.match(r"(\d+):(\d+\.\d+)", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return np.nan

def extract_qualifying_results(data):
    """
    Given a race data list, extract qualifying results records.
    Returns a list of dicts with the relevant fields.
    """
    results = []
    # Flag to track the current section
    in_qualifying = False
    for item in data:
        # Check if the item is a section marker
        if "section" in item:
            # Switch on the section name
            in_qualifying = (item["section"] == "qualifying_results")
            continue
        if in_qualifying:
            # Expect qualifying records to have a "position" field
            if "position" in item:
                record = {
                    "driver": item.get("driver"),
                    "constructor": item.get("constructor"),
                    "position": int(item.get("position", 0)),
                    "q1": time_to_seconds(item.get("q1", "")),
                    "q2": time_to_seconds(item.get("q2", "")),
                    "q3": time_to_seconds(item.get("q3", ""))
                }
                results.append(record)
    return results

def load_training_data(file_paths):
    """
    Load JSON files from provided paths and combine qualifying results.
    """
    all_results = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            races = json.load(f)
            for race in races:
                race_name = race.get("race_name", "Unknown Race")
                q_results = extract_qualifying_results(race.get("data", []))
                # Add race_name for context if needed
                for rec in q_results:
                    rec["race_name"] = race_name
                all_results.extend(q_results)
    return pd.DataFrame(all_results)

def calculate_summary_statistics(df):
    """
    Calculate summary statistics (count, mean, median, std, min, max)
    for q1, q2, and q3 times.
    """
    stats = {}
    for col in ["q1", "q2", "q3"]:
        col_data = df[col].dropna()
        stats[col] = {
            "count": len(col_data),
            "mean": col_data.mean(),
            "median": col_data.median(),
            "std": col_data.std(),
            "min": col_data.min(),
            "max": col_data.max()
        }
    return stats

if __name__ == "__main__":
    # List of file paths for training sets (adjust these paths as needed)
    training_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
    ]

    # Load and combine training data
    df_training = load_training_data(training_files)
    if df_training.empty:
        print("No qualifying data found. Check file paths and data format.")
    else:
        # Calculate summary statistics for qualifying times
        stats = calculate_summary_statistics(df_training)
        print("Summary Statistics for Qualifying Sessions:")
        for session, session_stats in stats.items():
            print(f"\n{session.upper()}:")
            for stat_name, value in session_stats.items():
                print(f"  {stat_name}: {value:.3f}" if isinstance(value, float) else f"  {stat_name}: {value}")
