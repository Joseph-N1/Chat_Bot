import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 8)

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
    Given race data (list of dicts), extract qualifying session records.
    Returns a list of dictionaries containing driver, constructor, position,
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
                "position": int(item.get("position", 0)),
                "q1": time_to_seconds(item.get("q1", "")),
                "q2": time_to_seconds(item.get("q2", "")),
                "q3": time_to_seconds(item.get("q3", ""))
            }
            results.append(record)
    return results

def load_data_with_season(file_path):
    """
    Load a JSON file and extract qualifying results.
    Infers the season from the file name (expects a 4-digit year).
    """
    season = "Unknown"
    # Try to extract the 4-digit year from the file name
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

def load_all_seasons(file_paths):
    """Load data from multiple season files and combine into one DataFrame."""
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

def plot_seasonal_boxplots(df):
    """Create box plots for q1, q2, and q3 by season."""
    sessions = ["q1", "q2", "q3"]
    for session in sessions:
        plt.figure()
        sns.boxplot(x="season", y=session, data=df, order=sorted(df["season"].unique()))
        plt.title(f"{session.upper()} Times by Season")
        plt.xlabel("Season")
        plt.ylabel("Time (seconds)")
        plt.savefig(f"{session}_seasonal_boxplot.png")
        plt.close()
        print(f"Saved {session.upper()} seasonal boxplot as {session}_seasonal_boxplot.png")

if __name__ == "__main__":
    # List of training file paths (update paths as needed)
    training_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
    ]
    
    df_all = load_all_seasons(training_files)
    if df_all.empty:
        print("No data loaded. Check file paths and data format.")
    else:
        # Display first few rows for verification
        print("Combined data preview:")
        print(df_all.head())
        # Create and save seasonal box plots for each qualifying session
        plot_seasonal_boxplots(df_all)
