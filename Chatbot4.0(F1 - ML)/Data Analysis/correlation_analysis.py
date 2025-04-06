import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 8)

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
    Extract qualifying results records from race data.
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
    Extracts the season from the file name (expects a 4-digit year).
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

def perform_correlation_analysis(df):
    """
    Calculate the correlation matrix for key fields and visualize using a heatmap.
    The key fields include:
        - 'position'
        - 'q1'
        - 'q2'
        - 'q3'
    """
    # Select only the relevant numerical columns
    numeric_df = df[['position', 'q1', 'q2', 'q3']].dropna()
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Plot heatmap
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Qualifying Times and Position")
    plt.savefig("correlation_heatmap.png")
    plt.close()
    print("Correlation heatmap saved as 'correlation_heatmap.png'.")

if __name__ == "__main__":
    # List of training file paths (update as needed)
    training_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
    ]
    
    # Load combined data
    df_all = load_all_seasons(training_files)
    if df_all.empty:
        print("No data loaded. Check file paths and data format.")
    else:
        print("Combined data preview:")
        print(df_all.head())
        # Perform correlation analysis and visualize
        perform_correlation_analysis(df_all)
