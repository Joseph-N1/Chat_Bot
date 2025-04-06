import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Increase default figure size for better readability
plt.rcParams['figure.figsize'] = (10, 6)

def time_to_seconds(time_str):
    """
    Convert time string in format M:SS.mmm to total seconds (float).
    Returns np.nan if the string is empty or invalid.
    """
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
    Returns a list of dicts with qualifying session data.
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
                for rec in q_results:
                    rec["race_name"] = race_name
                all_results.extend(q_results)
    return pd.DataFrame(all_results)

def visualize_distribution(df):
    """
    Generate and save distribution plots (histogram and boxplot) for q1, q2, and q3.
    """
    sessions = ["q1", "q2", "q3"]
    for session in sessions:
        # Drop NaNs
        session_data = df[session].dropna()
        
        # Create a histogram
        plt.figure()
        sns.histplot(session_data, kde=True, bins=30)
        plt.title(f"Distribution of {session.upper()} Times")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(f"{session}_distribution_histogram.png")
        plt.close()
        
        # Create a box plot
        plt.figure()
        sns.boxplot(x=session_data)
        plt.title(f"Boxplot of {session.upper()} Times")
        plt.xlabel("Time (seconds)")
        plt.savefig(f"{session}_boxplot.png")
        plt.close()
        print(f"Plots for {session.upper()} saved as {session}_distribution_histogram.png and {session}_boxplot.png.")

if __name__ == "__main__":
    # List your training set file paths here
    training_files = [
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2020.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2021.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2022.json",
        r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot4.0(F1 - ML)\Training-Set\F1_Seasons_Cleaned_2023.json"
    ]
    
    df_training = load_training_data(training_files)
    if df_training.empty:
        print("No qualifying data found. Check file paths and data format.")
    else:
        visualize_distribution(df_training)
