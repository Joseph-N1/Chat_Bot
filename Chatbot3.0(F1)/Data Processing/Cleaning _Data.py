import os
import time
import pandas as pd
import ast
from openpyxl import Workbook

# --- Define paths ---
INPUT_FILE = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot3.0(F1)\F1_Seasons_2020.xlsx"
OUTPUT_DIR = r"C:\Users\Joseph N Nimyel\OneDrive\Documents\Chatbot\Chatbot3.0(F1)\Cleaned Data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "F1_Seasons_Cleaned_(2020).xlsx")
OUTPUT_ODS = os.path.join(OUTPUT_DIR, "F1_Seasons_Cleaned_(2020).ods")

# --- Parsing helper functions for Qualifying ---
def parse_driver_name(x):
    if not x:
        return ""
    if isinstance(x, dict):
        return x.get("givenName", "") + " " + x.get("familyName", "")
    try:
        obj = ast.literal_eval(str(x))
        return obj.get("givenName", "") + " " + obj.get("familyName", "")
    except:
        return str(x)

def parse_constructor_name(x):
    if not x:
        return ""
    if isinstance(x, dict):
        return x.get("name", "")
    try:
        obj = ast.literal_eval(str(x))
        return obj.get("name", "")
    except:
        return str(x)

# --- Parsing helper functions for Standings ---
def parse_driver_standings(x, race_year):
    if not x:
        return ""
    try:
        obj = ast.literal_eval(str(x))
        first = obj.get("givenName", "")
        last = obj.get("familyName", "")
        dob = obj.get("dateOfBirth", "")
        nat = obj.get("nationality", "")
        
        age_str = ""
        if dob:
            if race_year is not None:
                try:
                    birth_year = pd.to_datetime(dob).year
                    age = race_year - birth_year
                    age_str = str(age)
                except:
                    age_str = dob  # Fallback to dob if parsing fails
            else:
                age_str = dob  # Use dob if race_year is unavailable
        return f"{first} {last} / {nat} / {age_str}"
    except:
        return str(x)

def parse_constructor_standings(x):
    if not x:
        return ""
    try:
        arr = ast.literal_eval(str(x))
        if arr and isinstance(arr, list):
            c = arr[0]
            cname = c.get("name", "")
            cnat = c.get("nationality", "")
            return f"{cname} / {cnat}"
        return ""
    except:
        return str(x)

# --- Function to clean a single sheet ---
def clean_sheet(df):
    """
    Processes one raw sheet (DataFrame) and returns a dictionary with 4 DataFrames:
      - meta: Race metadata (e.g. Race name and Date)
      - quali: Qualifying table (with parsed Driver and Constructor)
      - race: Race Results table (as-is)
      - standings: Championship Standings table (cleaned and renamed)
    The function locates each section by scanning for header keywords.
    """
    quali_start_idx = None
    race_start_idx = None
    standings_start_idx = None

    for i, row in df.iterrows():
        row_vals = [str(x).lower() for x in row.tolist()]
        if ("number" in row_vals and "position" in row_vals and 
            "driver" in row_vals and "constructor" in row_vals and "q1" in row_vals):
            quali_start_idx = i
        if ("driver" in row_vals and "team" in row_vals and 
            "starting position" in row_vals and "finish position" in row_vals):
            race_start_idx = i
        if ("position" in row_vals and "pts" in row_vals and "wins" in row_vals and "driver" in row_vals):
            standings_start_idx = i

    total_rows = len(df)
    quali_end = race_start_idx if race_start_idx is not None else total_rows
    race_end = standings_start_idx if standings_start_idx is not None else total_rows
    standings_end = total_rows

    def extract_table(start, end):
        if start is None or start >= end:
            return pd.DataFrame()
        header = df.iloc[start].tolist()
        data = df.iloc[start+1:end].copy()
        data.columns = header
        data = data.dropna(how='all')
        return data

    quali_df = extract_table(quali_start_idx, quali_end)
    race_df = extract_table(race_start_idx, race_end)
    standings_df = extract_table(standings_start_idx, standings_end)

    # Clean Qualifying table
    if not quali_df.empty:
        if "Driver" in quali_df.columns:
            quali_df["Driver"] = quali_df["Driver"].apply(parse_driver_name)
        if "Constructor" in quali_df.columns:
            quali_df["Constructor"] = quali_df["Constructor"].apply(parse_constructor_name)

    # Clean Standings table
    if not standings_df.empty:
        # Parse race year from meta data
        race_year = None
        try:
            race_date = pd.to_datetime(meta["Date"], errors='coerce')
            if not pd.isnull(race_date):
                race_year = race_date.year
        except:
            pass
        
        # Drop unwanted columns
        drop_cols = [col for col in standings_df.columns if "positionText" in col]
        if drop_cols:
            standings_df = standings_df.drop(columns=drop_cols)
        
        # Parse driver and constructor
        if "Driver" in standings_df.columns:
            standings_df["Driver"] = standings_df["Driver"].apply(
                lambda x: parse_driver_standings(x, race_year))
        if "Constructors" in standings_df.columns:
            standings_df["Constructors"] = standings_df["Constructors"].apply(parse_constructor_standings)
        
        # Rename columns
        rename_map = {
            "position": "Pos",
            "points": "Pts",
            "wins": "Wins",
            "Driver": "Driver (Name/Nationality/Age)",
            "Constructors": "Team (Name/Nationality)"
        }
        rename_map = {k: v for k, v in rename_map.items() if k in standings_df.columns}
        standings_df.rename(columns=rename_map, inplace=True)

    # Extract metadata from the second row (index 1)
    meta = {}
    try:
        meta["Race"] = df.iloc[1,0]
        meta["Date"] = df.iloc[1,1] if df.shape[1] > 1 else ""
    except Exception as e:
        meta["Race"] = ""
        meta["Date"] = ""
    meta_df = pd.DataFrame([meta])
    
    return {
        "meta": meta_df,
        "quali": quali_df,
        "race": race_df,
        "standings": standings_df
    }

# --- Process all sheets from the input file ---
def clean_all_sheets(input_file):
    sheets_dict = pd.read_excel(input_file, sheet_name=None)
    cleaned = {}
    for sheet_name, df in sheets_dict.items():
        print(f"Cleaning sheet: {sheet_name}")
        cleaned[sheet_name] = clean_sheet(df)
    return cleaned

# --- Write cleaned sheets to XLSX and ODS ---
def save_cleaned_sheets(cleaned_sheets, output_xlsx, output_ods):
    # Write XLSX with blank rows between sections
    with pd.ExcelWriter(output_xlsx, engine='openpyxl', mode='w') as writer:
        for sheet_name, blocks in cleaned_sheets.items():
            new_sheet = f"Cleaned_{sheet_name}"[:31]
            current_row = 0
            
            # Meta
            blocks["meta"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
            current_row += len(blocks["meta"]) + 1  # +1 for blank row after
            
            # Qualifying
            if not blocks["quali"].empty:
                blocks["quali"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                current_row += len(blocks["quali"]) + 1
            
            # Race
            if not blocks["race"].empty:
                blocks["race"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                current_row += len(blocks["race"]) + 1
            
            # Standings
            if not blocks["standings"].empty:
                blocks["standings"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                current_row += len(blocks["standings"]) + 1
    
    print(f"Cleaned XLSX saved to {output_xlsx}")
    
    # Write ODS similarly
    try:
        with pd.ExcelWriter(output_ods, engine='odf', mode='w') as writer:
            for sheet_name, blocks in cleaned_sheets.items():
                new_sheet = f"Cleaned_{sheet_name}"[:31]
                current_row = 0
                
                blocks["meta"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                current_row += len(blocks["meta"]) + 1
                
                if not blocks["quali"].empty:
                    blocks["quali"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                    current_row += len(blocks["quali"]) + 1
                
                if not blocks["race"].empty:
                    blocks["race"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                    current_row += len(blocks["race"]) + 1
                
                if not blocks["standings"].empty:
                    blocks["standings"].to_excel(writer, sheet_name=new_sheet, startrow=current_row, index=False)
                    current_row += len(blocks["standings"]) + 1
        
        print(f"Cleaned ODS saved to {output_ods}")
    except Exception as e:
        print(f"Error saving ODS file: {e}")

# --- Main ---
if __name__ == "__main__":
    print("Starting cleaning process...")
    cleaned_sheets = clean_all_sheets(INPUT_FILE)
    save_cleaned_sheets(cleaned_sheets, OUTPUT_XLSX, OUTPUT_ODS)