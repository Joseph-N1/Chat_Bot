# Quick fix for path issues - add this to the top of your predicted_podium.py file

import os
import glob

def find_f1_data_file():
    """Find F1 2024 data file in various locations"""
    
    # Common file patterns to look for
    patterns = [
        "F1_Seasons_Cleaned_2024.json",
        "F1_2024.json", 
        "f1_seasons_2024.json",
        "2024_season.json",
        "*2024*.json"
    ]
    
    # Common directory locations to search
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    search_dirs = [
        current_dir,
        script_dir,
        parent_dir,
        os.path.join(parent_dir, "DataBase"),
        os.path.join(parent_dir, "Database"),
        os.path.join(parent_dir, "data"),
        os.path.join(parent_dir, "Data"),
        os.path.join(current_dir, "DataBase"),
    ]
    
    print("🔍 Searching for F1 2024 data file...")
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for pattern in patterns:
                files = glob.glob(os.path.join(search_dir, pattern))
                for file in files:
                    if os.path.exists(file):
                        print(f"✅ Found F1 data file: {file}")
                        return file
    
    return None

def setup_dynamic_db_dir():
    """Dynamically set DB_DIR based on where F1 files are found"""
    
    # Try to find the F1 data file
    found_file = find_f1_data_file()
    
    if found_file:
        # Set DB_DIR to the directory containing the found file
        return os.path.dirname(found_file)
    else:
        # Fall back to original logic
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(script_dir), "DataBase")

# Replace the DB_DIR line in your script with this:
# DB_DIR = setup_dynamic_db_dir()

# Test the function
if __name__ == "__main__":
    db_dir = setup_dynamic_db_dir()
    print(f"DB_DIR would be set to: {db_dir}")
    
    # Check if the target file exists
    target_file = os.path.join(db_dir, "F1_Seasons_Cleaned_2024.json")
    if os.path.exists(target_file):
        print(f"✅ Target file found: {target_file}")
    else:
        print(f"❌ Target file not found: {target_file}")
        print("\nAvailable files in that directory:")
        if os.path.exists(db_dir):
            for file in os.listdir(db_dir):
                if file.endswith('.json'):
                    print(f"   📄 {file}")
        else:
            print(f"   ❌ Directory doesn't exist: {db_dir}")