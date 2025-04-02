import fastf1

# Enable caching for faster repeated calls
fastf1.Cache.enable_cache('cache') 

year = 2024
gp = 'Bahrain'
session_type = 'R'  # Race, can be 'Q', 'FP1', etc.

session = fastf1.get_session(year, gp, session_type)
session.load()  # Ensure session data is loaded

# Verify session data is available
print(session.results)  # Should not be empty
print(session.laps.head())  # Check lap data

