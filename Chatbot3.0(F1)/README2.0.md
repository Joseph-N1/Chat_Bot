####

# F1 Data Project

This project collects and processes Formula 1 race data for multiple seasons (2020, 2021, 2022, 2023, 2024) with the goal of cleaning the data, creating graphs in Excel, and converting the results to JSON.

## API Comparison

The table below outlines key differences between the two APIs used in this project:

| **Feature**     | **FastF1**                                                      | **Ergast**                                        |
| --------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| **Data Source** | Official Formula 1 timing and telemetry API                     | Ergast Developer API                              |
| **Main Issue**  | May load the wrong session (e.g., São Paulo instead of Bahrain) | Generally reliable with no major issues           |
| **Data Scope**  | Detailed lap-by-lap and telemetry data                          | Race results, standings, and metadata             |
| **Accuracy**    | Some laps may be missing or marked as inaccurate                | Clean, structured, and consistent historical data |

## Recommendation

For this project, I recommend primarily using the **Ergast API** because:

- **Consistency Across Seasons:** Ergast provides historical race results and metadata for multiple seasons (2020-2024) reliably.
- **Clean and Structured Data:** The data is well-organized, making it easier to clean, visualize in Excel, and convert to JSON.
- **Simpler Data Handling:** Ergast is less prone to session-loading issues than FastF1, which simplifies data processing.

**However**, if detailed lap-by-lap or telemetry data is needed for advanced analysis, you may consider combining FastF1 with Ergast:

- Use Ergast as the primary source for overall race results, driver standings, and pitstop data.
- Supplement with FastF1 for detailed lap data, telemetry, and weather information when available.

## Combining Both APIs

To achieve a comprehensive dataset:

1. **Primary Data Collection:**

   - Retrieve race results, standings, and metadata from the Ergast API.
   - This data is reliable and consistent across all seasons.

2. **Supplementary Data:**

   - Use FastF1 to fetch detailed lap times, telemetry, and weather data for selected races that require in-depth analysis.

3. **Data Merging:**

   - **Match on Key Identifiers:** Merge the datasets using common identifiers such as race name, date, and driver ID.
   - **Handle Missing Data:** Prioritize Ergast data for overall results, while incorporating FastF1 data for detailed laps.
   - **Unified Dataset:** Combine both datasets into a master DataFrame that can be cleaned, visualized, and converted to JSON.

4. **Graphing and Visualization:**
   - After merging and cleaning, create graphs in Excel or use Python libraries (e.g., matplotlib or seaborn) to visualize:
     - Fastest lap times versus finishing positions.
     - Number of laps completed and pitstop counts.
     - Weather impacts on race performance.
