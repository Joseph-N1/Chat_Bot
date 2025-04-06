### About summary_stats.py

The `summary_stats.py` script is a key component of the F1 project, designed to process and analyze qualifying session data from multiple races. It begins by converting time strings (e.g., "1:30.499") into numerical seconds using the `time_to_seconds` function, ensuring consistent and comparable data. The script then extracts relevant qualifying results from race data files with the `extract_qualifying_results` function, isolating the necessary information for further analysis.

Once the data is extracted, the `load_training_data` function consolidates all qualifying records from multiple JSON files into a single Pandas DataFrame. The `calculate_summary_statistics` function then computes detailed statistics, including count, mean, median, standard deviation, minimum, and maximum values for each qualifying session (Q1, Q2, and Q3). Finally, the script executes these steps in its main block, where file paths are specified, and the computed statistics are displayed for review.

#### visualize_distribution.py

The script loads your F1 qualifying session data (from the cleaned JSON training sets: 2020–2023), extracts the q1, q2, and q3 times, and generates visualizations to help you understand how the qualifying times are distributed across all races.

It gives you:

Histograms with KDE curves (for visualizing how times are spread out)

Boxplots (for spotting outliers and seeing the data’s range)

This script helps you answer:

Are Q1/Q2/Q3 times normally distributed?

Do any years or sessions have unusual outliers?

How much variation exists between Q1 and Q3?

This is critical before feeding the data into ML models — so you understand what you're working with and whether to normalize or clean further.

#### compare_seasonal_trends.py

How This Script Works
Time Conversion & Data Extraction:
The script reuses the time_to_seconds and extract_qualifying_results functions to convert qualifying time strings into seconds and extract the relevant fields from each race's JSON data.

Season Tagging:
In load_data_with_season(), the script extracts a 4-digit year (season) from the file name. Each record in the DataFrame is tagged with its season (e.g., "2020", "2021", etc.).

Combining Data:
The load_all_seasons() function reads multiple season files and concatenates the results into a single DataFrame.

Visualization:
The plot_seasonal_boxplots() function generates box plots for each qualifying session (q1, q2, q3) comparing the distribution of times across the different seasons. The plots are saved as PNG files.

Execution:
In the main block, the script loads the specified JSON files, prints a preview of the combined data, and creates the seasonal box plots.

#### correlation_analysis.py

How the Script Works
Data Loading & Extraction:

The script reuses functions to convert qualifying time strings into seconds and to extract qualifying session records from each JSON file.

It tags each record with the corresponding season (extracted from the file name) and combines data from all specified seasons into one DataFrame.

Correlation Calculation:

It selects the key numerical columns (position, q1, q2, and q3), removes missing values, and calculates a correlation matrix using Pandas' .corr() method.

The matrix is printed to the console.

Visualization:

The script uses Seaborn to generate a heatmap of the correlation matrix, complete with annotations.

The heatmap is saved as a PNG file named correlation_heatmap.png.

#### train_and_evaluate_model.py

Loads your training data (from 2020–2023) and test data (2024).

Prepares features and the target variable (for example, predicting finishing positions).

Trains a baseline model (like a Random Forest Regressor) using the qualifying times as inputs.

Evaluates the model on the 2024 test set and outputs performance metrics.

How This Script Works
Data Loading:

It reuses our functions to load JSON files from your specified training and test sets.

Each record is tagged with a season (extracted from the file name).

Feature Preparation:

The script uses qualifying times (q1, q2, and q3) as features.

In this example, we predict the grid_position as the target variable.
(If you have finishing positions available, you can adjust the target accordingly.)

Model Training:

A RandomForestRegressor is initialized and trained using the training data.

This model is a strong baseline for regression tasks.

Evaluation:

Predictions are made on the test data (2024).

The model's performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

These metrics help you understand how well the model is performing on unseen data.

Output:

The script prints out the shapes of the training and test sets as well as the performance metrics.
