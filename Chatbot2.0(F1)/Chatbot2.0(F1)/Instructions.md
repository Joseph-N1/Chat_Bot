
## Chatbot2.0(F1)

### Overview
Chatbot2.0(F1) is an interactive chatbot designed to provide information about Formula 1 races, drivers, and statistics from the years 2020 to 2024. This document outlines the process of setting up and running the chatbot, including the steps to fetch and process data, manage a knowledge base, and handle user interactions.

### System Requirements
- Python 3.x
- Spacy Natural Language Processing Library
- JSON processing capabilities

### Setup Instructions

#### 1. Getting F1 Data Using APIs
To retrieve data about F1 races, you must first get access to relevant APIs that provide historical data for the years 2020 to 2024. This project assumes the data is available in JSON format and is saved locally.

#### 2. Converting Data to JSON Files
Use the `data_transformation.py` script to process raw JSON data files, restructuring them into a more manageable format:
- Directories of JSON files are processed to collate information about drivers, races, and tracks.
- The script aggregates data into a centralized JSON file, `f1_knowledge_base.json`.

#### 3. Creating and Updating Knowledge Base
- **Knowledge Base Creation**: Data from JSON files is converted into a structured format for the chatbot to use, which includes details about each race, the drivers, and their statistics.
- **Dynamic Learning**: `main.py` and `f1_stats.py` handle the creation and updating of the knowledge base, allowing the chatbot to learn from past queries and include new data as it becomes available.

#### 4. Python Libraries for Handling Queries
- **Spacy**: Used for natural language processing to understand and classify user queries.
- **JSON**: For loading and saving data structures.
- **re** and **unidecode**: For regular expressions and unicode data handling, respectively.

### Features
- **Interactive Chat Interface**: Users can ask questions about F1 races, and the chatbot, using its knowledge base, answers those queries.
- **Handling Various Intents**: The chatbot can determine user intent (like querying for a race winner or driver statistics) and fetch relevant data.
- **Scalability**: The design allows easy updates to the database to include additional years or more detailed data without significant restructuring.

### Running the Chatbot
Run `main.py` to start the chatbot. It loads the knowledge base and waits for user inputs. The chatbot can answer queries related to F1 data and maintains a session until the user decides to quit.

### Troubleshooting Common Issues
- **Incorrect Intent Handling**: Ensure the intent patterns in `setup_intent_patterns()` are comprehensive and match typical user queries.
- **Data Parsing Issue**: Check the data structure in the JSON files if there are loading or parsing errors.
- **Error in Driver Matching**: Verify that the regex and data normalization in `get_f1_response()` correctly identify and match drivers and races.
- **Debugging Output**: Use logging statements provided in the scripts to trace values and understand the flow of data.

### Future Development
- **Front-end Development**: Plans to integrate the backend with a frontend using HTML, CSS, and JavaScript to enhance user interaction.
- **Additional Data Sources**: Incorporate more detailed data sources and expand the temporal coverage beyond 2024.

### Conclusion
Chatbot2.0(F1) offers a dynamic and interactive way to explore Formula 1 data. With continuous updates and improvements, it aims to provide a comprehensive tool for F1 enthusiasts to query historical data efficiently.