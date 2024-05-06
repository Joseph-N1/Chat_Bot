import json
import os
import re
import spacy
from spacy.matcher import PhraseMatcher
import logging
import unidecode

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

def setup_intent_patterns():
    intent_patterns = {
        "winner_query": ["who won the", "race winner in", "who was the winner in"],
        "race_count_query": ["how many races were there in", "number of races in"],
        "driver_statistics": ["how many races has", "won?", "races won by", "number of races won by"],
        "team_statistics": ["races won by", "how many races did", "win in"]
    }
    for intent, patterns in intent_patterns.items():
        pattern_docs = [nlp.make_doc(text) for text in patterns]
        matcher.add(intent, pattern_docs)

def load_f1_data(file_path="C:\\Users\\Joseph N Nimyel\\OneDrive\\Documents\\Chatbot\\Chatbot2.0(F1)\\CHAT\\F1_stats.json"):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            logging.info(f"Loaded data from {os.path.basename(file_path)}")
            return data
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return {}

def get_f1_response(doc, F1_stats):
    response = "Bot: I'm sorry, I couldn't find any data for that query."
    matches = matcher(doc)
    year_match = re.search(r"\b(20\d{2})\b", doc.text)
    year = year_match.group(1) if year_match else None
    
    if not year:
        return "Bot: Please specify the year for your query."
    
    print(F1_stats['F1_data'].get(year, {}).get('drivers', {}))  # Debugging line

    for match_id, start, end in matches:
        intent = nlp.vocab.strings[match_id]
        if intent == "driver_statistics" or intent == "winner_query":
            race_name_search = re.search(r"who won the (.+ Grand Prix)", doc.text, re.IGNORECASE)
            race_name = race_name_search.group(1) if race_name_search else None
            race_name_normalized = unidecode.unidecode(race_name).lower() if race_name else None

            if race_name:
                found = False
                for driver, details in F1_stats['F1_data'][year]['drivers'].items():
                    for win in details['wins']:
                        race_normalized = unidecode.unidecode(win['race']).lower()
                        if race_normalized == race_name_normalized + " grand prix":
                            response = f"Bot: {driver.replace('_', ' ').title()} won the {race_name} Grand Prix in {year}."
                            found = True
                            break
                    if found:
                        break
                if not found:
                    response += " No data found for this race."
    return response

def chat_bot():
    knowledge_base = load_f1_data()
    print("Bot: Hello! Ask me anything about F1 stats, or type 'quit' to exit.")
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        doc = nlp(user_input)
        print(get_f1_response(doc, knowledge_base))

if __name__ == '__main__':
    setup_intent_patterns()
    chat_bot()
