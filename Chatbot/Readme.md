# Self-Learning Chatbot

## Overview
This project implements a self-learning chatbot using Python. It is designed to improve its knowledge over time by interacting with users and storing new information in a JSON-based knowledge base.

## Features

### Python Functions Employed
- **JSON Processing**: The chatbot uses the `json` library to load and save its knowledge base, allowing it to remember past interactions.
- **Text Matching**: Utilizing the `difflib.get_close_matches` function, the bot can find the closest match to user queries from the stored questions, enhancing its ability to understand and respond accurately.
- **Persistent Storage**: Data is saved in a `knowledge_base.json` file, which means the chatbot's knowledge persists between sessions.

### Functions
- **load_knowledge_base(file_path: str) -> dict**: Loads the knowledge base from the specified JSON file.
- **save_knowledge_base(file_path: str, data: dict)**: Saves the current state of the knowledge base to a JSON file.
- **find_best_match(user_question: str, questions: list[str]) -> str | None**: Finds the closest match for a user's question within the knowledge base.
- **get_answer_for_question(question: str, knowledge_base: dict) -> str | None**: Retrieves the answer for a given question from the knowledge base.
- **chat_bot()**: Main function to run the chatbot which interacts with the user, learns from new inputs, and updates the knowledge base.

## Future Plans
### Integration with Business Websites and Agencies
The chatbot aims to support small businesses, travel agencies, and other client services by providing a cost-effective way to handle customer inquiries. Future integrations will include:
- **Customizable Modules**: Tailoring responses based on specific business needs or domains.
- **Scalable Learning**: As the bot interacts with more users, it will gather more data and provide more accurate and relevant responses.

### Open-Source Collaboration
Plans to make this chatbot open-source are underway, which will:
- **Encourage Community Contributions**: Allowing developers to contribute and extend the bot's capabilities.
- **Diverse Knowledge Base**: Enabling the bot to learn from various fields and become more versatile in its responses.
- **Continuous Improvement**: With community input, the bot will continuously improve, adapting to new information and retaining all it has learned.

## Conclusion
This self-learning chatbot represents a significant step forward in providing automated, intelligent customer service solutions for businesses and individual developers. By leveraging Python's capabilities and open-source collaboration, it aims to become more robust and helpful over time.

