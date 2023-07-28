# Sthapatya AI Chatbot

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.6.2-brightgreen)

Sthapatya AI Chatbot is a simple conversational bot developed to provide accurate and relevant information about the company Sthapatya Consultancy Services. The chatbot is trained to respond to user queries related to greetings, company information, contact details, office locations, services provided, and more.

## Features

- Responds to common greetings like "Hi," "Hello," and "Hey."
- Provides information about the company "Sthapatya Consultancy Services."
- Offers contact details and ways to get in touch with the company.
- Shares office locations and addresses for both Pune and Amaravati.
- Describes the various services provided by the company.
- Handles unrecognized queries with appropriate responses.
- Offers a prompt (">") for users to ask the next question after the bot's response.

## Prerequisites

- Python 3.7 or higher is required to run the chatbot.
- TensorFlow 2.0 or higher for training and using the deep learning model.
- NLTK (Natural Language Toolkit) 3.6.2 for text processing and tokenization.

## Installation

1. Clone this repository to your local machine.
```
git clone https://github.com/your_username/sthapatya-ai-chatbot.git
cd sthapatya-ai-chatbot
```
2. Install the required Python libraries.
```
pip install tensorflow nltk
```
3. Run the training script to create the chatbot model.
```
python training.py
```
4. Start the Chatbot
```
python chatbot.py
```

## Usage

1. When the chatbot is running, enter your queries or questions in the console.
2. The chatbot will process your input and provide appropriate responses.
3. The chatbot's responses will be displayed with a slight delay to create a more natural conversation.
4. After each response, a prompt (">") will be shown, indicating that you can ask the next question.

## Training Data

The chatbot has been trained using a JSON file `intents.json`, which contains different intent patterns and corresponding responses. The file is organized as follows:

```json
{
"intents": [
 {
   "tag": "greetings",
   "patterns": ["Hi", "Hello", "Hey"],
   "responses": ["Hello!", "Hi there!", "Hey!"]
 },
 {
   "tag": "sthapatya",
   "patterns": ["Sthapatya", "What is Sthapatya", "What Sthapatya"],
   "responses": ["Sthapatya is a Consultancy Service Provider.", "..."]
 },
 ...
]
}
```
You can modify the intents.json file to customize the chatbot's responses and add more intents as needed.

## Credits

This chatbot project was developed by Moksh. It is based on a simple deep learning model using TensorFlow and NLTK for natural language processing.

## License
This project is licensed under the MIT License.
