import random, json, pickle, numpy as np, nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from tensorflow.keras.models import load_model
import time

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')  # Use 'chatbot_model.h5' instead of 'chatbot_model.model'

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def get_greeting():
    greetings = ["Hello!", "Hi there!", "Greetings!", "Hey, how can I help you?", "Welcome! How may I assist you?"]
    return random.choice(greetings)

def handle_unrecognized():
    responses = ["I'm sorry, I didn't quite understand. Can you please rephrase your question?",
                 "Apologies, I'm having trouble understanding that. Could you try again?",
                 "I'm not sure what you meant. Can you please provide more context?"]
    return random.choice(responses)

print("Bot is Running")

while True:
    message = input("").lower()
    
    if message == "exit":
        print("Goodbye! Have a great day!")
        break

    if message == "":
        continue

    ints = predict_class(message)

    # Sort intents by probability and select the one with the highest confidence
    ints.sort(key=lambda x: float(x['probability']), reverse=True)
    intent = ints[0]
    if float(intent['probability']) > 0.5:
        res = get_response(ints, intents)
    else:
        res = handle_unrecognized()

    if intent['intent'] == 'greetings':
        print(get_greeting())
    else:
        for char in res:
            print(char, end='', flush=True)  # Print each character with flush=True to ensure immediate display
            time.sleep(0.03)  # Add a delay of 0.03 seconds for each character
        print()  # Print a new line after the response is complete

    print("> ", end="", flush=True)  # Print the prompt for the user to ask the next question
