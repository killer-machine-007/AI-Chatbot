import random, json, pickle, time, numpy as np, nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

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
            if i['tag'] == 'services':
                # Concatenate all responses for 'services' intent
                result = '\n'.join(i['responses'])
            elif 'pune' in message and i['tag'] == 'locations':
                # Handle query for Pune office address
                result = [response for response in i['responses'] if 'pune' in response.lower()]
                result = '\n'.join(result)
            elif 'amaravati' in message and i['tag'] == 'locations':
                # Handle query for Amaravati office address
                result = [response for response in i['responses'] if 'amaravati' in response.lower()]
                result = '\n'.join(result)
            else:
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
        print(res)
