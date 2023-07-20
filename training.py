import random, json, pickle, nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = set()  # Initialize classes as a set to avoid duplicates
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if not isinstance(intent['tag'], str):
            # If the tag is not a string, convert it to a string before adding to the set
            classes.add(str(intent['tag']))
        else:
            classes.add(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Convert classes back to a sorted list
classes = sorted(classes)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([np.array(bag, dtype=np.int32), np.array(output_row, dtype=np.int32)])  # Convert bag and output_row to numpy arrays with int32 data type

random.shuffle(training)
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Separate the input and output sequences from the training data
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Pad the sequences to have the same length
train_x = pad_sequences(train_x, padding='post')
train_y = pad_sequences(train_y, padding='post')

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')