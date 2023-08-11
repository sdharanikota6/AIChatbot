# Import libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize WordNetLemmatizer for word lemmatization
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Lists to store words, classes, and documents
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Preprocess the intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern into a list of words
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Store the document as a tuple of word list and intent tag
        documents.append((wordList, intent['tag']))
        # Add the intent tag to classes if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Normalize words by lemmatization and remove duplicates
words = {lemmatizer.lemmatize(word.lower())
         for word in words if word not in ignoreLetters}

classes = sorted(set(classes))

# Save words and classes to pickle files for later use
pickle.dump(list(words), open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data in a format suitable for training a neural network
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(
        word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle the training data to ensure randomization
random.shuffle(training)
training = np.array(training)

# Split training data into input features (trainX) and output labels (trainY)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Create a sequential neural network model using Keras
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Configure the optimizer and compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# Train the model on the prepared data
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the trained model for later use
model.save('sudeep_model.keras')

# Training complete
print('Done Training')
