from tensorflow.keras.layers import Activation, Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
import random
import numpy as np
import tensorflow as tf


# =========================
# LOAD DATASET (SHAKESPEARE TEXT)
# =========================

# Download Shakespeare dataset
filepath = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Read text file and convert to lowercase
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Keep only a subset of the text for training (reduces computation time)
text = text[300000:800000]

# Get unique characters in the dataset
characters = sorted(set(text))

# Create mappings between characters and indices
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))


# =========================
# DATA PREPARATION
# =========================

SEQ_LENGTH = 40   # length of input sequence
STEP_SIZE = 3     # step size for sliding window

sentences = []    # input sequences
next_char = []    # target characters

# Create training sequences using sliding window
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

# One-hot encoded input data
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)

# One-hot encoded output labels
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

# Convert characters into one-hot vectors
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1


# =========================
# BUILD LSTM MODEL
# =========================

model = Sequential()

# LSTM layer learns sequential patterns in character sequences
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))

# Fully connected layer to map LSTM output to character probabilities
model.add(Dense(len(characters)))

# Softmax activation to output probability distribution over characters
model.add(Activation('softmax'))

# Compile model with categorical crossentropy loss (multi-class classification)
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.01)
)


# =========================
# TRAIN MODEL
# =========================

model.fit(x, y, batch_size=256, epochs=4)


# =========================
# SAMPLING FUNCTION
# =========================

def sample(preds, temperature=1.0):
    """
    Sample next character index from probability distribution.
    Temperature controls randomness:
    - Low temperature → more deterministic text
    - High temperature → more creative/random text
    """
    preds = np.asarray(preds).astype('float64')

    # Apply temperature scaling
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)

    # Normalize probabilities
    preds = exp_preds / np.sum(exp_preds)

    # Sample one index from probability distribution
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


# =========================
# TEXT GENERATION FUNCTION
# =========================

def generate_text(length, temperature):
    """
    Generate text using trained LSTM model.

    Args:
        length (int): number of characters to generate
        temperature (float): randomness level of generation
    """

    # Pick a random starting point in the text
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)

    # Initial seed sequence
    sentence = text[start_index: start_index + SEQ_LENGTH]

    generated = sentence

    # Generate characters one by one
    for i in range(length):

        # Prepare input tensor for prediction
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))

        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        # Predict next character probabilities
        predictions = model.predict(x_predictions, verbose=0)[0]

        # Sample next character index using temperature-based sampling
        next_index = sample(predictions, temperature)

        # Convert index to actual character
        next_character = index_to_char[next_index]

        # Append generated character
        generated += next_character

        # Slide window forward (keep last SEQ_LENGTH characters)
        sentence = sentence[1:] + next_character

    return generated


# =========================
# GENERATE TEXT (DIFFERENT TEMPERATURES)
# =========================

print(generate_text(300, 0.2))  # more deterministic / repetitive
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))  # more creative / chaotic
