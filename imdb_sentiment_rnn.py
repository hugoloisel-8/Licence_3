import numpy as np
import re

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence


# =========================
# LOAD IMDB DATASET
# =========================

# Only keep the top 10,000 most frequent words
max_features = 10000

# Maximum length of each review (truncate longer ones)
maxlen = 500

# Load dataset (already split into train/test)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


# =========================
# PADDING SEQUENCES
# =========================

# Ensure all reviews have the same length (required for RNN input)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


# =========================
# BUILD RNN MODEL
# =========================

model = Sequential()

# Embedding layer converts word indices into dense vectors
model.add(Embedding(max_features, 32))

# Simple RNN layer processes sequences sequentially
model.add(SimpleRNN(32))

# Output layer for binary sentiment classification (positive/negative)
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)


# =========================
# EVALUATE MODEL
# =========================

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.2f}")


# =========================
# WORD INDEX MAPPING
# =========================

# Dictionary mapping words → integer index
word_index = imdb.get_word_index()


# =========================
# CUSTOM REVIEW ENCODING
# =========================

def encode_review(text):
    """
    Convert raw text into IMDB-compatible integer sequence.
    """

    # Extract words from text
    words = re.findall(r"\b\w+\b", text.lower())

    encoded = []

    for w in words:
        # Get word index from IMDB dictionary
        idx = word_index.get(w)

        if idx is not None:
            # IMDB index shift (+3 reserved tokens)
            idx = idx + 3

            # Keep only known vocabulary
            if idx < max_features:
                encoded.append(idx)
            else:
                encoded.append(2)  # <UNK> token
        else:
            encoded.append(2)  # <UNK> token if word not found

    return encoded


# =========================
# PREDICTION ON CUSTOM TEXT
# =========================

# Example review
my_review = "this movie was amazing and the acting was wonderful"

# Convert text → integer sequence
encoded = encode_review(my_review)

# Pad sequence to match model input shape
padded = sequence.pad_sequences([encoded], maxlen=maxlen)

# Predict sentiment probability
proba = model.predict(padded, verbose=0)[0][0]

# Convert probability → label
pred = "positive" if proba >= 0.5 else "negative"


# =========================
# DISPLAY RESULT
# =========================

print("\nReview:", my_review)
print("Positive probability:", proba)
print("Prediction:", pred)
