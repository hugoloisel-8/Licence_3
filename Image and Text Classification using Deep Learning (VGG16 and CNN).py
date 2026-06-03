# =========================
# IMPORTS
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import train_test_split

# Reduce TensorFlow logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =========================
# IMAGE CLASSIFICATION (VGG16)
# =========================

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

# Load pretrained VGG16 model (trained on ImageNet dataset)
vgg_model = vgg16.VGG16()

# Load image and resize it to 224x224 (required input size for VGG16)
img_src = image.load_img("border-collie-2648945_1920.jpg", target_size=(224, 224))

# Convert image to numerical array
img_test = image.img_to_array(img_src)

# Add batch dimension (model expects shape: (batch_size, height, width, channels))
img_test = np.expand_dims(img_test, axis=0)

# Apply VGG16-specific preprocessing (normalization + channel adjustments)
img_test = vgg16.preprocess_input(img_test)

# Run prediction
predictions_img = vgg_model.predict(img_test)

# Decode top-5 predicted classes into human-readable labels
decoded = vgg16.decode_predictions(predictions_img, top=5)

print("\nImage classification result:")
print(decoded)


# =========================
# TEXT DATASET LOADING
# =========================

df = pd.read_csv("./BBC_Text_CLS.csv")

print(df.head(5))

# Convert string labels into numeric IDs (required for neural networks)
df["targets"] = df["labels"].astype("category").cat.codes

# Number of unique classes (topics/categories)
K = df["targets"].nunique()

# Split dataset into training and testing sets
train, test = train_test_split(df, test_size=0.3, random_state=42)


# =========================
# TOKENIZATION
# =========================

MAX_VOCAB_SIZE = 2000

# Create tokenizer (keeps only most frequent words)
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

# Build vocabulary from training text only (avoids data leakage)
tokenizer.fit_on_texts(train["text"])

# Convert text into sequences of integers (word IDs)
seq_train = tokenizer.texts_to_sequences(train["text"])
seq_test = tokenizer.texts_to_sequences(test["text"])

# Word dictionary (word -> index)
word2idx = tokenizer.word_index

# Total vocabulary size
V = len(word2idx)

print(f"\nVocabulary size: {V}")


# =========================
# PADDING SEQUENCES
# =========================

MAX_SEQUENCE_LENGTH = 100

# Pad sequences so all inputs have same length (required for CNN)
data_train = pad_sequences(seq_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
data_test = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Final sequence length used by model
T = data_train.shape[1]

print(f"Sequence length: {T}")
print(f"Train shape: {data_train.shape}, Test shape: {data_test.shape}")


# =========================
# LABEL MAPPING DISPLAY
# =========================

classes = (
    df[['targets', 'labels']]
    .drop_duplicates()
    .sort_values('targets')
)

for _, row in classes.iterrows():
    print(f"{row['targets']} -> {row['labels']}")


# =========================
# CNN MODEL (TEXT CLASSIFICATION)
# =========================

EMBEDDING_DIM = 50

# Input layer: sequence of word indices
input_ = Input(shape=(T,))

# Convert word indices into dense vectors (word embeddings)
x = Embedding(input_dim=V + 1, output_dim=EMBEDDING_DIM)(input_)

# First convolution layer (extract local patterns like word groups)
x = Conv1D(filters=128, kernel_size=5, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)

# Second convolution layer (learn higher-level features)
x = Conv1D(filters=64, kernel_size=5, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)

# Third convolution layer + global pooling (compress full sequence into vector)
x = Conv1D(filters=64, kernel_size=5, activation="relu")(x)
x = GlobalMaxPooling1D()(x)

# Dropout to reduce overfitting (randomly disables neurons during training)
x = Dropout(0.5)(x)

# Output layer: one neuron per class (softmax gives probabilities)
output = Dense(K, activation="softmax")(x)

# Define model
model = Model(inputs=input_, outputs=output)

# Compile model (define optimizer + loss function)
model.compile(
    loss=SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# =========================
# TRAINING
# =========================

history = model.fit(
    data_train,
    train["targets"],
    epochs=50,
    validation_data=(data_test, test["targets"])
)


# =========================
# TRAINING VISUALIZATION
# =========================

plt.figure(figsize=(14, 6))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# =========================
# NEW TEXT PREDICTION SECTION
# =========================

# New unseen texts (real-world examples for testing generalization)
new_texts = [
    "Pope Francis says migration is a reality in call for action during France visit",
    "The company reported higher profits and rising sales this year.",
    "The football team won the match after scoring twice in the second half.",
    "The new software update improves computer security and performance.",
    "The actor received an award for the film at the festival."
]

# Step 1: Convert raw text into sequences using the SAME tokenizer
# IMPORTANT: we reuse training tokenizer so word mapping stays consistent
seq_new = tokenizer.texts_to_sequences(new_texts)

# Step 2: Pad sequences to match model input size
data_new = pad_sequences(seq_new, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Step 3: Run model prediction
# Output = probability distribution over all classes for each text
predictions = model.predict(data_new, verbose=0)

# Step 4: Convert probabilities to class index (highest probability wins)
predicted_classes = np.argmax(predictions, axis=1)

# Step 5: Build mapping from numeric labels back to original category names
label_mapping = (
    df[['targets', 'labels']]
    .drop_duplicates()
    .sort_values('targets')
    .set_index('targets')['labels']
    .to_dict()
)

# Step 6: Convert predicted indices into human-readable labels
predicted_labels = [label_mapping[idx] for idx in predicted_classes]


# =========================
# DISPLAY RESULTS
# =========================

for i in range(len(new_texts)):
    print("\n==============================")
    print(f"Text {i+1}:")
    print(new_texts[i])

    # Show predicted class index (numeric output of model)
    print(f"Predicted class index: {predicted_classes[i]}")

    # Show human-readable category (business, sport, tech, etc.)
    print(f"Predicted label: {predicted_labels[i]}")

    # Show full probability distribution (confidence for each class)
    print("Class probabilities:")
    print(predictions[i])
