from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Example sentences (translated into English)
sentences = [
    "Pineapple originates from South America (Paraguay, northeastern Argentina, and southern Brazil).",
    "The hospital receives patients.",
    "The nurse cares for patients.",
    "The banana is a yellow fruit.",
    "The apple is a sweet fruit.",
    "The kiwi pulp, usually green (sometimes yellow), is sweet and tangy.",
    "The car is a means of transportation.",
    "The bus is public transport.",
    "The doctor works at the hospital.",
    "The grape is a small fruit.",
    "The train is a fast means of transport.",
    "The bicycle is an eco-friendly means of transportation."
]

# Load pretrained multilingual sentence embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Convert sentences into vector embeddings
X = model.encode(sentences)

# Display embedding information
print("Embedding shape:", X.shape)
print("\nFirst 3 sentence embeddings (first 10 values only):\n")

for i in range(3):
    print("Sentence:", sentences[i])
    print("First 10 values:", X[i][:10])  # shorten output for readability
    print("-" * 50)

# Normalize embeddings (important for clustering performance)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Apply KMeans clustering to group similar sentences
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)

# Reduce dimensions from high-dimensional space to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(8, 6))

for i in range(len(sentences)):
    # Plot each sentence in 2D space
    plt.scatter(X_2d[i, 0], X_2d[i, 1])

    # Annotate each point with the sentence text
    plt.text(X_2d[i, 0] + 0.01, X_2d[i, 1] + 0.01, sentences[i], fontsize=8)

plt.title("K-means clustering with sentence embeddings")
plt.show()
