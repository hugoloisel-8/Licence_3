texts = [
    "hello sir thank you very much",
    "hey how are you",
    "please receive my regards",
    "hi thanks :)",
    "I kindly ask you",
    "hello :)"
]

labels = ["formal", "informal", "formal", "informal", "formal", "informal"]

vectorizer = CountVectorizer()

# Convert text into numerical feature vectors (Bag of Words)
X = vectorizer.fit_transform(texts)

# Train the model on the data
model.fit(X, labels)

# Test new sentences
test = vectorizer.transform([
    "hello",
    "hi how are you",
    "thank you very much"
])

print(model.predict(test))
