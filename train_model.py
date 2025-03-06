import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Kaggle dataset
print("Loading dataset...")
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "text"]

# Convert labels to binary values (spam = 1, ham = 0)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Transform text using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
print("Training model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save Model and Vectorizer
print("Saving model and vectorizer...")
with open("models/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model training complete! Files saved in 'models/' directory.")

