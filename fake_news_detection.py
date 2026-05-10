# ==========================================
# Fake News Detection System
# ==========================================

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================================
# Load Dataset
# ==========================================

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add Labels
fake['label'] = 0
real['label'] = 1

# Merge datasets
data = pd.concat([fake.head(50), real.head(50)])

print("\nDataset Loaded Successfully!")

# ==========================================
# Text Cleaning
# ==========================================

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = re.sub(r'[^a-zA-Z]', ' ', str(text))

    text = text.lower()

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

# Clean text
data['text'] = data['text'].apply(clean_text)

print("\nText Cleaning Completed!")

# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['text'])

y = data['label']

print("\nTF-IDF Vectorization Completed!")

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Logistic Regression Model
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# Prediction
# ==========================================

pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()