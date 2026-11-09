
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

INPUT_FILE = "data/processed/support_tickets_cleaned.csv"
OUTPUT_DIR = "outputs/part_c"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 55)
print("CUSTOMER SUPPORT SENTIMENT ANALYSIS")
print("=" * 55)

print("\nDATASET SUMMARY")
print("-" * 55)

print("Total Tickets:", len(df))
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

X = df["cleaned_text"].fillna("")
y = df["sentiment"]

# --------------------------------------------------
# 4. TRAIN AND TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------------------------
# 5. CREATE SENTIMENT MODEL
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nSentiment model training completed!")

# --------------------------------------------------
# 7. MODEL PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 55)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    "ticket_sentiment_model.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved:", model_path)

# --------------------------------------------------
# 10. SAVE TEST PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "actual_sentiment": y_test.values,
    "predicted_sentiment": y_pred
})

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ticket_sentiment_predictions.csv"
    ),
    index=False
)

print("Predictions saved successfully!")

# --------------------------------------------------
# 11. TEST NEW CUSTOMER MESSAGES
# --------------------------------------------------

sample_messages = [
    "I am very happy with the fast delivery",
    "My payment failed and money was deducted",
    "The order was delivered yesterday"
]

# Apply the same basic text cleaning used earlier.
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    stop_words = {
        "the", "is", "are", "was", "were",
        "a", "an", "to", "of", "in", "on",
        "for", "my", "i", "it", "this", "that",
        "and", "with"
    }

    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

cleaned_messages = [
    clean_text(message)
    for message in sample_messages
]

sample_predictions = model.predict(cleaned_messages)

print("\nNEW CUSTOMER SENTIMENT PREDICTIONS")
print("-" * 55)

for message, prediction in zip(
    sample_messages,
    sample_predictions
):
    print("\nCustomer Message:", message)
    print("Predicted Sentiment:", prediction)

print("\n" + "=" * 55)
print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 55)
