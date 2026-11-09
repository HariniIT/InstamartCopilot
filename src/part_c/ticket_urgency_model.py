
import os
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

print("=" * 60)
print("CUSTOMER SUPPORT TICKET URGENCY PREDICTION")
print("=" * 60)

print("\nDATASET SUMMARY")
print("-" * 60)

print("Total Tickets:", len(df))

print("\nUrgency Distribution:")
print(df["urgency"].value_counts())

# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

X = df["cleaned_text"].fillna("")
y = df["urgency"]

# --------------------------------------------------
# 4. TRAIN TEST SPLIT
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
# 5. CREATE MODEL
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

print("\nTraining urgency prediction model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    "ticket_urgency_model.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved:", model_path)

# --------------------------------------------------
# 10. SAVE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "actual_urgency": y_test.values,
    "predicted_urgency": y_pred
})

prediction_path = os.path.join(
    OUTPUT_DIR,
    "ticket_urgency_predictions.csv"
)

results.to_csv(prediction_path, index=False)

print("Predictions saved:", prediction_path)

# --------------------------------------------------
# 11. TEST NEW CUSTOMER MESSAGES
# --------------------------------------------------

sample_messages = [
    "My order is slightly delayed",
    "My payment is stuck and I need help",
    "I need immediate assistance with my order"
]

sample_predictions = model.predict(sample_messages)

print("\nNEW CUSTOMER URGENCY PREDICTIONS")
print("-" * 60)

for message, prediction in zip(
    sample_messages,
    sample_predictions
):
    print("\nCustomer Message:", message)
    print("Predicted Urgency:", prediction)

# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("URGENCY PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
