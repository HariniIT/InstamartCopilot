
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD CLEANED DATASET
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/support_tickets_cleaned.csv"
)

print("=" * 60)
print("CUSTOMER SUPPORT TICKET CATEGORY PREDICTION")
print("=" * 60)

print("\nTotal tickets:", len(df))

# --------------------------------------------------
# 2. SELECT INPUT AND TARGET
# --------------------------------------------------

X = df["cleaned_text"]
y = df["category"]

print("\nCategories:")
print(y.value_counts())

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. CREATE TF-IDF + LOGISTIC REGRESSION PIPELINE
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
            random_state=42
        )
    )
])

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining category prediction model...")

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. PREDICT TEST DATA
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print("-" * 50)

labels = sorted(y.unique())

print("Category order:", labels)
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )
)

# --------------------------------------------------
# 8. SAVE MODEL AND PREDICTIONS
# --------------------------------------------------

output_dir = Path("outputs/part_c")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    output_dir / "ticket_category_model.pkl"
)

test_results = pd.DataFrame({
    "actual_category": y_test,
    "predicted_category": predictions
})

test_results.to_csv(
    output_dir / "ticket_category_predictions.csv",
    index=True,
    index_label="ticket_index"
)

# --------------------------------------------------
# 9. TEST WITH NEW CUSTOMER COMPLAINTS
# --------------------------------------------------

new_tickets = [
    "My order delivery is delayed",
    "Money was deducted but my order was not placed",
    "I received a damaged product",
    "I have not received my refund",
    "I cannot log in to my account"
]

print("\nNEW CUSTOMER TICKET PREDICTIONS")
print("-" * 50)

for ticket in new_tickets:

    cleaned_ticket = ticket.lower()

    predicted_category = model.predict(
        [cleaned_ticket]
    )[0]

    print("\nTicket:", ticket)
    print("Predicted category:", predicted_category)

# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORY PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved files:")
print("- outputs/part_c/ticket_category_model.pkl")
print("- outputs/part_c/ticket_category_predictions.csv")
