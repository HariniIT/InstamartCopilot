
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

data_path = Path("data/processed/churn_training_data.csv")
df = pd.read_csv(data_path)

print("=" * 60)
print("CHURN PREDICTION - MODEL COMPARISON V3")
print("=" * 60)

print("\nDATASET SUMMARY")
print("-" * 40)
print("Total customers:", len(df))
print("\nChurn distribution:")
print(df["churn"].value_counts())

# --------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# --------------------------------------------------

# Customer ID is not a useful prediction feature
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# Numerical and categorical columns
numeric_features = [
    "age",
    "past_orders",
    "orders_previous_90d",
    "orders_recent_90d",
    "avg_order_value",
    "past_spending",
    "days_since_last_order"
]

categorical_features = ["membership"]

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

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 5. CREATE MODELS
# --------------------------------------------------

models = {
    "Dummy Baseline": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Churn Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro",
            zero_division=0
        )
    })

    trained_models[name] = model

# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="Balanced Accuracy",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.round(3).to_string(index=False))

# Select model using balanced accuracy
best_model_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# --------------------------------------------------
# 8. BEST MODEL EVALUATION
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print("-" * 40)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Active", "Churn"],
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print("-" * 40)
print(confusion_matrix(y_test, best_predictions))

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

output_dir = Path("outputs/part_b")
output_dir.mkdir(parents=True, exist_ok=True)

# Save model comparison
comparison.to_csv(
    output_dir / "model_comparison_v3.csv",
    index=False
)

# Save test predictions
test_results = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"],
    "actual_churn": y_test,
    "predicted_churn": best_predictions
})

test_results.to_csv(
    output_dir / "churn_test_predictions_v3.csv",
    index=False
)

# Refit selected model using all available data
best_model.fit(X, y)

joblib.dump(
    best_model,
    output_dir / "best_churn_model_v3.pkl"
)

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved files:")
print("- outputs/part_b/model_comparison_v3.csv")
print("- outputs/part_b/churn_test_predictions_v3.csv")
print("- outputs/part_b/best_churn_model_v3.pkl")
