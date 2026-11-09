
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/processed/churn_training_data.csv")

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

print("=" * 60)
print("CHURN MODEL - 5 FOLD CROSS VALIDATION")
print("=" * 60)

print("\nTotal customers:", len(df))
print("Active customers:", (y == 0).sum())
print("Churn customers:", (y == 1).sum())

# --------------------------------------------------
# 2. FEATURE PREPROCESSING
# --------------------------------------------------

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

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

models = {
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
# 4. STRATIFIED 5-FOLD CROSS VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall"
}

results = []

for name, model in models.items():

    print("\nEvaluating:", name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy": scores[
            "test_balanced_accuracy"
        ].mean(),
        "Precision": scores["test_precision"].mean(),
        "Churn Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean(),
        "Accuracy Std": scores["test_accuracy"].std()
    })

    print("Fold Accuracy:",
          np.round(scores["test_accuracy"], 3))

# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="Balanced Accuracy",
    ascending=False
)

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 60)

print(comparison.round(3).to_string(index=False))

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

output_dir = Path("outputs/part_b")
output_dir.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output_dir / "churn_cross_validation.csv",
    index=False
)

print("\nCROSS VALIDATION COMPLETED SUCCESSFULLY!")
print("Saved: outputs/part_b/churn_cross_validation.csv")
