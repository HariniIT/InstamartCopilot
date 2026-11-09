
import sqlite3
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "instamart.db"
OUTPUT_DIR = BASE_DIR / "outputs" / "part_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
connection = sqlite3.connect(DB_PATH)

orders = pd.read_sql_query(
    "SELECT * FROM orders",
    connection
)

customers = pd.read_sql_query(
    "SELECT customer_id, age, city, membership FROM customers",
    connection
)

connection.close()

orders["order_date"] = pd.to_datetime(orders["order_date"])

# 2. Define observation and prediction windows
reference_date = orders["order_date"].max()

observation_start = reference_date - pd.Timedelta(days=180)
prediction_start = reference_date - pd.Timedelta(days=90)

historical = orders[
    (orders["order_date"] >= observation_start) &
    (orders["order_date"] < prediction_start) &
    (orders["status"] == "Delivered")
].copy()

future = orders[
    (orders["order_date"] >= prediction_start) &
    (orders["order_date"] <= reference_date) &
    (orders["status"] == "Delivered")
].copy()

# 3. Create historical features
features = historical.groupby("customer_id").agg(
    past_orders=("order_id", "count"),
    past_spending=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

data = customers.merge(
    features,
    on="customer_id",
    how="left"
)

data[["past_orders", "past_spending", "avg_order_value"]] = (
    data[["past_orders", "past_spending", "avg_order_value"]]
    .fillna(0)
)

# 4. Create future churn label
future_customers = future["customer_id"].unique()

data["churn"] = (
    ~data["customer_id"].isin(future_customers)
).astype(int)

# 5. Features - Age removed for comparison
feature_columns = [
    "past_orders",
    "past_spending",
    "avg_order_value"
]

X = data[feature_columns]
y = data["churn"]

print("DATASET SUMMARY")
print("----------------")
print("Total customers:", len(data))
print("Churn distribution:")
print(y.value_counts())

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Define models
models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
}

results = []
trained_models = {}

# 8. Train and evaluate
for name, model in models.items():

    if name == "Dummy Baseline":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 3),
        "Precision": round(
            precision_score(y_test, predictions, zero_division=0), 3
        ),
        "Recall": round(
            recall_score(y_test, predictions, zero_division=0), 3
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, zero_division=0), 3
        )
    })

    trained_models[name] = model

# 9. Compare results
comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("----------------")
print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# 10. Select best model by F1 score
# Dummy is baseline only, not a candidate
candidate_results = comparison[
    comparison["Model"] != "Dummy Baseline"
]

best_name = candidate_results.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_name]

# 11. Evaluate best model
best_predictions = best_model.predict(X_test)

print("\nBEST MODEL:", best_name)

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, best_predictions))

# 12. Save best model
joblib.dump(
    best_model,
    OUTPUT_DIR / "best_churn_model.pkl"
)

# 13. Save comparison details
print("\nSaved files:")
print("outputs/part_b/model_comparison.csv")
print("outputs/part_b/best_churn_model.pkl")

print("\nMODEL COMPARISON COMPLETED SUCCESSFULLY!")
