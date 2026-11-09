
import sqlite3
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "instamart.db"
OUTPUT_DIR = BASE_DIR / "outputs" / "part_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load orders
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

# 2. Define historical observation and future prediction periods
reference_date = orders["order_date"].max()

observation_start = reference_date - pd.Timedelta(days=180)
prediction_start = reference_date - pd.Timedelta(days=90)

# Historical period: 180 to 90 days before reference date
historical = orders[
    (orders["order_date"] >= observation_start) &
    (orders["order_date"] < prediction_start) &
    (orders["status"] == "Delivered")
].copy()

# Future period: last 90 days
future = orders[
    (orders["order_date"] >= prediction_start) &
    (orders["order_date"] <= reference_date) &
    (orders["status"] == "Delivered")
].copy()

# 3. Historical customer behaviour features
features = historical.groupby("customer_id").agg(
    past_orders=("order_id", "count"),
    past_spending=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

# Include all customers
data = customers.merge(
    features,
    on="customer_id",
    how="left"
)

data[["past_orders", "past_spending", "avg_order_value"]] = (
    data[["past_orders", "past_spending", "avg_order_value"]]
    .fillna(0)
)

# 4. Create target using future 90-day behaviour
future_customers = future["customer_id"].unique()

data["churn"] = (
    ~data["customer_id"].isin(future_customers)
).astype(int)

# 5. Select only historical features
feature_columns = [
    "age",
    "past_orders",
    "past_spending",
    "avg_order_value"
]

X = data[feature_columns]
y = data["churn"]

print("CUSTOMER DATASET SUMMARY")
print("------------------------")
print("Total customers:", len(data))
print("Historical period:", observation_start.date(), "to", prediction_start.date())
print("Future period:", prediction_start.date(), "to", reference_date.date())
print("\nChurn distribution:")
print(y.value_counts())

# 6. Check whether both classes exist
if y.nunique() < 2:
    raise ValueError(
        "Only one churn class found. "
        "This dataset is not sufficient for classification."
    )

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 9. Evaluate
predictions = model.predict(X_test)

print("\nCHURN PREDICTION V2 COMPLETED!")
print("--------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions))

# 10. Predict customer risk
data["churn_probability"] = model.predict_proba(X)[:, 1]

data["churn_status"] = data["churn_probability"].apply(
    lambda probability: "At Risk" if probability >= 0.5 else "Active"
)

# 11. Save predictions
result = data[
    [
        "customer_id",
        "past_orders",
        "past_spending",
        "avg_order_value",
        "churn",
        "churn_probability",
        "churn_status"
    ]
]

result.to_csv(
    OUTPUT_DIR / "churn_predictions_v2.csv",
    index=False
)

# 12. Save trained model
joblib.dump(
    model,
    OUTPUT_DIR / "churn_model_v2.pkl"
)

# 13. Feature importance
importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "churn_feature_importance.csv",
    index=False
)

print("\nCUSTOMER RISK SUMMARY:")
print(result["churn_status"].value_counts())

print("\nFEATURE IMPORTANCE:")
print(importance.to_string(index=False))

print("\nSaved files:")
print("churn_predictions_v2.csv")
print("churn_model_v2.pkl")
print("churn_feature_importance.csv")
