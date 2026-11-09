
import pandas as pd
import sqlite3
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "instamart.db"
OUTPUT_DIR = BASE_DIR / "outputs" / "part_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load customer and order data
connection = sqlite3.connect(DB_PATH)

customers = pd.read_sql_query(
    "SELECT * FROM customers", connection
)

orders = pd.read_sql_query(
    "SELECT * FROM orders", connection
)

connection.close()

# 2. Convert order date
orders["order_date"] = pd.to_datetime(orders["order_date"])

# 3. Calculate customer purchasing behaviour
delivered_orders = orders[orders["status"] == "Delivered"].copy()

reference_date = orders["order_date"].max()

customer_features = delivered_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_spent=("total_amount", "sum"),
    last_order_date=("order_date", "max")
).reset_index()

customer_features["days_since_last_order"] = (
    reference_date - customer_features["last_order_date"]
).dt.days

# 4. Merge customer information
data = customers.merge(
    customer_features,
    on="customer_id",
    how="left",
    suffixes=("_original", "")
)

# Fill missing purchasing behaviour
data["total_orders"] = data["total_orders"].fillna(0)
data["total_spent"] = data["total_spent"].fillna(0)

data["days_since_last_order"] = (
    data["days_since_last_order"].fillna(
        (reference_date - pd.Timestamp("2025-01-01")).days
    )
)

data["days_since_last_order"] = (
    data["days_since_last_order"].fillna(
        (reference_date - pd.Timestamp("2025-01-01")).days
    )
)

# 5. Create demo churn target
# Churn = No delivered order in the last 90 days
data["churn"] = (
    data["days_since_last_order"] > 90
).astype(int)

# 6. Select features
features = [
    "age",
    "total_orders",
    "total_spent",
    "days_since_last_order"
]

X = data[features]
y = data["churn"]

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 9. Predict churn
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nCHURN PREDICTION COMPLETED!")
print("--------------------------------")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 10. Predict for all customers
data["churn_prediction"] = model.predict(X)

data["churn_status"] = data["churn_prediction"].map({
    0: "Active",
    1: "At Risk"
})

# 11. Save customer predictions
result = data[
    [
        "customer_id",
        "total_orders",
        "total_spent",
        "days_since_last_order",
        "churn",
        "churn_prediction",
        "churn_status"
    ]
]

result.to_csv(
    OUTPUT_DIR / "churn_predictions.csv",
    index=False
)

# 12. Save model
import joblib

joblib.dump(
    model,
    OUTPUT_DIR / "churn_model.pkl"
)

# 13. Display summary
print("\nCHURN CUSTOMER SUMMARY:")
print(result["churn_status"].value_counts())

print("\nSaved files:")
print("outputs/part_b/churn_predictions.csv")
print("outputs/part_b/churn_model.pkl")
