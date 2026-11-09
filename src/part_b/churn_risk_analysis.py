
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

# --------------------------------------------------
# 1. LOAD DATASET AND TRAINED MODEL
# --------------------------------------------------

data_path = Path("data/processed/churn_training_data.csv")
model_path = Path("outputs/part_b/best_churn_model_v3.pkl")

df = pd.read_csv(data_path)
model = joblib.load(model_path)

print("=" * 60)
print("CUSTOMER CHURN RISK ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 2. PREPARE FEATURES
# --------------------------------------------------

X = df.drop(columns=["customer_id", "churn"])

# --------------------------------------------------
# 3. PREDICT CHURN PROBABILITY
# --------------------------------------------------

probabilities = model.predict_proba(X)[:, 1]

# --------------------------------------------------
# 4. CLASSIFY CUSTOMER RISK
# --------------------------------------------------

risk_levels = []

for probability in probabilities:

    if probability >= 0.65:
        risk_levels.append("High Risk")

    elif probability >= 0.40:
        risk_levels.append("Medium Risk")

    else:
        risk_levels.append("Low Risk")

# --------------------------------------------------
# 5. CREATE CUSTOMER RISK REPORT
# --------------------------------------------------

results = pd.DataFrame({
    "customer_id": df["customer_id"],
    "actual_churn": df["churn"],
    "churn_probability": probabilities.round(4),
    "risk_level": risk_levels
})

# Sort by highest churn probability
results = results.sort_values(
    by="churn_probability",
    ascending=False
)

# --------------------------------------------------
# 6. RISK SUMMARY
# --------------------------------------------------

risk_summary = (
    results["risk_level"]
    .value_counts()
    .reindex(
        ["High Risk", "Medium Risk", "Low Risk"],
        fill_value=0
    )
    .reset_index()
)

risk_summary.columns = ["Risk Level", "Customer Count"]

print("\nCUSTOMER RISK SUMMARY")
print("-" * 40)
print(risk_summary.to_string(index=False))

print("\nTOTAL CUSTOMERS:", len(results))

# --------------------------------------------------
# 7. SAVE CSV REPORTS
# --------------------------------------------------

output_dir = Path("outputs/part_b")
output_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(
    output_dir / "churn_risk_predictions.csv",
    index=False
)

risk_summary.to_csv(
    output_dir / "churn_risk_summary.csv",
    index=False
)

# --------------------------------------------------
# 8. CREATE RISK DASHBOARD
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    risk_summary["Risk Level"],
    risk_summary["Customer Count"]
)

plt.title("Customer Churn Risk Analysis")
plt.xlabel("Customer Risk Level")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    output_dir / "churn_risk_dashboard.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 9. DISPLAY HIGH-RISK CUSTOMERS
# --------------------------------------------------

print("\nTOP 10 HIGH-RISK CUSTOMERS")
print("-" * 40)

print(
    results[
        results["risk_level"] == "High Risk"
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("CHURN RISK ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved files:")
print("- churn_risk_predictions.csv")
print("- churn_risk_summary.csv")
print("- churn_risk_dashboard.png")
