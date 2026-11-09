
import pandas as pd
import numpy as np
from pathlib import Path

# Reproducible random data
rng = np.random.default_rng(42)

# Create 1000 synthetic customers
n = 1000

# Customer basic details
customer_id = np.arange(1, n + 1)
age = rng.integers(18, 65, n)

membership = rng.choice(
    ["Basic", "Silver", "Gold"],
    size=n,
    p=[0.45, 0.35, 0.20]
)

# Historical customer behaviour
past_orders = rng.poisson(lam=12, size=n) + 1

orders_previous_90d = rng.poisson(lam=6, size=n)
orders_recent_90d = rng.poisson(lam=5, size=n)

avg_order_value = np.round(
    rng.uniform(150, 1200, n), 2
)

past_spending = np.round(
    past_orders * avg_order_value, 2
)

days_since_last_order = rng.integers(1, 121, n)

# Membership influence
membership_effect = np.where(
    membership == "Basic", 0.15,
    np.where(membership == "Silver", 0.05, -0.10)
)

# Churn probability based on historical behaviour
probability = (
    0.15
    + (days_since_last_order / 120) * 0.35
    + np.maximum(orders_previous_90d - orders_recent_90d, 0) * 0.035
    + membership_effect
)

# Keep probability within a realistic range
probability = np.clip(probability, 0.05, 0.85)

# Generate future churn outcome with randomness
churn = rng.binomial(1, probability)

# Create dataset
df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "membership": membership,
    "past_orders": past_orders,
    "orders_previous_90d": orders_previous_90d,
    "orders_recent_90d": orders_recent_90d,
    "avg_order_value": avg_order_value,
    "past_spending": past_spending,
    "days_since_last_order": days_since_last_order,
    "churn": churn
})

# Save separately; original project data remains untouched
output_path = Path("data/processed/churn_training_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

# Display results
print("=" * 50)
print("CHURN DATASET GENERATED SUCCESSFULLY!")
print("=" * 50)

print("\nDataset shape:", df.shape)

print("\nChurn distribution:")
print(df["churn"].value_counts().rename({
    0: "Active",
    1: "Churn"
}))

print("\nChurn percentage:")
print((df["churn"].value_counts(normalize=True) * 100).round(2))

print("\nFirst 5 records:")
print(df.head())

print("\nSaved:", output_path)
