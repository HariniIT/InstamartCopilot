
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

# Load cleaned customer data
customers = pd.read_csv(
    BASE_DIR / "data/processed/customers_cleaned.csv"
)

# Select customer behaviour features
features = customers[
    ["age", "total_orders", "total_spent"]
].copy()

# Handle missing values
features = features.fillna(features.median())

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

customers["cluster"] = kmeans.fit_predict(scaled_data)

# Analyse clusters
summary = customers.groupby("cluster").agg(
    customer_count=("customer_id", "count"),
    avg_orders=("total_orders", "mean"),
    avg_spent=("total_spent", "mean")
).round(2)

# Name customer segments based on spending
spending_order = summary["avg_spent"].sort_values(
    ascending=False
).index.tolist()

segment_names = {
    spending_order[0]: "High Value",
    spending_order[1]: "Regular",
    spending_order[2]: "Low Value"
}

customers["segment"] = customers["cluster"].map(segment_names)

# Save results
output_dir = BASE_DIR / "outputs" / "part_b"
output_dir.mkdir(parents=True, exist_ok=True)

customers.to_csv(
    output_dir / "customer_segments.csv",
    index=False
)

summary.to_csv(
    output_dir / "segment_summary.csv"
)

# Create visualization
plt.figure(figsize=(10, 6))

for segment in customers["segment"].unique():
    subset = customers[customers["segment"] == segment]

    plt.scatter(
        subset["total_orders"],
        subset["total_spent"],
        label=segment,
        alpha=0.6
    )

plt.title("Customer Segmentation - K-Means")
plt.xlabel("Total Orders")
plt.ylabel("Total Spending")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "customer_segments.png",
    dpi=150
)

plt.close()

print("CUSTOMER SEGMENTATION COMPLETED!")
print("\nSEGMENT SUMMARY:")
print(summary)
print("\nSaved files:")
print("outputs/part_b/customer_segments.csv")
print("outputs/part_b/segment_summary.csv")
print("outputs/part_b/customer_segments.png")
