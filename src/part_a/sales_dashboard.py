
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Database connection
connection = sqlite3.connect("database/instamart.db")

# Output folder
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

# 1. City-wise revenue
city_data = pd.read_sql_query("""
    SELECT c.city, SUM(o.total_amount) AS revenue
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'Delivered'
    GROUP BY c.city
    ORDER BY revenue DESC
""", connection)

# 2. Monthly revenue
monthly_data = pd.read_sql_query("""
    SELECT
        strftime('%Y-%m', order_date) AS month,
        SUM(total_amount) AS revenue
    FROM orders
    WHERE status = 'Delivered'
    GROUP BY month
    ORDER BY month
""", connection)

connection.close()

# Create dashboard
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# City-wise bar chart
axes[0].bar(city_data["city"], city_data["revenue"])
axes[0].set_title("Instamart - City Wise Revenue")
axes[0].set_xlabel("City")
axes[0].set_ylabel("Revenue (INR)")
axes[0].grid(axis="y", alpha=0.3)

# Monthly revenue line chart
axes[1].plot(
    monthly_data["month"],
    monthly_data["revenue"],
    marker="o"
)
axes[1].set_title("Instamart - Monthly Revenue")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Revenue (INR)")
axes[1].tick_params(axis="x", rotation=60)
axes[1].grid(alpha=0.3)

plt.tight_layout()

# Save dashboard
plt.savefig(
    OUTPUT_PATH / "sales_dashboard.png",
    dpi=150,
    bbox_inches="tight"
)

print("Sales Dashboard created successfully!")
print("Saved: outputs/sales_dashboard.png")

plt.show()
