
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Data storage location
DATA_PATH = Path("data/raw")
DATA_PATH.mkdir(parents=True, exist_ok=True)


# 1. CUSTOMER DATASET
customers = pd.DataFrame({
    "customer_id": range(1, 1001),
    "age": np.random.randint(18, 65, 1000),
    "city": np.random.choice(
        ["Chennai", "Trichy", "Coimbatore", "Madurai"], 1000
    ),
    "total_orders": np.random.randint(1, 50, 1000),
    "total_spent": np.random.randint(100, 20000, 1000),
    "membership": np.random.choice(
        ["Gold", "Silver", "Regular"], 1000
    )
})


# 2. PRODUCT DATASET
products = pd.DataFrame({
    "product_id": range(1, 101),
    "product_name": np.random.choice(
        ["Milk", "Rice", "Bread", "Eggs", "Apple",
         "Oil", "Sugar", "Coffee", "Biscuits", "Juice"],
        100
    ),
    "category": np.random.choice(
        ["Grocery", "Dairy", "Fruits", "Beverages"],
        100
    ),
    "price": np.random.randint(20, 1000, 100),
    "stock": np.random.randint(0, 500, 100)
})


# 3. ORDER DATASET
number_of_orders = 3000

product_ids = np.random.randint(1, 101, number_of_orders)
quantities = np.random.randint(1, 6, number_of_orders)

unit_prices = products.loc[
    product_ids - 1, "price"
].to_numpy()

orders = pd.DataFrame({
    "order_id": range(1, number_of_orders + 1),
    "customer_id": np.random.randint(1, 1001, number_of_orders),
    "product_id": product_ids,
    "quantity": quantities,
    "unit_price": unit_prices,
    "total_amount": quantities * unit_prices,
    "order_date": (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            np.random.randint(0, 669, number_of_orders), unit="D"
        )
    ),
    "payment_method": np.random.choice(
        ["UPI", "Card", "Cash", "Wallet"], number_of_orders
    ),
    "status": np.random.choice(
        ["Delivered", "Cancelled", "Pending"], number_of_orders
    )
})


# 4. SAVE ALL CSV FILES
customers.to_csv(DATA_PATH / "customers.csv", index=False)
products.to_csv(DATA_PATH / "products.csv", index=False)
orders.to_csv(DATA_PATH / "orders.csv", index=False)


# 5. DISPLAY RESULTS
print("ALL DATASETS CREATED SUCCESSFULLY!")

print("\nCustomers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)

print("\nProduct Sample:")
print(products.head())

print("\nOrder Sample:")
print(orders.head())
