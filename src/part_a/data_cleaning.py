
import pandas as pd
from pathlib import Path

# File locations
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# General data cleaning function
def clean_data(df, name):

    print(f"\nCleaning {name} dataset...")

    print("Before cleaning:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove unwanted spaces from text columns
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # Handle missing values
    for col in df.columns:

        if df[col].isnull().any():

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

            else:
                mode = df[col].mode()

                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])

    print("After cleaning:", df.shape)

    return df


# 1. Load datasets
customers = pd.read_csv(RAW_PATH / "customers.csv")
products = pd.read_csv(RAW_PATH / "products.csv")
orders = pd.read_csv(RAW_PATH / "orders.csv")


# 2. Clean datasets
customers = clean_data(customers, "Customers")
products = clean_data(products, "Products")
orders = clean_data(orders, "Orders")


# 3. Convert order date
orders["order_date"] = pd.to_datetime(
    orders["order_date"], errors="coerce"
)

orders = orders.dropna(subset=["order_date"])


# 4. Validate customer and product references
orders = orders[
    orders["customer_id"].isin(customers["customer_id"])
]

orders = orders[
    orders["product_id"].isin(products["product_id"])
]


# 5. Save cleaned datasets
customers.to_csv(
    PROCESSED_PATH / "customers_cleaned.csv", index=False
)

products.to_csv(
    PROCESSED_PATH / "products_cleaned.csv", index=False
)

orders.to_csv(
    PROCESSED_PATH / "orders_cleaned.csv", index=False
)


# 6. Final report
print("\n========== DATA CLEANING COMPLETED ==========")

print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)

print("\nAll cleaned CSV files saved successfully!")
