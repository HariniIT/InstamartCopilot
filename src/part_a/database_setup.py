
import sqlite3
import pandas as pd
from pathlib import Path

# Project paths
DATA_PATH = Path("data/processed")
DB_PATH = Path("database/instamart.db")

# Create database folder
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Read cleaned CSV files
customers = pd.read_csv(DATA_PATH / "customers_cleaned.csv")
products = pd.read_csv(DATA_PATH / "products_cleaned.csv")
orders = pd.read_csv(DATA_PATH / "orders_cleaned.csv")

# Connect to SQLite database
connection = sqlite3.connect(DB_PATH)

# Enable foreign key support
connection.execute("PRAGMA foreign_keys = ON")

cursor = connection.cursor()

# Create Customers table
cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    age INTEGER,
    city TEXT,
    total_orders INTEGER,
    total_spent REAL,
    membership TEXT
)
""")

# Create Products table
cursor.execute("""
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price REAL,
    stock INTEGER
)
""")

# Create Orders table
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price REAL,
    total_amount REAL,
    order_date TEXT,
    payment_method TEXT,
    status TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
""")

# Clear old records so the script can be run again
cursor.execute("DELETE FROM orders")
cursor.execute("DELETE FROM customers")
cursor.execute("DELETE FROM products")

# Insert customer records
customers.to_sql(
    "customers", connection, if_exists="append", index=False
)

# Insert product records
products.to_sql(
    "products", connection, if_exists="append", index=False
)

# Insert order records
orders.to_sql(
    "orders", connection, if_exists="append", index=False
)

# Save changes
connection.commit()

# Display database information
print("DATABASE CREATED SUCCESSFULLY!")

for table in ["customers", "products", "orders"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"{table}: {count} records")

connection.close()

print("\nSQLite connection closed.")
