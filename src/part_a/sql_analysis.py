
import sqlite3
import pandas as pd

# Connect to existing database
connection = sqlite3.connect("database/instamart.db")


def run_query(title, query):

    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)

    result = pd.read_sql_query(query, connection)

    print(result.to_string(index=False))


# 1. Total Sales
run_query(
    "TOTAL DELIVERED SALES",
    """
    SELECT
        COUNT(*) AS total_orders,
        ROUND(SUM(total_amount), 2) AS total_revenue
    FROM orders
    WHERE status = 'Delivered'
    """
)


# 2. Top 10 Customers
run_query(
    "TOP 10 CUSTOMERS",
    """
    SELECT
        c.customer_id,
        c.city,
        COUNT(o.order_id) AS orders,
        SUM(o.total_amount) AS spending
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'Delivered'
    GROUP BY c.customer_id, c.city
    ORDER BY spending DESC
    LIMIT 10
    """
)


# 3. Best Selling Products
run_query(
    "TOP 10 PRODUCTS",
    """
    SELECT
        p.product_id,
        p.product_name,
        SUM(o.quantity) AS units_sold,
        SUM(o.total_amount) AS revenue
    FROM products p
    JOIN orders o
        ON p.product_id = o.product_id
    WHERE o.status = 'Delivered'
    GROUP BY p.product_id, p.product_name
    ORDER BY units_sold DESC
    LIMIT 10
    """
)


# 4. City-wise Sales
run_query(
    "CITY WISE SALES",
    """
    SELECT
        c.city,
        COUNT(o.order_id) AS total_orders,
        SUM(o.total_amount) AS revenue
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'Delivered'
    GROUP BY c.city
    ORDER BY revenue DESC
    """
)


# 5. Monthly Revenue
run_query(
    "MONTHLY REVENUE",
    """
    SELECT
        strftime('%Y-%m', order_date) AS month,
        SUM(total_amount) AS revenue
    FROM orders
    WHERE status = 'Delivered'
    GROUP BY month
    ORDER BY month
    """
)


connection.close()

print("\nSQL ANALYSIS COMPLETED SUCCESSFULLY!")
