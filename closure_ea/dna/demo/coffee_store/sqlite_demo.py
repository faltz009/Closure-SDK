from __future__ import annotations

import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

from closure_ea.dna.demo.coffee_store.common import (
    checkout_spec,
    decode_record,
    make_dataset,
    output_root,
)


DB_PATH = output_root() / "coffee_shop.sqlite3"


def build_database(path: Path = DB_PATH) -> Path:
    if path.exists():
        path.unlink()

    dataset = make_dataset()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id REAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            city TEXT NOT NULL,
            tier TEXT NOT NULL
        );
        CREATE TABLE products (
            product_id REAL PRIMARY KEY,
            sku TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_qty REAL NOT NULL
        );
        CREATE TABLE orders (
            order_id REAL PRIMARY KEY,
            customer_id REAL NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            total REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE order_items (
            item_id REAL PRIMARY KEY,
            order_id REAL NOT NULL,
            product_id REAL NOT NULL,
            qty REAL NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        CREATE TABLE payments (
            payment_id REAL PRIMARY KEY,
            order_id REAL NOT NULL,
            method TEXT NOT NULL,
            amount REAL NOT NULL,
            payment_status TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        CREATE INDEX idx_customers_city ON customers(city);
        CREATE INDEX idx_products_category ON products(category);
        CREATE INDEX idx_orders_status ON orders(status);
        CREATE INDEX idx_payments_method ON payments(method);
        """
    )
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", dataset["customers"])
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", dataset["products"])
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", dataset["orders"])
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", dataset["order_items"])
    conn.executemany("INSERT INTO payments VALUES (?,?,?,?,?)", dataset["payments"])
    conn.commit()
    conn.close()
    return path


def run_checkout(conn: sqlite3.Connection) -> dict:
    spec = checkout_spec()
    next_order_id = conn.execute("SELECT COALESCE(MAX(order_id), 0) + 1 FROM orders").fetchone()[0]
    next_payment_id = conn.execute("SELECT COALESCE(MAX(payment_id), 0) + 1 FROM payments").fetchone()[0]
    next_item_id = conn.execute("SELECT COALESCE(MAX(item_id), 0) + 1 FROM order_items").fetchone()[0]

    total = 0.0
    priced_items = []
    for item in spec["items"]:
        row = conn.execute(
            "SELECT product_id, sku, name, category, price, stock_qty FROM products WHERE product_id = ?",
            (item["product_id"],),
        ).fetchone()
        unit_price = float(row[4])
        total += unit_price * item["qty"]
        priced_items.append((row, unit_price, item["qty"]))

    conn.execute(
        "INSERT INTO orders VALUES (?,?,?,?,?)",
        (next_order_id, spec["customer_id"], spec["order_date"], spec["status"], round(total, 2)),
    )
    conn.execute(
        "INSERT INTO payments VALUES (?,?,?,?,?)",
        (next_payment_id, next_order_id, spec["payment_method"], round(total, 2), "captured"),
    )

    for row, unit_price, qty in priced_items:
        conn.execute(
            "INSERT INTO order_items VALUES (?,?,?,?,?)",
            (next_item_id, next_order_id, float(row[0]), qty, unit_price),
        )
        conn.execute(
            "UPDATE products SET stock_qty = ? WHERE product_id = ?",
            (float(row[5]) - qty, float(row[0])),
        )
        next_item_id += 1

    return {"order_id": float(next_order_id), "total": round(total, 2)}


def build_summary(path: Path = DB_PATH) -> dict:
    build_database(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    with conn:
        checkout = run_checkout(conn)

    counts = {}
    for table in ["customers", "order_items", "orders", "payments", "products"]:
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    tokyo_customers = [
        dict(row)
        for row in conn.execute(
            "SELECT customer_id, name, email, city, tier FROM customers WHERE city = 'Tokyo' ORDER BY customer_id LIMIT 5"
        ).fetchall()
    ]
    top_paid_orders = [
        dict(row)
        for row in conn.execute(
            "SELECT order_id, customer_id, order_date, status, total FROM orders WHERE status = 'paid' ORDER BY total DESC LIMIT 5"
        ).fetchall()
    ]
    tokyo_order_join = [
        {
            "orders.order_id": row["order_id"],
            "orders.customer_id": row["customer_id"],
            "orders.order_date": row["order_date"],
            "orders.status": row["status"],
            "orders.total": row["total"],
            "customers.customer_id": row["joined_customer_id"],
            "customers.name": row["name"],
            "customers.email": row["email"],
            "customers.city": row["city"],
            "customers.tier": row["tier"],
        }
        for row in conn.execute(
            """
            SELECT orders.order_id, orders.customer_id, orders.order_date, orders.status, orders.total,
                   customers.customer_id AS joined_customer_id, customers.name, customers.email, customers.city, customers.tier
            FROM orders JOIN customers ON orders.customer_id = customers.customer_id
            WHERE customers.city = 'Tokyo'
            LIMIT 5
            """
        ).fetchall()
    ]

    revenue = defaultdict(float)
    for city, total in conn.execute(
        "SELECT customers.city, orders.total FROM orders JOIN customers ON orders.customer_id = customers.customer_id"
    ).fetchall():
        revenue[city] += float(total)
    top_city, top_city_revenue = max(revenue.items(), key=lambda item: item[1])

    largest_product_sort_head = [
        row[0]
        for row in conn.execute("SELECT product_id FROM products ORDER BY price DESC LIMIT 5").fetchall()
    ]

    summary = {
        "tables": ["customers", "order_items", "orders", "payments", "products"],
        "counts": counts,
        "tokyo_customers": [decode_record(row) for row in tokyo_customers],
        "top_paid_orders": [decode_record(row) for row in top_paid_orders],
        "tokyo_order_join": [decode_record(row) for row in tokyo_order_join],
        "top_city_revenue": {top_city: round(top_city_revenue, 2)},
        "avg_order_total": round(conn.execute("SELECT AVG(total) FROM orders").fetchone()[0], 2),
        "largest_product_sort_head": largest_product_sort_head,
        "checkout": checkout,
    }
    conn.close()
    return summary


def main() -> int:
    summary = build_summary()
    print("SQLite coffee-store demo")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
