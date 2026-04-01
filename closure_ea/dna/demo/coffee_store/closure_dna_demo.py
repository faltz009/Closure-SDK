from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from closure_ea.dna import Database

from closure_ea.dna.demo.coffee_store.common import (
    SCHEMAS,
    checkout_spec,
    columnize,
    decode_record,
    make_dataset,
    output_root,
)


DB_PATH = output_root() / "coffee_shop.cdb"


def build_database(path: Path = DB_PATH) -> Path:
    if path.exists():
        shutil.rmtree(path)

    dataset = make_dataset()
    with Database.create(path) as db:
        for table_name, schema in SCHEMAS.items():
            table = db.create_table(table_name, schema)
            table.insert_columns(columnize(schema, dataset[table_name]))
    return path


def run_checkout(db: Database) -> dict:
    spec = checkout_spec()
    products = db.table("products")
    orders = db.table("orders")
    order_items = db.table("order_items")
    payments = db.table("payments")

    next_order_id = float(orders.count() + 1)
    next_payment_id = float(payments.count() + 1)
    next_item_id = float(order_items.count() + 1)

    total = 0.0
    priced_items = []
    for item in spec["items"]:
        product_row = products.get_row(int(item["product_id"] - 1))
        unit_price = float(product_row[4])
        total += unit_price * item["qty"]
        priced_items.append((product_row, unit_price, item["qty"]))

    orders.insert([next_order_id, spec["customer_id"], spec["order_date"].encode("utf-8"), spec["status"].encode("utf-8"), round(total, 2)])
    payments.insert([next_payment_id, next_order_id, spec["payment_method"].encode("utf-8"), round(total, 2), b"captured"])

    for product_row, unit_price, qty in priced_items:
        order_items.insert([next_item_id, next_order_id, float(product_row[0]), qty, unit_price])
        products.update(int(product_row[0] - 1), [
            float(product_row[0]),
            product_row[1],
            product_row[2],
            product_row[3],
            float(product_row[4]),
            float(product_row[5]) - qty,
        ])
        next_item_id += 1

    return {"order_id": next_order_id, "total": round(total, 2)}


def build_summary(path: Path = DB_PATH) -> dict:
    build_database(path)

    with Database.open(path) as db:
        with db.transaction() as tx:
            checkout = run_checkout(tx)

        foreign_key_rejected = False
        foreign_key_message = ""
        try:
            db.table("orders").insert([9999.0, 999999.0, b"2026-03-31", b"paid", 10.0])
        except ValueError as exc:
            foreign_key_rejected = True
            foreign_key_message = str(exc)

        draft_order_id = float(db.table("orders").count() + 1)
        db.table("orders").insert([draft_order_id, 1.0, b"2026-03-31", b"cancelled", 0.0])
        db.table("orders").delete(int(draft_order_id - 1))
        compact_removed = db.compact("orders")

        customers = db.table("customers")
        products = db.table("products")
        orders = db.table("orders")

        tokyo_customers = db.select("customers", where=[("city", "=", "Tokyo")], order_by="customer_id", limit=5)
        paid_orders = db.select("orders", where=[("status", "=", "paid")], order_by="total", descending=True, limit=5)
        joined = db.join("orders", "customers", "customer_id", where=[("customers.city", "=", b"Tokyo")], limit=5)
        top_products = db.select("products", order_by="price", descending=True, limit=5)

        by_city = defaultdict(float)
        for record in db.join("orders", "customers", "customer_id"):
            by_city[record["customers.city"]] += float(record["orders.total"])
        top_city, top_city_revenue = max(by_city.items(), key=lambda item: item[1])

        orders.build_genome()
        exact_query = orders.get_row(int(checkout["order_id"] - 1))
        hits = orders.search(exact_query, k=3)
        similar_orders = []
        for hit in hits:
            row = orders.get_row(hit.position)
            similar_orders.append(
                {
                    "order_id": float(row[0]),
                    "customer_id": float(row[1]),
                    "status": row[3].decode("utf-8"),
                    "total": float(row[4]),
                    "drift": round(hit.drift, 9),
                    "phase": round(hit.phase, 9),
                }
            )

        summary = {
            "tables": db.tables(),
            "counts": {name: db.table(name).count() for name in db.tables()},
            "tokyo_customers": [decode_record(row) for row in tokyo_customers],
            "top_paid_orders": [decode_record(row) for row in paid_orders],
            "tokyo_order_join": [decode_record(row) for row in joined],
            "top_city_revenue": {top_city.decode("utf-8"): round(top_city_revenue, 2)},
            "avg_order_total": round(orders.avg("total"), 2),
            "largest_product_sort_head": [row["product_id"] for row in top_products],
            "checkout": checkout,
            "transaction_demo": {
                "checkout_committed": True,
                "how_it_works": "checkout runs in one staged transaction, then swaps into place atomically",
            },
            "foreign_key_demo": {
                "rejected_missing_customer": foreign_key_rejected,
                "message": foreign_key_message,
            },
            "compaction_demo": compact_removed,
            "identity_norm_hint": round(float(sum(v * v for v in orders.identity())), 6),
            "table_check": round(orders.check(), 9),
            "table_hopf": db.table("orders").check_hopf(),
            "genome": {
                "depth": orders.genome_depth(),
                "codons": orders.genome_codons(),
            },
            "resonance_query": {
                "order_id": checkout["order_id"],
                "what_it_means": "Find the closest existing order patterns to the new checkout row.",
            },
            "search_hits": [
                {
                    "position": hit.position,
                    "drift": round(hit.drift, 9),
                    "phase": round(hit.phase, 9),
                }
                for hit in hits
            ],
            "similar_orders": similar_orders,
        }
    return summary


def main() -> int:
    summary = build_summary()
    print("Closure DNA coffee-store demo")
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0



def _json_default(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
