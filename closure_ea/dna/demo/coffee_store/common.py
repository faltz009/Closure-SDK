from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

CUSTOMERS_SCHEMA = [
    {"name": "customer_id", "type": "f64", "indexed": False, "primary": True},
    {"name": "name", "type": "bytes", "indexed": False},
    {"name": "email", "type": "bytes", "indexed": False},
    {"name": "city", "type": "bytes", "indexed": True},
    {"name": "tier", "type": "bytes", "indexed": True},
]

PRODUCTS_SCHEMA = [
    {"name": "product_id", "type": "f64", "indexed": False, "primary": True},
    {"name": "sku", "type": "bytes", "indexed": False},
    {"name": "name", "type": "bytes", "indexed": False},
    {"name": "category", "type": "bytes", "indexed": True},
    {"name": "price", "type": "f64", "indexed": False},
    {"name": "stock_qty", "type": "f64", "indexed": False},
]

ORDERS_SCHEMA = [
    {"name": "order_id", "type": "f64", "indexed": False, "primary": True},
    {"name": "customer_id", "type": "f64", "indexed": False, "references": "customers.customer_id"},
    {"name": "order_date", "type": "bytes", "indexed": False},
    {"name": "status", "type": "bytes", "indexed": True},
    {"name": "total", "type": "f64", "indexed": False},
]

ORDER_ITEMS_SCHEMA = [
    {"name": "item_id", "type": "f64", "indexed": False, "primary": True},
    {"name": "order_id", "type": "f64", "indexed": False, "references": "orders.order_id"},
    {"name": "product_id", "type": "f64", "indexed": False, "references": "products.product_id"},
    {"name": "qty", "type": "f64", "indexed": False},
    {"name": "unit_price", "type": "f64", "indexed": False},
]

PAYMENTS_SCHEMA = [
    {"name": "payment_id", "type": "f64", "indexed": False, "primary": True},
    {"name": "order_id", "type": "f64", "indexed": False, "references": "orders.order_id"},
    {"name": "method", "type": "bytes", "indexed": True},
    {"name": "amount", "type": "f64", "indexed": False},
    {"name": "payment_status", "type": "bytes", "indexed": True},
]

SCHEMAS = {
    "customers": CUSTOMERS_SCHEMA,
    "products": PRODUCTS_SCHEMA,
    "orders": ORDERS_SCHEMA,
    "order_items": ORDER_ITEMS_SCHEMA,
    "payments": PAYMENTS_SCHEMA,
}


@dataclass(frozen=True)
class DemoConfig:
    customers: int = 120
    products: int = 36
    orders: int = 280
    max_items_per_order: int = 4
    seed: int = 7


def demo_root() -> Path:
    return Path(__file__).resolve().parent


def output_root() -> Path:
    root = demo_root() / "output"
    root.mkdir(parents=True, exist_ok=True)
    return root


def encode(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return float(value)


def columnize(schema: list[dict], rows: list[list[object]]) -> list[list[object]]:
    columns: list[list[object]] = [[] for _ in schema]
    for row in rows:
        for i, value in enumerate(row):
            columns[i].append(encode(value))
    return columns


def decode_record(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, bytes):
            out[key] = value.decode("utf-8")
        else:
            out[key] = value
    return out


def make_dataset(config: DemoConfig | None = None) -> dict[str, list[list[object]]]:
    cfg = config or DemoConfig()
    rng = random.Random(cfg.seed)

    first = [
        "Ada", "Bruno", "Clara", "Diego", "Elena", "Farah", "Gabriel", "Hana",
        "Iris", "Joao", "Keiko", "Lina", "Mateo", "Nadia", "Omar", "Priya",
        "Quinn", "Ravi", "Sara", "Theo",
    ]
    last = [
        "Silva", "Tanaka", "Costa", "Almeida", "Singh", "Park", "Lopez", "Ibrahim",
        "Santos", "Moreau", "Kim", "Mendes",
    ]
    cities = ["Sao Paulo", "Tokyo", "Paris", "Lima", "Bogota", "Lisbon"]
    tiers = ["bronze", "silver", "gold"]

    categories = ["beans", "brew", "gear", "filters", "mugs", "tea"]
    bean_names = ["Cerrado", "Yirgacheffe", "Huehuetenango", "Nariño", "Sidamo", "Bourbon"]
    brew_names = ["V60", "Aeropress", "French Press", "Kalita", "Cold Brew"]
    gear_names = ["Kettle", "Grinder", "Scale", "Dripper", "Server"]
    filter_names = ["Paper Filter", "Metal Filter", "Cloth Filter"]
    mug_names = ["Travel Mug", "Ceramic Mug", "Glass Cup"]
    tea_names = ["Earl Grey", "Sencha", "Jasmine", "Chai"]
    payment_methods = ["card", "pix", "cash"]

    customers = []
    for i in range(cfg.customers):
        given = rng.choice(first)
        family = rng.choice(last)
        name = f"{given} {family}"
        city = rng.choice(cities)
        tier = tiers[min(rng.randrange(100) // 40, 2)]
        customers.append([
            float(i + 1),
            name,
            f"{given.lower()}.{family.lower()}{i}@example.com",
            city,
            tier,
        ])

    products = []
    product_names = {
        "beans": bean_names,
        "brew": brew_names,
        "gear": gear_names,
        "filters": filter_names,
        "mugs": mug_names,
        "tea": tea_names,
    }
    for i in range(cfg.products):
        category = categories[i % len(categories)]
        stem = product_names[category][i % len(product_names[category])]
        price = round(8.0 + (i % 7) * 3.5 + rng.random() * 4.0, 2)
        stock = float(30 + (i * 7) % 90)
        products.append([
            float(i + 1),
            f"SKU-{i+1:04d}",
            f"{stem} {category.title()}",
            category,
            price,
            stock,
        ])

    orders = []
    order_items = []
    payments = []
    item_id = 1
    for i in range(cfg.orders):
        customer_id = float(1 + rng.randrange(cfg.customers))
        item_count = 1 + rng.randrange(cfg.max_items_per_order)
        chosen = rng.sample(products, k=item_count)
        total = 0.0
        order_id = float(i + 1)
        for product in chosen:
            qty = float(1 + rng.randrange(3))
            unit_price = float(product[4])
            total += qty * unit_price
            order_items.append([
                float(item_id),
                order_id,
                float(product[0]),
                qty,
                unit_price,
            ])
            item_id += 1
        total = round(total, 2)
        status = "paid" if i % 7 != 0 else "pending"
        orders.append([
            order_id,
            customer_id,
            f"2026-03-{1 + (i % 28):02d}",
            status,
            total,
        ])
        payments.append([
            float(i + 1),
            order_id,
            rng.choice(payment_methods),
            total,
            "captured" if status == "paid" else "pending",
        ])

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
    }


def checkout_spec() -> dict:
    return {
        "customer_id": 8.0,
        "status": "paid",
        "order_date": "2026-03-30",
        "payment_method": "pix",
        "items": [
            {"product_id": 2.0, "qty": 2.0},
            {"product_id": 9.0, "qty": 1.0},
            {"product_id": 14.0, "qty": 1.0},
        ],
    }
