# Coffee Store Demo

Same coffee shop. Two databases. One does something the other can't.

Both scripts create the same tables (customers, products, orders,
order_items, payments), seed the same fake data, run a checkout
transaction, and answer the same business questions:

- Which customers are in Tokyo?
- What are the top paid orders?
- What's the average order total?
- Sort products by price.
- Join orders with customers.

Then Closure DNA shows what it can do on top:

- checkout as one atomic transaction
- reject an order that points at a missing customer
- delete a cancelled order, then compact the table to remove its tombstone

## "Find me orders similar to this one"

After the checkout, we take the new order and ask the database:
which existing orders look most like this one?

This isn't a WHERE clause. There's no rule to write. The database
already knows what "similar" means — records with similar fields
land near each other geometrically. The closest match comes back
with a number (drift) that tells you how close, and a decomposition
that tells you what's different.

Think about what this does for a real business:

- **Fraud detection**: "this transaction looks unlike anything
  we've seen" = high drift from all neighbors.
- **Customer matching**: "is this the same customer from our
  other system, with a slightly different name?" = low drift.
- **Recommendations**: "customers who ordered something similar
  to you also ordered..." = nearest neighbors in order space.
- **Deduplication**: "these two records are probably the same
  thing" = drift near zero but not exactly zero.

No ML model. No training data. No feature engineering. Just store
your data and ask.

## "Is my data still intact?"

One number. 14 microseconds. At a million records. Every time.

The table's fingerprint is 32 bytes. If one byte of one record
changes, the fingerprint changes. You don't scan the table. You
don't run a checksum. You read one number and you know.

## Run

```bash
python3 closure_ea/dna/demo/coffee_store/sqlite_demo.py
python3 closure_ea/dna/demo/coffee_store/closure_dna_demo.py
```

Compare the outputs. Everything above the line is the same.
Everything below is what only Closure DNA can do, plus the product
basics a real database has to get right: commit, constraints, and cleanup.
