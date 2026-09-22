# Pandas Merging, Joining, and Concatenation

## Overview
Data science projects typically require combining data across multiple tables (e.g. joining customer demographic tables with transaction logs). Pandas provides `pd.merge()` (relational database-style joins), `pd.concat()` (stacking tables along an axis), and `DataFrame.join()` (joining on indices).

## Key Concepts and Code Examples
Understanding join types (`inner`, `left`, `right`, `outer`) ensures that no critical observations are unintentionally lost:

```python
import pandas as pd

customers = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "name": ["Alice", "Bob", "Charlie"]
})

orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4],
    "customer_id": [101, 102, 101, 104],
    "amount": [250, 120, 300, 75]
})

# Relational inner merge
merged_inner = pd.merge(customers, orders, on="customer_id", how="inner")

# Left join preserving all customers and showing merge status with indicator=True
merged_left = pd.merge(customers, orders, on="customer_id", how="left", indicator=True)

# Concatenating / Stacking rows
batch1 = pd.DataFrame({"feat": [1, 2]})
batch2 = pd.DataFrame({"feat": [3, 4]})
stacked = pd.concat([batch1, batch2], ignore_index=True)
```

## Common Mistake
**Silent Cartesian Explosion on Duplicate Merge Keys:**
If both the left and right DataFrames contain duplicate values in the join key, `pd.merge()` performs a many-to-many Cartesian product, dramatically inflating row counts and memory usage without raising an error:
```python
# MISTAKE:
# If left table has 1,000 rows with customer_id=101, and right table has 1,000 rows with customer_id=101,
# the merged result produces 1,000,000 rows!

# PREVENTION:
# Use validate parameter to enforce one-to-one, one-to-many, or many-to-one constraints:
pd.merge(customers, orders, on="customer_id", how="left", validate="1:m")
```
Always use `validate="1:m"`, `validate="m:1"`, or `validate="1:1"` in `pd.merge()` to catch unexpected duplicates.
