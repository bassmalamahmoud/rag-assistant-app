# Pandas GroupBy: Split-Apply-Combine Operations

## Overview
The `groupby` mechanism implements the classic **Split-Apply-Combine** pattern. It splits data into groups based on key columns, applies a transformation or aggregation function to each group independently, and combines the results into a new data structure.

## Key Concepts and Code Examples
Pandas provides `.agg()`, `.transform()`, and `.filter()` methods on GroupBy objects:

```python
import pandas as pd

df = pd.DataFrame({
    "store": ["North", "North", "South", "South", "East"],
    "product": ["A", "B", "A", "B", "A"],
    "revenue": [100, 150, 80, 200, 120]
})

# Aggregation: reduces each group to a single value
store_summary = df.groupby("store")["revenue"].agg(["sum", "mean", "count"])

# Transform: returns an aligned Series matching the original DataFrame's length
# (e.g. for calculating percentage of group total or group mean normalization)
group_mean = df.groupby("store")["revenue"].transform("mean")
df["revenue_diff_from_store_mean"] = df["revenue"] - group_mean

# Filter: drops groups that do not satisfy a group-level boolean predicate
high_volume_stores = df.groupby("store").filter(lambda g: g["revenue"].sum() > 200)
```

## Common Mistake
**Iterating Over Groups with `for name, group in df.groupby(...)` for Calculations:**
Writing manual loops over groupby objects in Python is orders of magnitude slower than using vectorized `.agg()`, `.transform()`, or built-in methods:
```python
# MISTAKE:
# means = {}
# for store, group in df.groupby("store"):
#     means[store] = group["revenue"].mean() # Very slow on large datasets!

# CORRECT:
means = df.groupby("store")["revenue"].mean() # Highly optimized in C
```
Leverage vectorized `.agg()` or `.transform()` instead of manual Python loops.
