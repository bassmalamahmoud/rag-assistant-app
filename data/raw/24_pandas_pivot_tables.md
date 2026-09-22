# Pandas Reshaping: Pivot, Pivot Tables, and Melt

## Overview
Reshaping data between wide format and long (tidy) format is essential for exploratory data analysis, time-series feature engineering, and visualization. Pandas provides `pivot()` for reshaping without aggregation, `pivot_table()` for aggregation over multiple dimensions, and `melt()` for unpivoting wide tables into long format.

## Key Concepts and Code Examples
Understanding the distinction between `pivot` and `pivot_table`:

```python
import pandas as pd

df = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
    "city": ["London", "Paris", "London", "Paris"],
    "temperature": [5, 7, 4, 8],
    "humidity": [80, 75, 85, 70]
})

# Pivot table: Aggregates multiple entries using aggfunc (default: mean)
pivot_df = df.pivot_table(
    index="date",
    columns="city",
    values="temperature",
    aggfunc="mean"
)

# Unpivoting using melt (wide to long format for ML pipelines)
melted = pd.melt(
    df,
    id_vars=["date", "city"],
    value_vars=["temperature", "humidity"],
    var_name="metric",
    value_name="value"
)
```

## Common Mistake
**Calling `df.pivot()` on Data with Duplicate Index/Column Combinations:**
Calling `df.pivot()` when the dataset contains duplicate entries for the chosen `index` and `columns` raises `ValueError: Index contains duplicate entries, cannot reshape`.
```python
# MISTAKE:
# df.pivot(index="date", columns="city", values="temperature") # Crashes if (date, city) repeats!

# CORRECT:
# Use pivot_table with an explicit aggregation function (e.g. mean or sum):
df.pivot_table(index="date", columns="city", values="temperature", aggfunc="mean")
```
Use `pivot_table()` whenever duplicate index/column pairs may exist in your dataset.
