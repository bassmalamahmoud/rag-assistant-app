# Pandas Data Structures: Series, DataFrame, and Index Alignment

## Overview
Pandas is the primary data manipulation and analysis library in Python. Its two core data structures are the 1D `Series` (a labeled, typed array) and the 2D `DataFrame` (a tabular structure of rows and columns). A defining superpower of Pandas is automatic **Index Alignment**, where operations align on labels rather than positions.

## Key Concepts and Code Examples
Every Series and DataFrame has an explicit index:

```python
import pandas as pd

# Creating a Series with custom index
s1 = pd.Series([10, 20, 30], index=["a", "b", "c"])
s2 = pd.Series([1, 2, 3], index=["b", "c", "d"])

# Automatic index alignment during arithmetic:
result = s1 + s2
# Result:
# a    NaN  (only in s1)
# b   21.0  (10 + 1)
# c   32.0  (20 + 2)
# d    NaN  (only in s2)

# Creating a DataFrame
data = {
    "feature_a": [1.2, 3.4, 5.6],
    "category": ["A", "B", "A"],
    "target": [0, 1, 0]
}
df = pd.DataFrame(data, index=["obs_1", "obs_2", "obs_3"])
print(df.dtypes)
print(df.shape)
```

## Common Mistake
**Ignoring Automatic Index Alignment when Adding New Columns:**
A common pitfall occurs when assigning an existing Series with a different or scrambled index into a DataFrame:
```python
# MISTAKE:
df = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
new_series = pd.Series([100, 200, 300], index=[2, 1, 0]) # Inverted index!

df["new_col"] = new_series
# Result: df['new_col'] aligns by label (0 gets 300, 1 gets 200, 2 gets 100),
# NOT by positional order!

# If position-based assignment is intended, pass the raw NumPy values:
df["new_col"] = new_series.to_numpy()
```
Always be mindful that Pandas aligns by index labels, not positional sequence.
