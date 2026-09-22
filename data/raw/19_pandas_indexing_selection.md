# Pandas Indexing and Selection: `loc` vs `iloc` and Chained Assignment

## Overview
Selecting rows and columns correctly in Pandas is essential. Pandas provides two distinct indexing accessors:
- `.loc[]`: Label-based indexing (uses index names and column names).
- `.iloc[]`: Integer position-based indexing (uses 0-indexed integer coordinates).

## Key Concepts and Code Examples
Using explicit accessors prevents ambiguity, especially when indices are integers:

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 30, 35, 40],
    "salary": [50000, 65000, 80000, 110000]
}, index=[10, 20, 30, 40]) # Non-sequential integer index

# Label-based selection with .loc
row_30 = df.loc[30, "salary"] # Finds row with index label 30 -> 80000

# Position-based selection with .iloc
row_pos2 = df.iloc[2, 1]      # Finds 3rd row (pos 2), 2nd col (pos 1) -> 80000

# Slicing with .loc includes both start AND end labels:
print(df.loc[10:30]) # Includes rows labeled 10, 20, AND 30!

# Slicing with .iloc excludes end position (standard Python slice):
print(df.iloc[0:2])  # Includes rows at indices 0 and 1 only
```

## Common Mistake
**The `SettingWithCopyWarning` and Chained Assignment:**
The most frequent error in Pandas is chained indexing assignment:
```python
# MISTAKE:
df[df["age"] > 30]["salary"] = 90000 # Triggers SettingWithCopyWarning!
# May not modify the original DataFrame because df[df['age'] > 30] returned a temporary copy!

# CORRECT:
df.loc[df["age"] > 30, "salary"] = 90000
```
Always use single-bracket `.loc[row_mask, col_selector] = value` when modifying subsets of a DataFrame.
