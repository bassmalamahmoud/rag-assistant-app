# Handling Missing Data in Pandas: Detection, Imputation, and Nullable Dtypes

## Overview
Missing values (`NaN`, `None`, `<NA>`) are commonplace in observational and machine learning datasets. Machine learning estimators in scikit-learn generally crash if input feature matrices contain missing values. Proper detection, deletion, or imputation is a mandatory preprocessing phase.

## Key Concepts and Code Examples
Pandas represents missing values using `np.nan` (for floats) and the modern pandas `<NA>` singleton (`pd.NA`) for nullable integer, boolean, and string dtypes:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age": [25, np.nan, 30, np.nan],
    "income": [50000, 60000, np.nan, 80000],
    "city": ["Cairo", "Alexandria", None, "Cairo"]
})

# Detecting missing values
print(df.isna().sum()) # Total missing values per feature

# Dropping rows or columns with missing values
clean_rows = df.dropna(subset=["age", "income"])

# Imputing missing values with summary statistics
mean_age = df["age"].mean()
df["age_imputed"] = df["age"].fillna(mean_age)

# Forward-fill / Backward-fill for time-series data
df["income_ffill"] = df["income"].ffill()
```

## Common Mistake
**Checking for NaN with Direct Equality `x == np.nan`:**
A notorious mistake is attempting to filter or test missing values using `df["age"] == np.nan`. In accordance with the IEEE 754 floating-point specification, `np.nan == np.nan` evaluates to `False`!
```python
# MISTAKE:
nan_count = len(df[df["age"] == np.nan]) # Always returns 0!

# CORRECT:
nan_count = df["age"].isna().sum()
# Or in pure python / numpy:
# np.isnan(val)
```
Always use `.isna()` or `.notna()` to identify missing values in Pandas.
