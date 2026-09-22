# Pandas Data Cleaning: String Accessor, Datetime Parsing, and Deduplication

## Overview
Real-world machine learning data is messy: inconsistent string formatting, erratic date stamps, and duplicate rows. Pandas provides the `.str` accessor for vectorized string manipulations, `.dt` accessor for timestamp extraction, and `.drop_duplicates()` for deduplication.

## Key Concepts and Code Examples
Vectorized data cleaning operations avoid slow Python string iterations:

```python
import pandas as pd

df = pd.DataFrame({
    "raw_text": ["  Standard MODEL  ", "advanced-model", "STANDARD model  "],
    "date_str": ["2026-03-15 14:30:00", "2026-03-16 09:15:00", "2026-03-15 14:30:00"],
    "user_id": [101, 102, 101]
})

# 1. Vectorized string cleaning using .str accessor
df["clean_model"] = df["raw_text"].str.strip().str.lower().str.replace("-", "_")

# 2. Datetime parsing and feature extraction using .dt accessor
df["datetime"] = pd.to_datetime(df["date_str"])
df["year"] = df["datetime"].dt.year
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()

# 3. Removing duplicate records
df_unique = df.drop_duplicates(subset=["clean_model", "user_id"], keep="first")
```

## Common Mistake
**Using Standard Python String Methods Instead of the `.str` Accessor:**
A common mistake is attempting to call Python string functions directly on a Series, which raises `AttributeError: 'Series' object has no attribute 'lower'`:
```python
# MISTAKE:
# df["clean"] = df["raw_text"].strip().lower() # AttributeError!

# CORRECT:
df["clean"] = df["raw_text"].str.strip().str.lower()
```
Always prefix string operations on Pandas Series with the `.str` accessor (e.g., `df['col'].str.replace(...)`).
