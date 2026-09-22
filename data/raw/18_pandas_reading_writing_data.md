# Pandas Data Ingestion: Efficient CSV, Parquet, and Chunking

## Overview
Real-world machine learning workflows begin with ingesting tabular data from CSV, JSON, Parquet, or SQL. `pd.read_csv()` provides dozens of parameters to control parsing, datetime conversions, chunking, and memory footprint.

## Key Concepts and Code Examples
Optimizing ingestion speed and memory usage is vital when handling datasets larger than available RAM:

```python
import pandas as pd

# Reading CSV with explicit types and datetime parsing
df = pd.read_csv(
    "dataset.csv",
    usecols=["user_id", "timestamp", "transaction_amount", "label"],
    dtype={"user_id": "int32", "transaction_amount": "float32", "label": "int8"},
    parse_dates=["timestamp"]
)

# Reading massive files in chunks to avoid Out-Of-Memory (OOM) crashes:
chunk_size = 50_000
total_records = 0
for chunk in pd.read_csv("huge_file.csv", chunksize=chunk_size):
    # Process chunk (e.g. filter or aggregate)
    filtered = chunk[chunk["transaction_amount"] > 100]
    total_records += len(filtered)

# Writing to high-performance Parquet format (preserves dtypes & compresses)
df.to_parquet("dataset.parquet", engine="pyarrow", compression="snappy")
```

Parquet files load up to 10x faster than raw CSVs and retain column datatypes intact.

## Common Mistake
**Loading Entire Massive Files into RAM Without `usecols` or `chunksize`:**
Attempting to load a multi-gigabyte CSV using default `pd.read_csv("data.csv")` leads to high memory consumption and browser/kernel termination (`Out of Memory`).
Always specify `usecols` to load only the required features, define efficient `dtypes` (e.g., `float32` instead of `float64`), or iterate using `chunksize`.
