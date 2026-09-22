# Python File I/O: Handling Text, JSON, and CSV

## Overview
Data ingestion and serialization are the foundation of ML pipelines. Datasets, hyperparameters, model weights, and inference logs are routinely read from and written to CSV, JSON, and text files. Proper encoding handling, resource buffering, and serialization integrity prevent data corruption.

## Key Concepts and Code Examples
Always use the `with open(...)` construct with explicit character encoding (`encoding="utf-8"`) to avoid platform-dependent encoding bugs.

```python
import json
import csv

# Writing and reading JSON configuration
config = {"model": "RandomForest", "n_estimators": 100, "random_state": 42}
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

with open("config.json", "r", encoding="utf-8") as f:
    loaded_config = json.load(f)

# Reading CSV files using standard csv module
with open("dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Process each row as a dictionary
        pass
```

For large files, read lazily line-by-line rather than reading the entire content into memory with `f.read()`.

## Common Mistake
**Omitting Explicit Encoding and Forgetting `newline=''` in CSV on Windows:**
On Windows, opening files without `encoding="utf-8"` defaults to legacy codepages (e.g., `cp1252`), which crashes with `UnicodeDecodeError` whenever text contains special symbols, emojis, or foreign characters.
Additionally, when writing CSV files on Windows without `newline=''`:
```python
# MISTAKE:
with open("data.csv", "w") as f: # Causes blank lines between rows on Windows!
    writer = csv.writer(f)
    writer.writerow(["id", "score"])

# CORRECT:
with open("data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "score"])
```
Always specify `encoding="utf-8"` and `newline=""` for CSV file writers.
