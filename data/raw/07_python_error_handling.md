# Robust Python Error Handling and Custom Exceptions

## Overview
Data pipelines, model training jobs, and deployment APIs must handle unexpected data anomalies gracefully. Robust error handling using `try`, `except`, `else`, and `finally` blocks prevents silent pipeline failures, corrupted training checkpoints, and uninformative server crashes.

## Key Concepts and Code Examples
Python organizes exceptions in a class hierarchy under `BaseException` and `Exception`. Production code should catch specific exceptions and define custom domain-specific errors.

```python
class DataValidationError(Exception):
    """Raised when an incoming dataset violates required schema constraints."""
    pass

def load_and_validate_features(filepath: str, required_cols: list[str]):
    try:
        with open(filepath, "r") as f:
            header = f.readline().strip().split(",")
    except FileNotFoundError:
        raise DataValidationError(f"File {filepath} does not exist.")
    except PermissionError:
        raise DataValidationError(f"Cannot read {filepath}: permission denied.")
    else:
        missing = [col for col in required_cols if col not in header]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        return header
    finally:
        print("Completed feature validation check.")
```

The `else` block executes solely when no exception occurred in the `try` block, keeping the `try` block minimal.

## Common Mistake
**Catching Bare `except:` or Suppressing Errors:**
The most harmful mistake in ML error handling is using a bare `except:` or catching generic `Exception` and doing nothing:
```python
# MISTAKE:
try:
    load_weights()
except:
    pass # Catches KeyboardInterrupt, SystemExit, and masks fatal syntax errors!
```
Bare `except:` catches process interruption signals like `KeyboardInterrupt` (Ctrl+C), preventing you from terminating a runaway script. Furthermore, suppressing errors silently causes models to train on uninitialized or garbage data. Always catch specific exceptions (e.g., `except (ValueError, KeyError) as e:`).
