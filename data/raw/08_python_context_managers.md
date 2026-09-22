# Python Context Managers and Resource Management

## Overview
Context managers manage the acquisition and safe release of resources, such as file handles, database connections, GPU memory locks, and temporary directories. They ensure that cleanup logic runs reliably even when unhandled exceptions occur during execution.

## Key Concepts and Code Examples
The `with` statement implements the context management protocol via `__enter__()` and `__exit__()` methods:

```python
class TimerContext:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.perf_counter() - self.start
        print(f"Task '{self.task_name}' finished in {duration:.4f}s")
        # Return False to let any exception propagate upwards
        return False

with TimerContext("Inference Pipeline"):
    # Run heavy prediction job
    sum(range(100_000))
```

Python's `contextlib` module provides a generator-based shortcut `@contextmanager`:
```python
from contextlib import contextmanager

@contextmanager
def temporary_learning_rate(optimizer, temp_lr):
    old_lr = optimizer['lr']
    optimizer['lr'] = temp_lr
    try:
        yield optimizer
    finally:
        optimizer['lr'] = old_lr
```

## Common Mistake
**Suppressing Exceptions Accidentally in `__exit__`:**
In custom context managers, returning a truthy value (such as `True`) from the `__exit__` method causes Python to silently suppress any exception that occurred inside the `with` block:
```python
# MISTAKE:
def __exit__(self, exc_type, exc_val, exc_tb):
    clean_up()
    return True # Accidental suppression! The caller will never know an error occurred!
```
Unless you explicitly intend to swallow the exception, return `False` or `None` from `__exit__`.
