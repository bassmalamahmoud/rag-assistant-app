# Python Decorators and Function Wrapping

## Overview
Decorators are callables that accept a function as input and return a modified or wrapped function. In machine learning infrastructure, decorators are indispensable for logging execution time, caching expensive dataset transformations, validating tensor input shapes, and handling API rate limits.

## Key Concepts and Code Examples
Under the hood, `@my_decorator` syntax is syntactic sugar for `func = my_decorator(func)`. Using `functools.wraps` is essential to preserve original docstrings, parameter signatures, and function names.

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[TIMING] {func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def train_step(batch_x, batch_y):
    # Simulated computation
    time.sleep(0.01)
    return "weights_updated"
```

Parameterized decorators take arguments by wrapping an extra outer factory function:
```python
def retry(attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1: raise e
        return wrapper
    return decorator
```

## Common Mistake
**Omitting `functools.wraps`:**
A common mistake when writing decorators is omitting `@wraps(func)` on the inner wrapper function. When omitted, the original function loses its identity: its `__name__` becomes `"wrapper"`, and `__doc__` is erased. This breaks introspection, documentation generators, and debugging tools (such as FastAPI route inspectors or pytest fixtures). Always decorate inner wrappers with `@functools.wraps(func)`.
