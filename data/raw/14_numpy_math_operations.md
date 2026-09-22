# NumPy Vectorized Mathematical Operations and Ufuncs

## Overview
Universal functions (`ufuncs`) are functions that operate element-by-element on ndarrays. By executing operations in compiled C code, ufuncs eliminate the overhead of Python bytecode interpretation, achieving speedups of 50x to 100x over standard Python loops.

## Key Concepts and Code Examples
NumPy provides vectorized implementations for arithmetic, trigonometric, exponential, and reduction operations:

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])

# Element-wise operations
y = np.exp(x)
log_y = np.log(y)
sig = 1 / (1 + np.exp(-x))  # Vectorized sigmoid activation function

# Reductions across specific axes
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(matrix, axis=0))  # Sum across columns -> [5, 7, 9]
print(np.mean(matrix, axis=1)) # Mean across rows -> [2.0, 5.0]
print(np.std(matrix))          # Standard deviation of entire matrix
```

Common aggregation functions include `np.sum()`, `np.mean()`, `np.std()`, `np.argmax()`, and `np.cumsum()`.

## Common Mistake
**Using Python's Built-In `sum()` or `math.sqrt()` on NumPy Arrays:**
A frequent performance mistake is passing NumPy arrays into Python's built-in `sum()` or using `math.sqrt()` instead of `np.sum()` and `np.sqrt()`:
```python
# MISTAKE:
import math
arr = np.random.rand(1_000_000)
# total = sum(arr)        # Slow: unpacks NumPy floats into Python objects
# roots = [math.sqrt(v) for v in arr] # Extremely slow!

# CORRECT:
total = np.sum(arr)       # Fast: executed in compiled C
roots = np.sqrt(arr)      # Fast: vectorized ufunc
```
Always use NumPy's built-in ufuncs (`np.sum`, `np.mean`, `np.sqrt`) when operating on arrays.
