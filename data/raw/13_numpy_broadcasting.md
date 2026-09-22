# NumPy Broadcasting Rules and Vectorized Alignment

## Overview
Broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is "broadcast" across the larger array so that they have compatible shapes without unnecessary memory duplication.

## Key Concepts and Code Examples
NumPy compares shapes element-wise, starting with the trailing (rightmost) dimensions and working backwards. Two dimensions are compatible when:
1. They are equal, or
2. One of them is 1.

If these conditions are not met, a `ValueError: operands could not be broadcast together` is thrown.

```python
import numpy as np

# Matrix: shape (3, 4)
matrix = np.ones((3, 4))
# Vector: shape (4,) -> aligns with trailing dimension 4
row_vec = np.array([1, 2, 3, 4])
result = matrix + row_vec  # Shape (3, 4), row_vec added to each row

# Column-wise addition using np.newaxis or reshape:
col_vec = np.array([10, 20, 30])[:, np.newaxis] # Shape (3, 1)
result_col = matrix + col_vec # Adds 10 to row 0, 20 to row 1, 30 to row 2
```

Broadcasting allows computing pairwise Euclidean distances, normalization, and feature scaling without memory-expensive explicit loops.

## Common Mistake
**Broadcasting a 1D Array along the Wrong Axis:**
When trying to subtract the mean of each column or each row from a 2D array, broadcasting a 1D array of shape `(N,)` often broadcats across columns instead of rows (or vice-versa):
```python
# MISTAKE:
A = np.ones((5, 3))     # 5 samples, 3 features
feature_means = np.mean(A, axis=0) # Shape (3,)
# But if subtracting sample means (shape 5):
sample_means = np.mean(A, axis=1)  # Shape (5,)
# A - sample_means -> Crashes! (5, 3) and (5,) cannot broadcast because trailing dimensions 3 and 5 differ.

# CORRECT:
normalized = A - sample_means[:, np.newaxis] # Shape (5, 1) broadcasts correctly with (5, 3)
```
Always check array dimensions using `.shape` and reshape with `[:, np.newaxis]` to specify exact broadcast axes.
