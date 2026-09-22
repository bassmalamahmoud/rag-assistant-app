# NumPy Linear Algebra: Matrix Multiplication, Inverses, and Decompositions

## Overview
Linear algebra is the mathematical engine behind linear regression, PCA (Principal Component Analysis), SVMs, and neural networks. NumPy provides robust, optimized BLAS and LAPACK routines under the `np.linalg` module for matrix operations.

## Key Concepts and Code Examples
In NumPy, `*` denotes element-wise multiplication, whereas `@` (or `np.matmul()`) denotes true matrix multiplication:

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
elem_wise = A * B # [[5, 12], [21, 32]]

# Matrix dot product (Linear Algebra)
dot_product = A @ B # [[19, 22], [43, 50]]

# Inverting a matrix and solving linear systems Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b) # Preferred over np.linalg.inv(A) @ b for numerical stability

# Eigenvalues and Eigenvectors (used in PCA)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)
```

## Common Mistake
**Confusing `*` with `@` (or `np.dot`):**
The most frequent mistake in machine learning implementations is using the `*` operator when matrix multiplication is intended:
```python
# MISTAKE:
weights = np.random.randn(4, 2)
inputs = np.random.randn(4, 2)
# predictions = inputs * weights # Element-wise! Not computing inner dot products!

# CORRECT:
inputs = np.random.randn(10, 4)
weights = np.random.randn(4, 2)
predictions = inputs @ weights # Correct shape (10, 2)
```
Always use `@` for matrix multiplication and verify that inner dimensions match.
