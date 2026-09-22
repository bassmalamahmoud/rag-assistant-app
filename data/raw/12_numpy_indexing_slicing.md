# NumPy Indexing, Slicing, and Views vs Copies

## Overview
Indexing and slicing allow you to extract subsets of multi-dimensional arrays. A deep architectural feature of NumPy is the distinction between a **View** (which shares memory with the original array) and a **Copy** (which allocates separate memory). Understanding this distinction is vital to prevent accidental data corruption.

## Key Concepts and Code Examples
Basic slicing using `:` creates a **view**, meaning modifications to the slice alter the original parent array:

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
slice_view = arr[1:4] # Elements [20, 30, 40]
slice_view[0] = 999
print(arr) # Output: [10, 999, 30, 40, 50] -> Parent array is modified!
```

To create an independent array, call `.copy()` explicitly:
```python
safe_copy = arr[1:4].copy()
safe_copy[0] = 777
print(arr) # Parent array remains unchanged
```

Conversely, **Fancy Indexing** (indexing with lists or boolean masks) always generates a **copy**, not a view:
```python
# Boolean mask indexing (generates a copy)
data = np.array([1, -2, 3, -4, 5])
positives = data[data > 0] # Copy of positive numbers
```

## Common Mistake
**Assuming Array Slicing Creates an Independent Copy:**
In standard Python lists, `list[1:3]` creates a shallow copy. Programmers switching to NumPy often mistakenly assume that `arr[1:3]` also creates an independent copy. Mutating `arr[1:3]` silently alters the source dataset in memory!
```python
# MISTAKE:
train_features = dataset[:, :-1]
train_features += 1.0 # Modifies dataset itself in-place!

# CORRECT:
train_features = dataset[:, :-1].copy()
```
Whenever a sliced subset needs independent transformation, always append `.copy()`.
