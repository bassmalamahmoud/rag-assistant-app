# Python Variables, Dynamic Typing, and Memory References

## Overview
In Python, variables are not memory containers that hold values directly; rather, they are symbolic names that act as references (pointers) to objects stored in memory. Every object in Python has a unique identity, a type, and a value. Understanding object mutability and memory referencing is critical in Machine Learning workflows where passing large datasets or model weights between functions can inadvertently modify in-place state.

## Key Concepts and Code Examples
Python datatypes are categorized into immutable types (integers, floats, strings, tuples, frozensets) and mutable types (lists, dictionaries, sets, NumPy ndarrays).

When assigning one variable to another:
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)  # Output: [1, 2, 3, 4] -> Both a and b point to the same memory object!
```

To create an independent duplicate:
```python
import copy

original_matrix = [[1, 2], [3, 4]]
shallow_copy = original_matrix.copy() # Nested lists still share references
deep_copy = copy.deepcopy(original_matrix) # Fully recursive independent copies
```

In numerical pipelines and preprocessing, failing to recognize mutable re-assignments causes silent data leakage across train and validation splits.

## Common Mistake
**The Mutable In-Place Mutation Pitfall:**
A common mistake occurs when passing a feature list or dictionary configuration into a data preprocessing function and modifying it directly instead of creating a copy. For example:
```python
def normalize_features(data_list):
    # Bug: Modifies caller's data directly
    for i in range(len(data_list)):
        data_list[i] = data_list[i] / 100.0
    return data_list
```
Because lists are mutable references, `data_list` mutates the original data in the calling scope. The correct approach is either returning a new list comprehension or using `copy.deepcopy()` before transformation.
