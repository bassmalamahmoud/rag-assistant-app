# List, Dict, and Set Comprehensions in Python

## Overview
Comprehensions provide a concise, readable, and computationally efficient syntax for transforming and filtering iterables. In data science, comprehensions are widely used for string cleaning, tokenization, batching indices, and parsing raw metadata before loading arrays into NumPy or Pandas.

## Key Concepts and Code Examples
Comprehensions replace verbose `for` loops and `append()` statements. Because they are implemented in C-level bytecode in CPython, they execute noticeably faster than traditional loops.

```python
# List comprehension with filtering condition
raw_tokens = ["  ML  ", "AI", "", " deep learning ", None]
clean_tokens = [t.strip().lower() for t in raw_tokens if t and isinstance(t, str)]
# Result: ['ml', 'ai', 'deep learning']

# Dictionary comprehension
features = ["age", "income", "credit_score"]
feature_map = {feat: idx for idx, feat in enumerate(features)}
# Result: {'age': 0, 'income': 1, 'credit_score': 2}

# Generator expression (memory efficient for massive streams)
sum_squares = sum(x * x for x in range(1_000_000))
```

## Common Mistake
**Over-Nesting and High Memory Consumption:**
A frequent mistake is using list comprehensions when a generator expression is appropriate, or writing deeply nested comprehensions (more than 2 levels) that consume massive RAM and degrade readability:
```python
# MISTAKE: Storing millions of elements in RAM when only iterating once
total = sum([x ** 2 for x in range(10_000_000)]) # Allocates full list in memory!

# CORRECT:
total = sum(x ** 2 for x in range(10_000_000))   # Computes values lazily on-the-fly
```
For streaming data or one-time aggregations, always prefer generator expressions over eager list comprehensions.
