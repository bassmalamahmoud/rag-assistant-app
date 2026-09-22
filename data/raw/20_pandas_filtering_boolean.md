# Pandas Boolean Filtering and Logical Operators

## Overview
Filtering datasets based on conditional logic is fundamental for cohort analysis, outlier removal, and preparing training and test subsets. In Pandas, multiple conditions require bitwise operators (`&`, `|`, `~`) instead of Python's logical keywords (`and`, `or`, `not`).

## Key Concepts and Code Examples
Boolean masks evaluate conditions element-wise across a Series:

```python
import pandas as pd

df = pd.DataFrame({
    "department": ["Engineering", "HR", "Engineering", "Marketing", "HR"],
    "experience": [5, 2, 8, 3, 1],
    "active": [True, False, True, True, True]
})

# Combining multiple conditions:
# Parentheses are MANDATORY around each condition due to operator precedence!
senior_engineers = df[(df["department"] == "Engineering") & (df["experience"] >= 5)]

# Using .isin() for membership filtering
tech_and_hr = df[df["department"].isin(["Engineering", "HR"])]

# Inverting conditions using ~
inactive = df[~df["active"]]

# Using query() for readable string-based filtering
high_exp = df.query("experience > 4 and department == 'Engineering'")
```

## Common Mistake
**Using Python `and`/`or` and Omitting Parentheses:**
A very common mistake is writing `df[df['experience'] > 2 and df['active']]`. Python's `and` keyword evaluates the truth value of the entire Series object, resulting in `ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()`.
Another common mistake is omitting parentheses: `df[df['experience'] > 2 & df['active']]`, which fails because the bitwise `&` operator has higher precedence than `>`.
Always use `&`, `|`, and wrap each condition in parentheses: `(condition_1) & (condition_2)`.
