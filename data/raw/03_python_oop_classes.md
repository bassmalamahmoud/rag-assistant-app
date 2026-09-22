# Object-Oriented Programming in Python: Classes and Instances

## Overview
Object-Oriented Programming (OOP) is crucial for structuring complex machine learning pipelines. Custom transformers, PyTorch dataset classes, neural network layers, and evaluation harnesses are all implemented as classes. Understanding the difference between class attributes and instance attributes is vital for building robust stateful components.

## Key Concepts and Code Examples
A class is defined with the `class` keyword. The `__init__` constructor method binds attributes to the specific instance (`self`), while `__repr__` and `__str__` provide readable string representations.

```python
class Scaler:
    # Class attribute (shared by all instances)
    category = "Data Preprocessor"

    def __init__(self, feature_name: str):
        # Instance attributes (unique to each instance)
        self.feature_name = feature_name
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values):
        self.mean = sum(values) / len(values)
        variance = sum((x - self.mean) ** 2 for x in values) / len(values)
        self.std = variance ** 0.5
        return self

    def transform(self, values):
        return [(x - self.mean) / self.std for x in values]

    def __repr__(self):
        return f"Scaler(feature={self.feature_name}, mean={self.mean:.2f}, std={self.std:.2f})"
```

## Common Mistake
**Class Attribute vs Instance Attribute Shadowing:**
A common mistake is declaring a mutable object (like a list or dictionary) at the class level instead of inside `__init__`. When defined as a class attribute, all instances share the exact same reference:
```python
# MISTAKE:
class ModelRegistry:
    models = []  # Shared across every instance of ModelRegistry!

# If registry_a.models.append("ResNet") is called, registry_b.models also contains "ResNet".

# CORRECT:
class ModelRegistry:
    def __init__(self):
        self.models = []  # Isolated per instance
```
Ensure all instance-specific state is instantiated inside `__init__`.
