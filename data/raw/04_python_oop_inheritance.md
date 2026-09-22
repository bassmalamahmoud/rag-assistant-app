# Python OOP Inheritance, Polymorphism, and Super()

## Overview
Inheritance enables code reusability by allowing a subclass to derive attributes and methods from a base class. In Machine Learning systems, standard interfaces (such as `BaseEstimator` and `TransformerMixin` in scikit-learn, or `torch.nn.Module` in PyTorch) rely heavily on inheritance and polymorphism to provide unified `.fit()`, `.transform()`, or `.forward()` APIs.

## Key Concepts and Code Examples
Using `super()` allows subclasses to invoke parent class initializers and methods without hard-coding parent class names, which maintains clean Method Resolution Order (MRO):

```python
class BaseEstimator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit()")

class LogisticRegressionModel(BaseEstimator):
    def __init__(self, model_name: str, C: float = 1.0):
        super().__init__(model_name)
        self.C = C

    def fit(self, X, y):
        print(f"Fitting {self.model_name} with regularization C={self.C}")
        self.is_fitted = True
        return self
```

Python supports multiple inheritance and resolves method dispatch using C3 Linearization (inspectable via `ClassName.mro()`).

## Common Mistake
**Forgetting to Call `super().__init__()`:**
When subclassing framework classes like `torch.nn.Module` or custom base classes, a frequent beginner mistake is defining `def __init__(self):` without calling `super().__init__()`:
```python
# MISTAKE:
class NeuralNet(BaseEstimator):
    def __init__(self, layers):
        self.layers = layers
        # Forgot super().__init__("NeuralNet")!
        # Accessing self.is_fitted will crash with AttributeError.
```
Without `super().__init__()`, parent attributes and internal registration hooks are never initialized, leading to cryptic `AttributeError` exceptions when running pipelines.
