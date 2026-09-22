# Python Functions, Variable Arguments, and Scope

## Overview
Functions are the fundamental units of execution and abstraction in Python. In data science and machine learning pipelines, functions encapsulate data cleaning, feature extraction, evaluation metrics, and model training routines. Grasping lexical scoping (LEGB rule: Local, Enclosing, Global, Built-in) and flexible argument parsing via `*args` and `**kwargs` is essential for writing modular ML code.

## Key Concepts and Code Examples
Python functions support positional arguments, keyword arguments, default arguments, and arbitrary argument packing:

```python
def train_model(model_name, *hyperparameters, learning_rate=0.01, **training_flags):
    print(f"Model: {model_name}, LR: {learning_rate}")
    print(f"Positional args: {hyperparameters}")
    print(f"Keyword flags: {training_flags}")

train_model("RandomForest", 100, 5, learning_rate=0.001, early_stopping=True, verbose=2)
```

In Machine Learning libraries like Scikit-Learn or PyTorch, wrappers frequently use `*args, **kwargs` to pass configuration dynamically to underlying estimators:
```python
def fit_estimator(estimator_cls, x_train, y_train, **fit_params):
    model = estimator_cls()
    return model.fit(x_train, y_train, **fit_params)
```

## Common Mistake
**The Mutable Default Argument Trap:**
The most notorious mistake in Python functions is defining default argument values with mutable objects (such as `list` or `dict`). In Python, default parameter expressions are evaluated once when the function is defined, not each time the function is invoked:
```python
# MISTAKE:
def append_metric(score, metric_history=[]):
    metric_history.append(score)
    return metric_history

print(append_metric(0.85)) # [0.85]
print(append_metric(0.92)) # [0.85, 0.92] -> Shares the same list object across calls!

# CORRECT:
def append_metric(score, metric_history=None):
    if metric_history is None:
        metric_history = []
    metric_history.append(score)
    return metric_history
```
Always use `None` as the sentinel default value for mutable arguments.
