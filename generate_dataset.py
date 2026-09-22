# Script to author the 25 dataset markdown files
import os

BASE_DIR = os.path.abspath("rag-assistant-project/data/raw")
os.makedirs(BASE_DIR, exist_ok=True)

docs = {
    "01_python_variables_datatypes.md": """# Python Variables, Dynamic Typing, and Memory References

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
""",

    "02_python_functions.md": """# Python Functions, Variable Arguments, and Scope

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
""",

    "03_python_oop_classes.md": """# Object-Oriented Programming in Python: Classes and Instances

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
""",

    "04_python_oop_inheritance.md": """# Python OOP Inheritance, Polymorphism, and Super()

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
""",

    "05_python_list_comprehensions.md": """# List, Dict, and Set Comprehensions in Python

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
""",

    "06_python_decorators.md": """# Python Decorators and Function Wrapping

## Overview
Decorators are callables that accept a function as input and return a modified or wrapped function. In machine learning infrastructure, decorators are indispensable for logging execution time, caching expensive dataset transformations, validating tensor input shapes, and handling API rate limits.

## Key Concepts and Code Examples
Under the hood, `@my_decorator` syntax is syntactic sugar for `func = my_decorator(func)`. Using `functools.wraps` is essential to preserve original docstrings, parameter signatures, and function names.

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[TIMING] {func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def train_step(batch_x, batch_y):
    # Simulated computation
    time.sleep(0.01)
    return "weights_updated"
```

Parameterized decorators take arguments by wrapping an extra outer factory function:
```python
def retry(attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == attempts - 1: raise e
        return wrapper
    return decorator
```

## Common Mistake
**Omitting `functools.wraps`:**
A common mistake when writing decorators is omitting `@wraps(func)` on the inner wrapper function. When omitted, the original function loses its identity: its `__name__` becomes `"wrapper"`, and `__doc__` is erased. This breaks introspection, documentation generators, and debugging tools (such as FastAPI route inspectors or pytest fixtures). Always decorate inner wrappers with `@functools.wraps(func)`.
""",

    "07_python_error_handling.md": """# Robust Python Error Handling and Custom Exceptions

## Overview
Data pipelines, model training jobs, and deployment APIs must handle unexpected data anomalies gracefully. Robust error handling using `try`, `except`, `else`, and `finally` blocks prevents silent pipeline failures, corrupted training checkpoints, and uninformative server crashes.

## Key Concepts and Code Examples
Python organizes exceptions in a class hierarchy under `BaseException` and `Exception`. Production code should catch specific exceptions and define custom domain-specific errors.

```python
class DataValidationError(Exception):
    \"\"\"Raised when an incoming dataset violates required schema constraints.\"\"\"
    pass

def load_and_validate_features(filepath: str, required_cols: list[str]):
    try:
        with open(filepath, "r") as f:
            header = f.readline().strip().split(",")
    except FileNotFoundError:
        raise DataValidationError(f"File {filepath} does not exist.")
    except PermissionError:
        raise DataValidationError(f"Cannot read {filepath}: permission denied.")
    else:
        missing = [col for col in required_cols if col not in header]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        return header
    finally:
        print("Completed feature validation check.")
```

The `else` block executes solely when no exception occurred in the `try` block, keeping the `try` block minimal.

## Common Mistake
**Catching Bare `except:` or Suppressing Errors:**
The most harmful mistake in ML error handling is using a bare `except:` or catching generic `Exception` and doing nothing:
```python
# MISTAKE:
try:
    load_weights()
except:
    pass # Catches KeyboardInterrupt, SystemExit, and masks fatal syntax errors!
```
Bare `except:` catches process interruption signals like `KeyboardInterrupt` (Ctrl+C), preventing you from terminating a runaway script. Furthermore, suppressing errors silently causes models to train on uninitialized or garbage data. Always catch specific exceptions (e.g., `except (ValueError, KeyError) as e:`).
""",

    "08_python_context_managers.md": """# Python Context Managers and Resource Management

## Overview
Context managers manage the acquisition and safe release of resources, such as file handles, database connections, GPU memory locks, and temporary directories. They ensure that cleanup logic runs reliably even when unhandled exceptions occur during execution.

## Key Concepts and Code Examples
The `with` statement implements the context management protocol via `__enter__()` and `__exit__()` methods:

```python
class TimerContext:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.perf_counter() - self.start
        print(f"Task '{self.task_name}' finished in {duration:.4f}s")
        # Return False to let any exception propagate upwards
        return False

with TimerContext("Inference Pipeline"):
    # Run heavy prediction job
    sum(range(100_000))
```

Python's `contextlib` module provides a generator-based shortcut `@contextmanager`:
```python
from contextlib import contextmanager

@contextmanager
def temporary_learning_rate(optimizer, temp_lr):
    old_lr = optimizer['lr']
    optimizer['lr'] = temp_lr
    try:
        yield optimizer
    finally:
        optimizer['lr'] = old_lr
```

## Common Mistake
**Suppressing Exceptions Accidentally in `__exit__`:**
In custom context managers, returning a truthy value (such as `True`) from the `__exit__` method causes Python to silently suppress any exception that occurred inside the `with` block:
```python
# MISTAKE:
def __exit__(self, exc_type, exc_val, exc_tb):
    clean_up()
    return True # Accidental suppression! The caller will never know an error occurred!
```
Unless you explicitly intend to swallow the exception, return `False` or `None` from `__exit__`.
""",

    "09_python_virtualenv_packages.md": """# Python Virtual Environments, Pip, and Dependency Management

## Overview
Machine learning projects rely on complex, interdependent third-party packages (NumPy, SciPy, PyTorch, Scikit-Learn, FastAPI, Transformers). If installed globally, package version conflicts will inevitably corrupt system environments. Virtual environments provide isolated directory trees containing specific Python interpreters and package versions.

## Key Concepts and Code Examples
The standard Python `venv` module manages local virtual environments:

```bash
# 1. Create a virtual environment named .venv
python -m venv .venv

# 2. Activate on Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Or on Linux/macOS:
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2

# 4. Freeze exact installed versions
pip freeze > requirements.txt

# 5. Recreate environment elsewhere
pip install -r requirements.txt
```

Environment isolation ensures that library upgrades in one project do not break dependencies in another project.

## Common Mistake
**Installing Packages Globally and Not Pinning Versions:**
A prevalent mistake is running `pip install package_name` without an active virtual environment, polluting the global operating system Python environment. Another critical mistake is writing unpinned `requirements.txt` files:
```
# MISTAKE: Unpinned dependencies
numpy
pandas
fastapi
```
Unpinned packages will automatically install future major versions that contain breaking API changes, causing pipelines to fail in production or CI/CD pipelines. Always use virtual environments and pin version numbers (e.g., `pandas==2.2.2`).
""",

    "10_python_file_io_json_csv.md": """# Python File I/O: Handling Text, JSON, and CSV

## Overview
Data ingestion and serialization are the foundation of ML pipelines. Datasets, hyperparameters, model weights, and inference logs are routinely read from and written to CSV, JSON, and text files. Proper encoding handling, resource buffering, and serialization integrity prevent data corruption.

## Key Concepts and Code Examples
Always use the `with open(...)` construct with explicit character encoding (`encoding="utf-8"`) to avoid platform-dependent encoding bugs.

```python
import json
import csv

# Writing and reading JSON configuration
config = {"model": "RandomForest", "n_estimators": 100, "random_state": 42}
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

with open("config.json", "r", encoding="utf-8") as f:
    loaded_config = json.load(f)

# Reading CSV files using standard csv module
with open("dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Process each row as a dictionary
        pass
```

For large files, read lazily line-by-line rather than reading the entire content into memory with `f.read()`.

## Common Mistake
**Omitting Explicit Encoding and Forgetting `newline=''` in CSV on Windows:**
On Windows, opening files without `encoding="utf-8"` defaults to legacy codepages (e.g., `cp1252`), which crashes with `UnicodeDecodeError` whenever text contains special symbols, emojis, or foreign characters.
Additionally, when writing CSV files on Windows without `newline=''`:
```python
# MISTAKE:
with open("data.csv", "w") as f: # Causes blank lines between rows on Windows!
    writer = csv.writer(f)
    writer.writerow(["id", "score"])

# CORRECT:
with open("data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "score"])
```
Always specify `encoding="utf-8"` and `newline=""` for CSV file writers.
""",

    "11_numpy_arrays_basics.md": """# NumPy Array Fundamentals: Ndarrays, Dtypes, and Memory Layout

## Overview
NumPy (Numerical Python) is the foundational library for scientific computing and machine learning in Python. At its core is the `ndarray` (N-dimensional array), a contiguous block of homogeneous memory optimized for high-performance vectorized operations implemented in C and Fortran.

## Key Concepts and Code Examples
Unlike standard Python lists that store pointers to scattered objects, NumPy arrays store raw binary data in contiguous memory blocks with explicit data types (`dtypes`):

```python
import numpy as np

# Creating arrays from Python lists
arr_1d = np.array([1.0, 2.5, 3.8], dtype=np.float32)
arr_2d = np.zeros((3, 4), dtype=np.int32)
arr_range = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
arr_linspace = np.linspace(0, 1, 5)   # [0.0, 0.25, 0.5, 0.75, 1.0]

# Inspecting array properties
print(arr_2d.shape)  # (3, 4) -> (rows, columns)
print(arr_2d.ndim)   # 2
print(arr_2d.dtype)  # int32
print(arr_2d.itemsize) # 4 bytes per element
print(arr_2d.nbytes)   # 3 * 4 * 4 = 48 bytes total
```

Understanding `shape` and `dtype` is crucial for feeding matrices into neural networks and scikit-learn models.

## Common Mistake
**Dtype Overflow and Unintended Type Casting:**
A dangerous mistake is using small integer types (like `np.uint8` commonly used for image pixel values) and performing arithmetic that overflows:
```python
# MISTAKE:
pixels = np.array([200, 250], dtype=np.uint8)
result = pixels + 60
print(result) # Output: [4, 54] -> Silently wrapped around due to 8-bit overflow (256)!

# CORRECT:
result = pixels.astype(np.int32) + 60 # Output: [260, 310]
```
Always cast low-precision arrays (such as `uint8` image tensors) to higher precision types (`float32` or `int32`) before performing arithmetic.
""",

    "12_numpy_indexing_slicing.md": """# NumPy Indexing, Slicing, and Views vs Copies

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
""",

    "13_numpy_broadcasting.md": """# NumPy Broadcasting Rules and Vectorized Alignment

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
""",

    "14_numpy_math_operations.md": """# NumPy Vectorized Mathematical Operations and Ufuncs

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
""",

    "15_numpy_linear_algebra.md": """# NumPy Linear Algebra: Matrix Multiplication, Inverses, and Decompositions

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
""",

    "16_numpy_random_module.md": """# NumPy Random Number Generation and Seed Reproducibility

## Overview
Stochastic processes are everywhere in Machine Learning: initializing neural network weights, splitting train/test sets, shuffling mini-batches, and Monte Carlo simulations. Modern NumPy uses a decoupled generator architecture (`np.random.default_rng`) for superior statistical properties and speed.

## Key Concepts and Code Examples
The modern, recommended approach to random numbers in NumPy is instantiating a `Generator`:

```python
import numpy as np

# Instantiate random Generator with fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Generate uniform floats in [0.0, 1.0)
uniform_data = rng.random((3, 3))

# Standard normal distribution (Gaussian, mean=0, std=1)
normal_weights = rng.normal(loc=0.0, scale=1.0, size=(100, 10))

# Random integers (e.g. for categorical labels)
labels = rng.integers(low=0, high=2, size=100)

# Shuffling an array along the first axis in-place
indices = np.arange(100)
rng.shuffle(indices)
```

## Common Mistake
**Using Legacy `np.random.seed()` in Multi-Threaded or Modern Code:**
Using the legacy `np.random.seed(42)` modifies a global random state. In multi-threaded applications, parallel worker processes, or Jupyter notebook reruns, this global state can become non-deterministic or cause race conditions.
```python
# MISTAKE (Legacy):
np.random.seed(42)
val = np.random.randn(5)

# CORRECT (Modern NumPy):
rng = np.random.default_rng(seed=42)
val = rng.standard_normal(5)
```
Always instantiate `rng = np.random.default_rng(seed)` for reproducible experiments.
""",

    "17_pandas_series_dataframe.md": """# Pandas Data Structures: Series, DataFrame, and Index Alignment

## Overview
Pandas is the primary data manipulation and analysis library in Python. Its two core data structures are the 1D `Series` (a labeled, typed array) and the 2D `DataFrame` (a tabular structure of rows and columns). A defining superpower of Pandas is automatic **Index Alignment**, where operations align on labels rather than positions.

## Key Concepts and Code Examples
Every Series and DataFrame has an explicit index:

```python
import pandas as pd

# Creating a Series with custom index
s1 = pd.Series([10, 20, 30], index=["a", "b", "c"])
s2 = pd.Series([1, 2, 3], index=["b", "c", "d"])

# Automatic index alignment during arithmetic:
result = s1 + s2
# Result:
# a    NaN  (only in s1)
# b   21.0  (10 + 1)
# c   32.0  (20 + 2)
# d    NaN  (only in s2)

# Creating a DataFrame
data = {
    "feature_a": [1.2, 3.4, 5.6],
    "category": ["A", "B", "A"],
    "target": [0, 1, 0]
}
df = pd.DataFrame(data, index=["obs_1", "obs_2", "obs_3"])
print(df.dtypes)
print(df.shape)
```

## Common Mistake
**Ignoring Automatic Index Alignment when Adding New Columns:**
A common pitfall occurs when assigning an existing Series with a different or scrambled index into a DataFrame:
```python
# MISTAKE:
df = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
new_series = pd.Series([100, 200, 300], index=[2, 1, 0]) # Inverted index!

df["new_col"] = new_series
# Result: df['new_col'] aligns by label (0 gets 300, 1 gets 200, 2 gets 100),
# NOT by positional order!

# If position-based assignment is intended, pass the raw NumPy values:
df["new_col"] = new_series.to_numpy()
```
Always be mindful that Pandas aligns by index labels, not positional sequence.
""",

    "18_pandas_reading_writing_data.md": """# Pandas Data Ingestion: Efficient CSV, Parquet, and Chunking

## Overview
Real-world machine learning workflows begin with ingesting tabular data from CSV, JSON, Parquet, or SQL. `pd.read_csv()` provides dozens of parameters to control parsing, datetime conversions, chunking, and memory footprint.

## Key Concepts and Code Examples
Optimizing ingestion speed and memory usage is vital when handling datasets larger than available RAM:

```python
import pandas as pd

# Reading CSV with explicit types and datetime parsing
df = pd.read_csv(
    "dataset.csv",
    usecols=["user_id", "timestamp", "transaction_amount", "label"],
    dtype={"user_id": "int32", "transaction_amount": "float32", "label": "int8"},
    parse_dates=["timestamp"]
)

# Reading massive files in chunks to avoid Out-Of-Memory (OOM) crashes:
chunk_size = 50_000
total_records = 0
for chunk in pd.read_csv("huge_file.csv", chunksize=chunk_size):
    # Process chunk (e.g. filter or aggregate)
    filtered = chunk[chunk["transaction_amount"] > 100]
    total_records += len(filtered)

# Writing to high-performance Parquet format (preserves dtypes & compresses)
df.to_parquet("dataset.parquet", engine="pyarrow", compression="snappy")
```

Parquet files load up to 10x faster than raw CSVs and retain column datatypes intact.

## Common Mistake
**Loading Entire Massive Files into RAM Without `usecols` or `chunksize`:**
Attempting to load a multi-gigabyte CSV using default `pd.read_csv("data.csv")` leads to high memory consumption and browser/kernel termination (`Out of Memory`).
Always specify `usecols` to load only the required features, define efficient `dtypes` (e.g., `float32` instead of `float64`), or iterate using `chunksize`.
""",

    "19_pandas_indexing_selection.md": """# Pandas Indexing and Selection: `loc` vs `iloc` and Chained Assignment

## Overview
Selecting rows and columns correctly in Pandas is essential. Pandas provides two distinct indexing accessors:
- `.loc[]`: Label-based indexing (uses index names and column names).
- `.iloc[]`: Integer position-based indexing (uses 0-indexed integer coordinates).

## Key Concepts and Code Examples
Using explicit accessors prevents ambiguity, especially when indices are integers:

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 30, 35, 40],
    "salary": [50000, 65000, 80000, 110000]
}, index=[10, 20, 30, 40]) # Non-sequential integer index

# Label-based selection with .loc
row_30 = df.loc[30, "salary"] # Finds row with index label 30 -> 80000

# Position-based selection with .iloc
row_pos2 = df.iloc[2, 1]      # Finds 3rd row (pos 2), 2nd col (pos 1) -> 80000

# Slicing with .loc includes both start AND end labels:
print(df.loc[10:30]) # Includes rows labeled 10, 20, AND 30!

# Slicing with .iloc excludes end position (standard Python slice):
print(df.iloc[0:2])  # Includes rows at indices 0 and 1 only
```

## Common Mistake
**The `SettingWithCopyWarning` and Chained Assignment:**
The most frequent error in Pandas is chained indexing assignment:
```python
# MISTAKE:
df[df["age"] > 30]["salary"] = 90000 # Triggers SettingWithCopyWarning!
# May not modify the original DataFrame because df[df['age'] > 30] returned a temporary copy!

# CORRECT:
df.loc[df["age"] > 30, "salary"] = 90000
```
Always use single-bracket `.loc[row_mask, col_selector] = value` when modifying subsets of a DataFrame.
""",

    "20_pandas_filtering_boolean.md": """# Pandas Boolean Filtering and Logical Operators

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
""",

    "21_pandas_groupby.md": """# Pandas GroupBy: Split-Apply-Combine Operations

## Overview
The `groupby` mechanism implements the classic **Split-Apply-Combine** pattern. It splits data into groups based on key columns, applies a transformation or aggregation function to each group independently, and combines the results into a new data structure.

## Key Concepts and Code Examples
Pandas provides `.agg()`, `.transform()`, and `.filter()` methods on GroupBy objects:

```python
import pandas as pd

df = pd.DataFrame({
    "store": ["North", "North", "South", "South", "East"],
    "product": ["A", "B", "A", "B", "A"],
    "revenue": [100, 150, 80, 200, 120]
})

# Aggregation: reduces each group to a single value
store_summary = df.groupby("store")["revenue"].agg(["sum", "mean", "count"])

# Transform: returns an aligned Series matching the original DataFrame's length
# (e.g. for calculating percentage of group total or group mean normalization)
group_mean = df.groupby("store")["revenue"].transform("mean")
df["revenue_diff_from_store_mean"] = df["revenue"] - group_mean

# Filter: drops groups that do not satisfy a group-level boolean predicate
high_volume_stores = df.groupby("store").filter(lambda g: g["revenue"].sum() > 200)
```

## Common Mistake
**Iterating Over Groups with `for name, group in df.groupby(...)` for Calculations:**
Writing manual loops over groupby objects in Python is orders of magnitude slower than using vectorized `.agg()`, `.transform()`, or built-in methods:
```python
# MISTAKE:
# means = {}
# for store, group in df.groupby("store"):
#     means[store] = group["revenue"].mean() # Very slow on large datasets!

# CORRECT:
means = df.groupby("store")["revenue"].mean() # Highly optimized in C
```
Leverage vectorized `.agg()` or `.transform()` instead of manual Python loops.
""",

    "22_pandas_merging_joining.md": """# Pandas Merging, Joining, and Concatenation

## Overview
Data science projects typically require combining data across multiple tables (e.g. joining customer demographic tables with transaction logs). Pandas provides `pd.merge()` (relational database-style joins), `pd.concat()` (stacking tables along an axis), and `DataFrame.join()` (joining on indices).

## Key Concepts and Code Examples
Understanding join types (`inner`, `left`, `right`, `outer`) ensures that no critical observations are unintentionally lost:

```python
import pandas as pd

customers = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "name": ["Alice", "Bob", "Charlie"]
})

orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4],
    "customer_id": [101, 102, 101, 104],
    "amount": [250, 120, 300, 75]
})

# Relational inner merge
merged_inner = pd.merge(customers, orders, on="customer_id", how="inner")

# Left join preserving all customers and showing merge status with indicator=True
merged_left = pd.merge(customers, orders, on="customer_id", how="left", indicator=True)

# Concatenating / Stacking rows
batch1 = pd.DataFrame({"feat": [1, 2]})
batch2 = pd.DataFrame({"feat": [3, 4]})
stacked = pd.concat([batch1, batch2], ignore_index=True)
```

## Common Mistake
**Silent Cartesian Explosion on Duplicate Merge Keys:**
If both the left and right DataFrames contain duplicate values in the join key, `pd.merge()` performs a many-to-many Cartesian product, dramatically inflating row counts and memory usage without raising an error:
```python
# MISTAKE:
# If left table has 1,000 rows with customer_id=101, and right table has 1,000 rows with customer_id=101,
# the merged result produces 1,000,000 rows!

# PREVENTION:
# Use validate parameter to enforce one-to-one, one-to-many, or many-to-one constraints:
pd.merge(customers, orders, on="customer_id", how="left", validate="1:m")
```
Always use `validate="1:m"`, `validate="m:1"`, or `validate="1:1"` in `pd.merge()` to catch unexpected duplicates.
""",

    "23_pandas_missing_data.md": """# Handling Missing Data in Pandas: Detection, Imputation, and Nullable Dtypes

## Overview
Missing values (`NaN`, `None`, `<NA>`) are commonplace in observational and machine learning datasets. Machine learning estimators in scikit-learn generally crash if input feature matrices contain missing values. Proper detection, deletion, or imputation is a mandatory preprocessing phase.

## Key Concepts and Code Examples
Pandas represents missing values using `np.nan` (for floats) and the modern pandas `<NA>` singleton (`pd.NA`) for nullable integer, boolean, and string dtypes:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age": [25, np.nan, 30, np.nan],
    "income": [50000, 60000, np.nan, 80000],
    "city": ["Cairo", "Alexandria", None, "Cairo"]
})

# Detecting missing values
print(df.isna().sum()) # Total missing values per feature

# Dropping rows or columns with missing values
clean_rows = df.dropna(subset=["age", "income"])

# Imputing missing values with summary statistics
mean_age = df["age"].mean()
df["age_imputed"] = df["age"].fillna(mean_age)

# Forward-fill / Backward-fill for time-series data
df["income_ffill"] = df["income"].ffill()
```

## Common Mistake
**Checking for NaN with Direct Equality `x == np.nan`:**
A notorious mistake is attempting to filter or test missing values using `df["age"] == np.nan`. In accordance with the IEEE 754 floating-point specification, `np.nan == np.nan` evaluates to `False`!
```python
# MISTAKE:
nan_count = len(df[df["age"] == np.nan]) # Always returns 0!

# CORRECT:
nan_count = df["age"].isna().sum()
# Or in pure python / numpy:
# np.isnan(val)
```
Always use `.isna()` or `.notna()` to identify missing values in Pandas.
""",

    "24_pandas_pivot_tables.md": """# Pandas Reshaping: Pivot, Pivot Tables, and Melt

## Overview
Reshaping data between wide format and long (tidy) format is essential for exploratory data analysis, time-series feature engineering, and visualization. Pandas provides `pivot()` for reshaping without aggregation, `pivot_table()` for aggregation over multiple dimensions, and `melt()` for unpivoting wide tables into long format.

## Key Concepts and Code Examples
Understanding the distinction between `pivot` and `pivot_table`:

```python
import pandas as pd

df = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
    "city": ["London", "Paris", "London", "Paris"],
    "temperature": [5, 7, 4, 8],
    "humidity": [80, 75, 85, 70]
})

# Pivot table: Aggregates multiple entries using aggfunc (default: mean)
pivot_df = df.pivot_table(
    index="date",
    columns="city",
    values="temperature",
    aggfunc="mean"
)

# Unpivoting using melt (wide to long format for ML pipelines)
melted = pd.melt(
    df,
    id_vars=["date", "city"],
    value_vars=["temperature", "humidity"],
    var_name="metric",
    value_name="value"
)
```

## Common Mistake
**Calling `df.pivot()` on Data with Duplicate Index/Column Combinations:**
Calling `df.pivot()` when the dataset contains duplicate entries for the chosen `index` and `columns` raises `ValueError: Index contains duplicate entries, cannot reshape`.
```python
# MISTAKE:
# df.pivot(index="date", columns="city", values="temperature") # Crashes if (date, city) repeats!

# CORRECT:
# Use pivot_table with an explicit aggregation function (e.g. mean or sum):
df.pivot_table(index="date", columns="city", values="temperature", aggfunc="mean")
```
Use `pivot_table()` whenever duplicate index/column pairs may exist in your dataset.
""",

    "25_pandas_data_cleaning.md": """# Pandas Data Cleaning: String Accessor, Datetime Parsing, and Deduplication

## Overview
Real-world machine learning data is messy: inconsistent string formatting, erratic date stamps, and duplicate rows. Pandas provides the `.str` accessor for vectorized string manipulations, `.dt` accessor for timestamp extraction, and `.drop_duplicates()` for deduplication.

## Key Concepts and Code Examples
Vectorized data cleaning operations avoid slow Python string iterations:

```python
import pandas as pd

df = pd.DataFrame({
    "raw_text": ["  Standard MODEL  ", "advanced-model", "STANDARD model  "],
    "date_str": ["2026-03-15 14:30:00", "2026-03-16 09:15:00", "2026-03-15 14:30:00"],
    "user_id": [101, 102, 101]
})

# 1. Vectorized string cleaning using .str accessor
df["clean_model"] = df["raw_text"].str.strip().str.lower().str.replace("-", "_")

# 2. Datetime parsing and feature extraction using .dt accessor
df["datetime"] = pd.to_datetime(df["date_str"])
df["year"] = df["datetime"].dt.year
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()

# 3. Removing duplicate records
df_unique = df.drop_duplicates(subset=["clean_model", "user_id"], keep="first")
```

## Common Mistake
**Using Standard Python String Methods Instead of the `.str` Accessor:**
A common mistake is attempting to call Python string functions directly on a Series, which raises `AttributeError: 'Series' object has no attribute 'lower'`:
```python
# MISTAKE:
# df["clean"] = df["raw_text"].strip().lower() # AttributeError!

# CORRECT:
df["clean"] = df["raw_text"].str.strip().str.lower()
```
Always prefix string operations on Pandas Series with the `.str` accessor (e.g., `df['col'].str.replace(...)`).
"""
}

for filename, content in docs.items():
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

print(f"Successfully generated {len(docs)} markdown files in {BASE_DIR}")
