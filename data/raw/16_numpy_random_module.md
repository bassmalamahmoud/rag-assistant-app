# NumPy Random Number Generation and Seed Reproducibility

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
