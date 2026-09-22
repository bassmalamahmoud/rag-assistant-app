# Python Virtual Environments, Pip, and Dependency Management

## Overview
Machine learning projects rely on complex, interdependent third-party packages (NumPy, SciPy, PyTorch, Scikit-Learn, FastAPI, Transformers). If installed globally, package version conflicts will inevitably corrupt system environments. Virtual environments provide isolated directory trees containing specific Python interpreters and package versions.

## Key Concepts and Code Examples
The standard Python `venv` module manages local virtual environments:

```bash
# 1. Create a virtual environment named .venv
python -m venv .venv

# 2. Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1
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
