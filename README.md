# ASOL Synthetic Data Generation

This repository contains a project for generating synthetic data for ASOL using computer vision techniques.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python** (>=3.9, <4.0)
- **Poetry** (Python dependency management tool)
- **CUDA 12.6** (for GPU support, if available)

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/JacobCrown/asol-synthetic-data-generation.git
cd asol-synthetic-data-generation
```

### 2. Install Poetry (if not installed)

Poetry is required to manage dependencies. You can install it using:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, ensure Poetry is available:

```sh
poetry --version
```

### 3. Install Project Dependencies

Run the following command to install dependencies in an isolated virtual environment:

```sh
poetry install
```

This will:

- Create a virtual environment.
- Install all required packages.
- Set up the project for execution.

### 4. Activate the Virtual Environment

Poetry manages its own virtual environment. To activate it, run:

```sh
poetry shell
```

Now you can execute Python scripts within this environment.

### 5. Verify Installation

Check if the installation is successful by running:

```sh
python -c "import torch; print(torch.cuda.is_available())"
```

If `True` is printed, PyTorch is using CUDA.

## Running the Project

To execute scripts, use:

```sh
poetry run python scripts/some_script.py
```

## Notes

- This project uses `Poetry` to ensure proper package management.
- The `pytorch-gpu` source is used to install `torch` and `torchvision` with CUDA 12.6 support.
- If using a different CUDA version, modify `pyproject.toml` accordingly.

## Troubleshooting

If you encounter issues:

1. Ensure Python version is between 3.9 and 4.0.
2. Check CUDA installation using `nvcc --version`.
3. Ensure Poetry is installed correctly by running `poetry --version`.
4. If `torch.cuda.is_available()` returns `False`, check if the correct PyTorch version is installed.
