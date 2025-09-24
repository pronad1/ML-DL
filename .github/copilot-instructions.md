# Copilot Instructions for ML-DL Codebase

## Overview
This workspace contains a variety of Jupyter notebooks and datasets for machine learning and deep learning experiments. The structure is organized by topic and day, supporting incremental learning and experimentation.

## Architecture & Structure
- **Root Notebooks**: General ML/DL topics (`demo_deep_learning.ipynb`, `Machine_learning_part_1.ipynb`, etc.)
- **100 days DL/ml**: Progressive daily learning folders, each with topic-focused notebooks and datasets.
- **Campus_X_100-days-of-machine-learning**: Subfolder with additional day-wise exercises and datasets.
- **Datasets**: CSV, TSV, and JSON files are used for hands-on data analysis and modeling.

## Developer Workflows
- **Jupyter Notebooks**: All code is run and debugged interactively in notebooks. No build system or test runner is present.
- **Data Loading**: Use pandas (`pd.read_csv`, `pd.read_json`) for loading datasets. Paths are relative to the notebook location.
- **Experimentation**: Each notebook is self-contained; results and models are not shared across notebooks.
- **Environment**: Python (>=3.7) with common ML packages (numpy, pandas, scikit-learn, matplotlib, seaborn, tensorflow, keras, etc.).

## Project-Specific Patterns
- **Naming**: Notebooks and datasets are named by topic and day for easy navigation.
- **Data Paths**: Always use relative paths for loading data (e.g., `../day-15_working_with_csv/aug_train.csv`).
- **Visualization**: Use matplotlib/seaborn for plots; save figures only if required by the exercise.
- **No Custom Modules**: All code is inline in notebooks; no reusable Python modules or packages.

## Integration Points
- **External Data**: Some notebooks fetch data from Kaggle or APIs; store API keys in `kaggle.json` (do not commit sensitive info).
- **Images**: Used for visualizations and explanations, stored alongside notebooks.

## Examples
- Loading a dataset:
  ```python
  import pandas as pd
  df = pd.read_csv('placement-dataset.csv')
  ```
- Plotting:
  ```python
  import matplotlib.pyplot as plt
  plt.plot(df['x'], df['y'])
  plt.show()
  ```

## Key Files & Directories
- `100 days DL/`, `100 days ml/`: Main learning tracks
- `Campus_X_100-days-of-machine-learning/`: Extended exercises
- `README.md`: Minimal, does not contain workflow details
- `.github/copilot-instructions.md`: This file

## Conventions
- Keep notebooks self-contained
- Use clear, descriptive names for files and variables
- Do not introduce external dependencies without documenting them in the notebook
- Avoid hardcoding absolute paths

---
_Last updated: September 14, 2025_
