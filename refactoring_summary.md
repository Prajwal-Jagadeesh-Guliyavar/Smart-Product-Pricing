# Refactoring Summary: ML AWS Challenge Project

This document summarizes the refactoring process undertaken to improve the structure, maintainability, and scalability of the ML AWS Challenge project.

## Initial Problems Identified

The initial codebase, while functional, exhibited several common issues found in projects evolving from notebook-based development:

*   **Code Duplication:** Significant portions of data preprocessing and feature engineering logic were duplicated across `train_lgbm.py` and `predict.py`. This led to increased maintenance effort and a high risk of inconsistencies (training-serving skew).
*   **Hardcoded Configurations:** File paths, model parameters, and other critical settings were hardcoded directly within multiple scripts (`main.py`, `download_data.py`, `train_lgbm.py`, `predict.py`), making configuration management difficult and error-prone.
*   **Lack of Modularity:** The project lacked a clear, modular structure, hindering code reuse and readability.
*   **Manual CLI:** The `main.py` script served as a manual command-line interface, but its internal logic was monolithic and less organized than ideal.
*   **Improper Imports:** The project used `sys.path` manipulation to import utility functions, which is an anti-pattern for Python package management.

## Refactoring Plan (TODOs)

A six-step plan was devised to address these issues:

1.  **Restructure the project:** Create a `src` directory and move all Python source files into it.
2.  **Centralize configuration:** Create a single configuration file (`config.yaml`) to store all settings.
3.  **Create a shared processing module:** Extract the duplicated data processing and feature engineering logic into a new module within `src`.
4.  **Refactor training and prediction scripts:** Modify the training and prediction scripts to import and use the shared processing module and the centralized configuration.
5.  **Refactor `main.py`:** Update `main.py` to use the new structure and configuration.
6.  **Add `__init__.py` files:** Add `__init__.py` files to the `src` directory and any subdirectories to make them importable as Python packages.

## Actions Taken

### 1. Project Restructuring
*   A new directory `src/` was created.
*   All primary Python scripts (`main.py`, `download_data.py`, `perform_eda.py`, `predict.py`, `train_lgbm.py`, and `challenge_utils.py` from `student_resource/src`) were moved into the `src/` directory.

### 2. Centralized Configuration
*   A `config.yaml` file was created in the project root.
*   All identified hardcoded paths, model parameters, and training settings were extracted and consolidated into `config.yaml`.
*   The `PyYAML` library was added to `requirements.txt` to enable YAML file parsing.

### 3. Shared Processing Module
*   A new module `src/processing.py` was created.
*   Common functions for data loading, preprocessing (`load_and_preprocess_data`), text embedding generation (`get_text_embeddings`), and image embedding generation (`get_image_embeddings`) were moved into `src/processing.py`.
*   A `load_config` function was added to `src/processing.py` to facilitate easy access to the centralized configuration.
*   Embedding functions were enhanced with caching logic and a `force_regenerate` flag for flexibility.

### 4. Refactoring Training and Prediction Scripts
*   `src/train_lgbm.py` and `src/predict.py` were updated to import necessary functions from `src/processing.py`.
*   All hardcoded configurations within these scripts were replaced with values loaded from the `config.yaml` via the `load_config` function.
*   The `sys.path` manipulation for `challenge_utils` was removed, as `challenge_utils.py` is now part of the `src` package.

### 5. Refactoring `main.py`
*   `src/main.py` was significantly refactored to improve modularity and readability.
*   Hardcoded paths were removed and replaced with values from `config.yaml`.
*   The main menu logic was broken down into smaller, dedicated handler functions (e.g., `handle_eda`, `handle_download`, `handle_train`).
*   A dictionary mapping user choices to these handler functions was implemented for a cleaner menu system.
*   The `if __name__ == '__main__':` block was updated to ensure the script runs from the project root, improving execution consistency.

### 6. Adding `__init__.py` Files
*   An empty `__init__.py` file was created in the `src/` directory. This signals to Python that `src/` is a package, allowing for proper relative and absolute imports within the project.

## Benefits of Refactoring

*   **Improved Maintainability:** Changes to data processing or configuration now only need to be made in one place.
*   **Reduced Bugs:** Eliminating code duplication drastically reduces the chance of introducing inconsistencies and errors.
*   **Enhanced Readability:** A clear project structure and modular functions make the codebase easier to understand and navigate.
*   **Easier Collaboration:** Developers can work on different modules without stepping on each other's toes.
*   **Scalability:** The modular design makes it easier to add new features or integrate new models in the future.
*   **Adherence to Best Practices:** The project now follows common software engineering and MLOps best practices for structuring machine learning code.
