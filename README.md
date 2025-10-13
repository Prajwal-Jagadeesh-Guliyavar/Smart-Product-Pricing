
# Smart Product Pricing Challenge

This project contains a complete, end-to-end machine learning pipeline to predict product prices based on their text descriptions and images.

---

## 1. Model Pipeline and Architecture

This solution uses a modern, multi-modal approach that processes tabular data, text, and images separately before combining them for a final prediction.

1.  **Feature Extraction (Embeddings):**
    *   **Text Features:** Product descriptions are converted into 384-dimensional numerical vectors using a pre-trained `all-MiniLM-L6-v2` model.
    *   **Image Features:** Product images are processed by a pre-trained `ResNet50` model to generate 2048-dimensional feature vectors.
    *   **Numerical Features:** The Item Pack Quantity (IPQ) is extracted from the text description.

2.  **Model Training:**
    *   The text embeddings, image embeddings, and IPQ are concatenated into a single feature vector.
    *   A **LightGBM (LGBM) Regressor** is trained on this combined feature set using 5-fold cross-validation. The final prediction is an average of the 5 models.

---

## 2. Project Structure

- `main.py`: The main, menu-driven interface to run all project tasks.
- `train_lgbm.py`: Contains the logic for the model training pipeline (called by `main.py`).
- `predict.py`: Contains the logic for the prediction pipeline (called by `main.py`).
- `download_data.py`: Contains the logic for downloading the image datasets (called by `main.py`).
- `perform_eda.py`: Contains the logic for generating the EDA plot (called by `main.py`).
- `student_resource/`: The directory containing the raw datasets and images.
- `artifacts/`: The directory where trained models and cached embeddings are stored.

---

## 3. Setup and Execution

Follow these steps to set up and run the project.

### Step 1: Install Dependencies

First, create and activate a Python virtual environment. Then, install all required libraries from the `requirements.txt` file.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (on Linux/macOS)
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Run the Main Menu

This project uses a user-friendly interactive menu. All actions (downloading, training, predicting) are performed through this single interface.

**To start the program, simply run:**
```bash
python main.py
```

You will be greeted with a numbered menu. Simply enter the number for the action you wish to perform.

- **First-time setup:** Choose option `2` to download all datasets.
- **To train the model:** Choose option `3`.
- **To generate the submission:** Choose option `4`.

The script includes pre-flight checks and will guide you if data is missing or models haven't been trained yet.
