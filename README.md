# Smart Product Pricing Challenge

This project contains a complete, end-to-end machine learning pipeline to predict product prices based on their text descriptions and images.

---

## 1. Model Pipeline and Architecture

This solution uses a modern, multi-modal approach that processes tabular data, text, and images separately before combining them for a final prediction.

1.  **Feature Extraction (Embeddings):**
    *   **Text Features:** Product descriptions are converted into 384-dimensional numerical vectors using a pre-trained `all-MiniLM-L6-v2` model from the `sentence-transformers` library.
    *   **Image Features:** Product images are processed by a pre-trained `ResNet50` model (trained on ImageNet) to generate 2048-dimensional feature vectors.
    *   **Numerical Features:** The Item Pack Quantity (IPQ) is extracted from the text description.

2.  **Model Training:**
    *   The text embeddings, image embeddings, and IPQ are concatenated into a single feature vector for each product.
    *   A **LightGBM (LGBM) Regressor** is trained on this combined feature set. LightGBM is a powerful and efficient gradient boosting framework well-suited for this kind of tabular data.
    *   To ensure robustness, the model is trained using **5-fold cross-validation**. This means 5 separate models are trained, and their predictions are averaged for the final result.

3.  **Prediction:**
    *   The prediction pipeline mirrors the training process, applying the same text and image feature extraction to the test data.
    *   The final price is an average of the predictions from the 5 models saved during cross-validation.

---

## 2. Project Structure

- `main.py`: The main, menu-driven interface to run all project tasks.
- `train_lgbm.py`: Contains the logic for the model training pipeline.
- `predict.py`: Contains the logic for the prediction pipeline.
- `download_data.py`: Contains the logic for downloading the image datasets.
- `student_resource/`: The directory containing the raw datasets and images.
- `artifacts/`: The directory where trained models and cached embeddings are stored.
- `obsolete_files/`: Contains scripts from previous experimental approaches.

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

### Step 2: Download the Data

Before you can train the model or make predictions, you must download the image datasets. Use the interactive menu for this.

1.  Run the main script:
    ```bash
    python main.py
    ```
2.  The main menu will appear. Choose option `1` for "Download Datasets".
3.  You will then be prompted to download for `train`, `test`, or `all`. It is recommended to choose `all` the first time.

    *This step will take a very long time as it downloads ~150,000 images.*

### Step 3: Train the Model

Once the data is downloaded, you can train the models.

1.  Run the main script:
    ```bash
    python main.py
    ```
2.  Choose option `2` for "Train Model".

    *This will start the full training process, including generating feature embeddings (if not already cached) and training the 5 cross-validation models. This is also a time-consuming step.*

### Step 4: Generate the Submission File

After the models are trained, you can generate the `test_out.csv` file for the competition.

1.  Run the main script:
    ```bash
    python main.py
    ```
2.  Choose option `3` for "Generate Official Submission".

    *This will create the `test_out.csv` file in the root directory.*

### (Optional) Step 5: Predict on Custom Data

You can also use the trained models to make predictions on your own data.

1.  **Prepare your data:**
    *   Create a CSV file (e.g., `my_data.csv`) with the columns: `sample_id`, `catalog_content`, `image_link`.
    *   Create a directory (e.g., `my_images/`) and place all the corresponding image files in it. The name of each image file **must** match the end of its `image_link` URL.
2.  **Run the prediction:**
    *   Run `python main.py` and choose option `4`.
    *   The script will prompt you to enter the path to your custom CSV and the directory containing your images.