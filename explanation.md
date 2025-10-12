# ML Challenge 2025: Smart Product Pricing - Detailed Explanation

This document provides a detailed walkthrough of the methodology, architecture, and implementation used to solve the Smart Product Pricing Challenge.

## 1. Project Goal and Strategy

**Goal:** The primary objective is to predict the price of a product using multi-modal data: unstructured text (`catalog_content`) and product images (`image_link`).

**Core Strategy:** Our approach was to build a multi-modal deep learning model using TensorFlow and Keras. The architecture processes each type of data through a separate "tower" before combining the learned features to make a final regression prediction. This allows the model to learn specialized patterns from each data type independently.

---

## 2. Phase 1: Exploratory Data Analysis (EDA)

We began our work in a Jupyter Notebook (`ML_Challenge_EDA.ipynb`) to rapidly explore and visualize the dataset.

### 2.1. Initial Data Inspection
- We loaded the `train.csv` dataset using pandas.
- An initial `.info()` check confirmed we had 75,000 entries with no missing values, which simplified our preprocessing pipeline.

### 2.2. Target Variable Analysis: `price`
- **Observation:** A histogram of the `price` column revealed a severe right-skew, with a large concentration of products at low prices and a very long tail of expensive items. The mean price was significantly higher than the median, confirming this skew.
- **Action:** Training a model on such a skewed distribution is often problematic. We applied a **logarithm transform** (`numpy.log1p`) to the price. The resulting `log_price` distribution was much closer to a normal (bell-shaped) curve. This transformed value became the actual target for our model, as it's much easier for the model to predict.

### 2.3. Input Feature Analysis: `catalog_content`
- **Observation:** By printing several full `catalog_content` entries, we identified a recurring and critical piece of information: the **Item Pack Quantity (IPQ)**. Patterns like `(Pack of 6)`, `(8 Count)`, and `12 per case` were clearly visible.
- **Action:** We determined that this IPQ was a crucial numerical feature that needed to be extracted explicitly.

---

## 3. Phase 2: Feature Engineering

This phase focused on converting raw data into numerical features suitable for a deep learning model.

### 3.1. IPQ Extraction
- We implemented a Python function `extract_ipq` that uses a series of regular expressions (regex) to search the `catalog_content` for the IPQ patterns identified during EDA.
- If a pattern was found, the corresponding number was extracted. If no pattern was found, we defaulted the IPQ to `1`, assuming the item is sold as a single unit.
- This function was applied to the dataframe to create a new numerical `ipq` column.

### 3.2. Text Cleaning
- To prepare the text for our NLP model, we created a `clean_text` function.
- This function performs several steps:
  1. Removes instructional headers like `Item Name:`, `Bullet Point 1:`, etc.
  2. Converts all text to lowercase.
  3. Removes newline characters (`\n`) and other non-alphanumeric symbols.
  4. Collapses multiple whitespace characters into a single space.
- This produced a `cleaned_text` column containing only the core descriptive text of the product.

### 3.3. Image Path Generation
- Initially, we assumed the local image filename would be `{sample_id}.jpg`. This led to a `File Not Found` error.
- By inspecting the provided `utils.py` script, we discovered that it names downloaded files based on the end of the URL (e.g., `51mo8htwTHL.jpg`).
- We corrected our pipeline to create an `image_path` column by combining the `IMAGES_DIR` with the filename extracted from the `image_link` URL.

---

## 4. Phase 3: Multi-Modal Model Architecture

We used the Keras Functional API to construct a flexible multi-input model. The architecture is defined in the `build_model` function.

### 4.1. Input Layers
- The model begins with three distinct `tf.keras.layers.Input` layers, one for each data type:
  - `image_input`: Shape `(128, 128, 3)` for the resized RGB images.
  - `text_input`: Shape `(1,)` with `dtype=tf.string` for the raw cleaned text.
  - `ipq_input`: Shape `(1,)` for the numerical IPQ feature.

### 4.2. The Image Tower (Transfer Learning)
- **Core Idea:** We used **transfer learning** to leverage a powerful, pre-existing model trained on millions of images.
- **Implementation:** 
  1. We chose `tf.keras.applications.ResNet50` with weights pre-trained on `imagenet`. (Note: `EfficientNetB0` was initially tried but had an environment-specific bug, so `ResNet50` was used as a robust alternative).
  2. We set `include_top=False` to remove the original classification layers and `trainable=False` to **freeze** the weights. This prevents the model from destroying the valuable pre-trained knowledge during initial training.
  3. The output of the ResNet base is passed through a `GlobalAveragePooling2D` layer to produce a fixed-size feature vector, which is then passed to a small `Dense` layer.

### 4.3. The Text Tower
- **`TextVectorization` Layer:** This layer is a core part of the text pipeline. It was `adapt`ed to the entire training set's text to build a vocabulary of the top 10,000 words. When called, it converts raw text strings into sequences of integer indices.
- **`Embedding` Layer:** This layer takes the integer sequences and maps each word to a dense vector (64 dimensions). This allows the model to learn relationships between words.
- **`GlobalAveragePooling1D` Layer:** This layer averages the word embeddings across the entire sequence to create a single feature vector representing the entire text.

### 4.4. The IPQ Tower
- The single `ipq` value is passed through a `tf.keras.layers.Normalization` layer. This layer scales the input to have a mean of 0 and a standard deviation of 1, which helps the model train more stably.

### 4.5. Final Assembly and Prediction
- **`Concatenate`:** The feature vectors from all three towers are merged into a single, larger vector.
- **Regression Head:** This final vector is passed through a stack of `Dense` layers with `ReLU` activation and a `Dropout` layer for regularization (to prevent overfitting). 
- **Output:** The final layer is a single `Dense` unit with a linear activation function, which outputs the predicted `log_price`.
- **Compilation:** The model is compiled with the `adam` optimizer and `mean_squared_error` loss, a standard and effective choice for regression tasks.

---

## 5. Phase 4: The `train.py` Script

To create a reproducible and scalable solution, we consolidated the entire process into a single script.

### 5.1. Data Pipeline (`tf.data`)
- For performance, we used the `tf.data.Dataset` API, which is highly efficient for loading large amounts of data.
- The `create_dataset` function takes a dataframe and creates a dataset that yields a dictionary of inputs (`image_input`, `text_input`, `ipq_input`) and the corresponding label (`log_price`).
- A crucial part of this pipeline is the `map` function, which loads and preprocesses images from their file paths **on-the-fly**. This means all 75,000 images do not need to be loaded into memory at once.
- `.batch()` and `.prefetch()` are used to further optimize data loading, ensuring the GPU never has to wait for data.

### 5.2. Main Execution Flow
- The script, when run, executes the following steps in order:
  1. **Loads and Preprocesses** the `train.csv` file.
  2. **Downloads All Images:** Iterates through the dataframe to download all 75,000 images (this is the most time-consuming step).
  3. **Adapts Text Vectorizer:** Builds the vocabulary for the `TextVectorization` layer on the full dataset.
  4. **Builds Model:** Calls the `build_model` function.
  5. **Splits Data:** Splits the dataframe into a 90% training set and a 10% validation set.
  6. **Creates Datasets:** Uses `create_dataset` to prepare the training and validation data loaders.
  7. **Trains Model:** Calls `model.fit()` to train on the full dataset.
  8. **Saves Model:** After training is complete, it saves the final trained model to disk at `saved_model/final_model.keras`.
