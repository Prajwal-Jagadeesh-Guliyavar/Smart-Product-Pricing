
# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** 

---

## 1. Executive Summary
Our solution predicts product prices using a two-stage, multi-modal pipeline. We first extract high-quality numerical features from text and images using pre-trained models, then train a robust LightGBM ensemble on the combined feature set to achieve accurate and stable predictions.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
*Describe how you interpreted the pricing challenge and key insights discovered during EDA.*

**Key Observations:**
- The target variable, `price`, was heavily right-skewed, necessitating a log-transform (`log(1+p)`) to create a more normal distribution for the model to predict.
- The `catalog_content` contained crucial structured information, specifically the Item Pack Quantity (IPQ), which was a strong independent price signal.
- A significant number of product images were either missing, corrupt, or duplicated, requiring a robust data processing pipeline that could handle these imperfections gracefully.

### 2.2 Solution Strategy
*Outline your high-level approach (e.g., multimodal learning, ensemble methods, etc.)*

**Approach Type:** Ensemble of Gradient Boosted Models on Multi-Modal Embeddings.
**Core Innovation:** Instead of a single end-to-end deep learning model, we decoupled feature extraction from regression. This allowed us to use best-in-class pre-trained models for text and images while leveraging the exceptional performance of LightGBM on the resulting tabular feature set.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Our architecture is a two-stage process:
1.  **Feature Extraction:** Text, image, and structured data are processed independently to create a single, flat feature matrix.
    - `catalog_content` -> SentenceTransformer -> `Text Embedding`
    - `image_link` -> ResNet50 -> `Image Embedding`
    - `catalog_content` -> Regex Parser -> `IPQ Feature`
2.  **Regression:** The `Text Embedding`, `Image Embedding`, and `IPQ Feature` are concatenated and used to train a 5-fold LightGBM ensemble model.


### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing steps:** Lowercasing, removal of headers (e.g., "Item Name:"), and stripping of all punctuation.
- **Model type:** `sentence-transformers/all-MiniLM-L6-v2`
- **Key parameters:** Output embedding dimension: 384.

**Image Processing Pipeline:**
- **Preprocessing steps:** Images resized to 128x128, pixel values normalized using `ResNet50`'s specific `preprocess_input` function.
- **Model type:** `ResNet50` (pre-trained on ImageNet), used as a frozen feature extractor.
- **Key parameters:** Output embedding dimension: 2048 (from Global Average Pooling).


---


## 4. Model Performance

### 4.1 Validation Results
- **Primary Metric (RMSE on log_price):** Our 5-fold cross-validation yielded an average RMSE of **~0.727**.
- **Note:** The final competition metric is SMAPE, which is calculated on the hidden test set. The cross-validated RMSE on the log-transformed price was our primary internal metric for optimization.


## 5. Conclusion
Our approach successfully combines the strengths of pre-trained deep learning models for feature extraction with the regression power of gradient boosting. By creating a robust and modular pipeline, we were able to handle data imperfections and produce a high-performing ensemble model ready for submission.

---

## Appendix

### A. Code artefacts
*The complete code is contained within the project directory, orchestrated by `main.py`.*


### B. Additional Results
*No additional charts are included, as the primary results are captured by the cross-validation scores.*

---
