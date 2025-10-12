
import pandas as pd
import numpy as np
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import re
import os
import joblib
import tensorflow as tf
from pathlib import Path
import sys
from tqdm import tqdm

# --- 0. Setup Paths and Models ---
TEST_DATA_PATH = 'student_resource/dataset/test.csv'
SUBMISSION_PATH = 'test_out.csv'
# --- UPDATED: Point to the new test images directory ---
IMAGES_DIR = 'student_resource/test_images'
ARTIFACTS_DIR = 'artifacts'

TEXT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
IMAGE_SIZE = (128, 128)

# --- 1. Import and Preprocessing Functions (Mirrors training) ---

# Add src directory to path to import utils
src_path = os.path.abspath('student_resource/src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from challenge_utils import download_images

def load_and_preprocess_test_data(data_path):
    print("Loading and preprocessing test data...")
    df = pd.read_csv(data_path)
    
    def extract_ipq(text):
        patterns = [r'pack of (\d+)', r'(\d+)\s*per case', r'\((\d+)\s*count\)', r'pack\s*\((\d+)\)', r'(\d+)\s*pack']
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        return 1
    df['ipq'] = df['catalog_content'].apply(extract_ipq)

    def clean_text(text):
        text = re.sub(r'item name:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'bullet point \d+:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'value:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'unit:', '', text, flags=re.IGNORECASE)
        text = text.lower().replace('\n', ' ')
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df['cleaned_text'] = df['catalog_content'].apply(clean_text)
    
    df['image_path'] = df['image_link'].apply(lambda url: os.path.join(IMAGES_DIR, Path(url).name))
    
    print("Test data preprocessing complete.")
    return df

# --- 2. Feature Generation Functions (Mirrors training) ---

def generate_text_embeddings(df):
    print("Generating text embeddings for test data...")
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    text_embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=128)
    return text_embeddings

def safe_load_and_preprocess(path_tensor):
    path = path_tensor.numpy().decode('utf-8')
    try:
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(img_bytes, channels=3)
        if tf.shape(image)[0] == 0: raise ValueError("Empty image")
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image
    except Exception:
        return tf.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)

def load_image_wrapper(path):
    return tf.py_function(safe_load_and_preprocess, [path], tf.float32)

def generate_image_embeddings(df):
    print("Generating image embeddings for test data...")
    
    # NOTE: We assume images have already been downloaded by download_data.py

    # 1. Build feature extractor
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False
    image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(image_input, training=False)
    pooled_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(image_input, pooled_output, name="image_feature_extractor")

    # 2. Create dataset using the robust wrapper function
    image_path_ds = tf.data.Dataset.from_tensor_slices(df['image_path'].tolist())
    image_ds = image_path_ds.map(load_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)

    # 3. Predict
    print("Generating embeddings... This should now be robust to corrupted images.")
    image_embeddings = feature_extractor.predict(image_ds, verbose=1)
    return image_embeddings

# --- 3. Main Prediction Execution ---

if __name__ == '__main__':
    # Load and preprocess test data
    df_test = load_and_preprocess_test_data(TEST_DATA_PATH)
    
    # Generate features
    test_text_embeddings = generate_text_embeddings(df_test)
    test_image_embeddings = generate_image_embeddings(df_test)
    test_ipq_features = df_test['ipq'].values.reshape(-1, 1)
    
    # Combine features into the final test matrix
    X_test = np.concatenate([test_text_embeddings, test_image_embeddings, test_ipq_features], axis=1)
    print(f"Final test feature matrix created with shape: {X_test.shape}")
    
    # Load the 5 trained models
    models = []
    for i in range(5):
        print(f"Loading model for fold {i+1}...")
        model = joblib.load(os.path.join(ARTIFACTS_DIR, f'lgbm_model_fold_{i+1}.pkl'))
        models.append(model)
    
    # Generate predictions (ensemble by averaging)
    print("Generating predictions from the 5 models...")
    all_predictions = np.array([model.predict(X_test) for model in models])
    avg_log_predictions = np.mean(all_predictions, axis=0)
    
    # Convert log predictions back to original price scale
    final_predictions = np.expm1(avg_log_predictions)
    
    # Ensure prices are positive
    final_predictions[final_predictions < 0] = 0
    
    # Create submission file
    print(f"Creating submission file at {SUBMISSION_PATH}...")
    submission_df = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': final_predictions
    })
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission file created successfully!")
