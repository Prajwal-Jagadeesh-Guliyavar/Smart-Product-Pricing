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

# --- 0. Configuration ---
TEST_CSV = 'student_resource/dataset/test.csv'
TEST_IMAGES_DIR = 'student_resource/test_images'
SUBMISSION_PATH = 'test_out.csv'
ARTIFACTS_DIR = 'artifacts'
TEXT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
IMAGE_SIZE = (128, 128)

# --- 1. Preprocessing and Feature Generation Functions ---

# (These functions are copied from train_lgbm.py for consistency)

def load_and_preprocess_data(data_path, image_dir):
    print(f"Loading and preprocessing data from {data_path}...")
    df = pd.read_csv(data_path)
    
    def extract_ipq(text):
        patterns = [r'pack of (\d+)', r'(\d+)\s*per case', r'\((\d+)\s*count\)', r'pack\s*\((\d+)\)', r'(\d+)\s*pack']
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match: return int(match.group(1))
        return 1
    df['ipq'] = df['catalog_content'].apply(extract_ipq)

    def clean_text(text):
        text = re.sub(r'item name:|bullet point \d+:|value:|unit:', '', text, flags=re.IGNORECASE)
        text = text.lower().replace('\n', ' ')
        text = re.sub(r'[^a-z0-9 ]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    df['cleaned_text'] = df['catalog_content'].apply(clean_text)
    df['image_path'] = df['image_link'].apply(lambda url: os.path.join(image_dir, Path(url).name))
    return df

def generate_text_embeddings(df):
    print("Generating text embeddings...")
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    return model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=128)

def safe_load_and_preprocess(path_tensor):
    path = path_tensor.numpy().decode('utf-8')
    try:
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(img_bytes, channels=3)
        if tf.shape(image)[0] == 0: raise ValueError("Empty image")
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.keras.applications.resnet50.preprocess_input(image)
    except Exception: return tf.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)

def load_image_wrapper(path):
    return tf.py_function(safe_load_and_preprocess, [path], tf.float32)

def generate_image_embeddings(df):
    print("Generating image embeddings...")
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False
    image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(image_input, training=False)
    pooled_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(image_input, pooled_output)

    image_path_ds = tf.data.Dataset.from_tensor_slices(df['image_path'].tolist())
    image_ds = image_path_ds.map(load_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)
    return feature_extractor.predict(image_ds, verbose=1)

# --- 2. Main Prediction Logic ---

def run_prediction(input_csv, image_dir, output_csv):
    """Runs the full prediction pipeline on a given CSV file."""
    # Load and preprocess data
    df_test = load_and_preprocess_data(input_csv, image_dir)
    
    # Generate features
    test_text_embeddings = generate_text_embeddings(df_test)
    test_image_embeddings = generate_image_embeddings(df_test)
    test_ipq_features = df_test['ipq'].values.reshape(-1, 1)
    
    # Combine features
    X_test = np.concatenate([test_text_embeddings, test_image_embeddings, test_ipq_features], axis=1)
    print(f"Final test feature matrix created with shape: {X_test.shape}")
    
    # Load models
    models = []
    for i in range(5):
        model_path = os.path.join(ARTIFACTS_DIR, f'lgbm_model_fold_{i+1}.pkl')
        if not os.path.exists(model_path):
            print(f"Error: Trained model not found at {model_path}. Please run the training script first.")
            return
        print(f"Loading model for fold {i+1}...")
        models.append(joblib.load(model_path))
    
    # Predict and average
    print("Generating predictions from the 5 models...")
    all_predictions = np.array([model.predict(X_test) for model in models])
    avg_log_predictions = np.mean(all_predictions, axis=0)
    
    # Post-process and save
    final_predictions = np.expm1(avg_log_predictions)
    final_predictions[final_predictions < 0] = 0
    
    print(f"Creating submission file at {output_csv}...")
    submission_df = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': final_predictions})
    submission_df.to_csv(output_csv, index=False)
    print("Submission file created successfully!")

if __name__ == '__main__':
    # This runs the prediction for the official competition test set
    run_prediction(input_csv=TEST_CSV, image_dir=TEST_IMAGES_DIR, output_csv=SUBMISSION_PATH)