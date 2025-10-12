
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# --- 0. Configuration ---
TRAIN_CSV = 'student_resource/dataset/train.csv'
TRAIN_IMAGES_DIR = 'student_resource/train_images'
ARTIFACTS_DIR = 'artifacts'

TEXT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TEXT_EMBEDDINGS_FILE = os.path.join(ARTIFACTS_DIR, 'text_embeddings.npy')
IMAGE_EMBEDDINGS_FILE = os.path.join(ARTIFACTS_DIR, 'image_embeddings.npy')
IMAGE_SIZE = (128, 128)

# --- 1. Add src to Path to Import Utils ---
src_path = os.path.abspath('student_resource/src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from challenge_utils import download_images

# --- 2. Preprocessing and Feature Generation Functions ---

def load_and_preprocess_data(data_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    df['log_price'] = np.log1p(df['price'])

    def extract_ipq(text):
        patterns = [r'pack of (\d+)', r'(\d+)\s*per case', r'\((\d+)\s*count\)', r'pack\s*(\d+)', r'(\d+)\s*pack']
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
    df['image_path'] = df['image_link'].apply(lambda url: os.path.join(TRAIN_IMAGES_DIR, Path(url).name))
    return df

def get_text_embeddings(df):
    if os.path.exists(TEXT_EMBEDDINGS_FILE):
        print(f"Loading cached text embeddings from {TEXT_EMBEDDINGS_FILE}...")
        return np.load(TEXT_EMBEDDINGS_FILE)
    
    print("Generating text embeddings...")
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=128)
    np.save(TEXT_EMBEDDINGS_FILE, embeddings)
    return embeddings

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

def get_image_embeddings(df):
    if os.path.exists(IMAGE_EMBEDDINGS_FILE):
        print(f"Loading cached image embeddings from {IMAGE_EMBEDDINGS_FILE}...")
        return np.load(IMAGE_EMBEDDINGS_FILE)

    print("Generating image embeddings...")
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False
    image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(image_input, training=False)
    pooled_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(image_input, pooled_output)

    image_path_ds = tf.data.Dataset.from_tensor_slices(df['image_path'].tolist())
    image_ds = image_path_ds.map(load_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)
    
    embeddings = feature_extractor.predict(image_ds, verbose=1)
    np.save(IMAGE_EMBEDDINGS_FILE, embeddings)
    return embeddings

def run_training():
    """Main logic for the training pipeline."""
    # --- Pre-flight checks ---
    if not os.path.exists(TRAIN_CSV):
        print(f"\nFATAL ERROR: Training data not found at '{TRAIN_CSV}'")
        print("Please ensure the file exists before running training.")
        sys.exit(1)
        
    if not os.path.exists(TRAIN_IMAGES_DIR) or not os.listdir(TRAIN_IMAGES_DIR):
        print(f"\nFATAL ERROR: Training images not found in '{TRAIN_IMAGES_DIR}'")
        print("Please run the following command to download the images first:")
        print("    python main.py download train")
        sys.exit(1)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Load and process data
    df = load_and_preprocess_data(TRAIN_CSV)
    
    # Get features
    text_embeddings = get_text_embeddings(df)
    image_embeddings = get_image_embeddings(df)
    ipq_features = df['ipq'].values.reshape(-1, 1)
    
    # Combine features
    X = np.concatenate([text_embeddings, image_embeddings, ipq_features], axis=1)
    y = df['log_price'].values
    print(f"\nFinal feature matrix created with shape: {X.shape}")

    # Cross-validation training
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_scores = []

    print(f"Starting {N_SPLITS}-fold cross-validation...\n")
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31, n_jobs=-1)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_preds = lgbm.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        oof_scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse}")
        
        joblib.dump(lgbm, os.path.join(ARTIFACTS_DIR, f'lgbm_model_fold_{fold+1}.pkl'))

    print(f"\nAverage RMSE across all folds: {np.mean(oof_scores)}")
    print("Training complete. All models saved to artifacts directory.")

if __name__ == '__main__':
    run_training()
