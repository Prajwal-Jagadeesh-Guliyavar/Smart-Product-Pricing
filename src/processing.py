import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
import tensorflow as tf
from pathlib import Path
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_preprocess_data(data_path, image_dir, is_train=True):
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    if is_train:
        df['log_price'] = np.log1p(df['price'])

    #extraction of the items per quantity
    def extract_ipq(text):
        patterns = [r'pack of (\d+)', r'(\d+)\s*per case', r'\((\d+)\s*count\)', r'pack\s*(\d+)', r'(\d+)\s*pack']
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match: return int(match.group(1))
        return 1
    df['ipq'] = df['catalog_content'].apply(extract_ipq)

    #to clean the metadata
    def clean_text(text):
        text = re.sub(r'item name:|bullet point \d+:|value:|unit:', '', text, flags=re.IGNORECASE)
        text = text.lower().replace('\n', ' ')
        text = re.sub(r'[^a-z0-9 ]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    df['cleaned_text'] = df['catalog_content'].apply(clean_text)
    df['image_path'] = df['image_link'].apply(lambda url: os.path.join(image_dir, Path(url).name))
    return df

def get_text_embeddings(df, config, force_regenerate=False):
    text_embeddings_file = config['paths']['text_embeddings']
    if not force_regenerate and os.path.exists(text_embeddings_file):
        print(f"Loading cached text embeddings from {text_embeddings_file}...")
        return np.load(text_embeddings_file)

    print("Generating text embeddings...")
    model = SentenceTransformer(config['model']['text_embedding_model'])
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=128)
    if not force_regenerate:
        np.save(text_embeddings_file, embeddings)
    return embeddings

def safe_load_and_preprocess(path_tensor, image_size):
    path = path_tensor.numpy().decode('utf-8')
    try:
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(img_bytes, channels=3)
        if tf.shape(image)[0] == 0:
            raise ValueError("Empty image")
        image = tf.image.resize(image, image_size)
        return tf.keras.applications.resnet50.preprocess_input(image)
    except Exception:
        return tf.zeros((image_size[0], image_size[1], 3), dtype=tf.float32)

def load_image_wrapper(path, image_size):
    return tf.py_function(safe_load_and_preprocess, [path, image_size], tf.float32)

def get_image_embeddings(df, config, force_regenerate=False):
    image_embeddings_file = config['paths']['image_embeddings']
    image_size = tuple(config['model']['image_size'])

    if not force_regenerate and os.path.exists(image_embeddings_file):
        print(f"Loading cached image embeddings from {image_embeddings_file}...")
        return np.load(image_embeddings_file)

    print("Generating image embeddings...")
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    base_model.trainable = False
    image_input = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
    x = base_model(image_input, training=False)
    pooled_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(image_input, pooled_output)

    image_path_ds = tf.data.Dataset.from_tensor_slices(df['image_path'].tolist())
    image_ds = image_path_ds.map(lambda path: load_image_wrapper(path, image_size), num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)

    embeddings = feature_extractor.predict(image_ds, verbose=1)
    if not force_regenerate:
        np.save(image_embeddings_file, embeddings)
    return embeddings
