import pandas as pd
import os
import sys
import argparse
from pathlib import Path

# --- Setup: Add src path to import the download utility ---
src_path = os.path.abspath('student_resource/src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from challenge_utils import download_images

# --- Define Paths ---
TRAIN_CSV = 'student_resource/dataset/train.csv'
TEST_CSV = 'student_resource/dataset/test.csv'
TRAIN_IMAGES_DIR = 'student_resource/train_images'
TEST_IMAGES_DIR = 'student_resource/test_images'

def download_for_set(csv_path, image_dir):
    """Reads a CSV file and downloads all images for that dataset."""
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} was not found.")
        return
        
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Starting download for {len(df)} images to '{image_dir}'...")
    os.makedirs(image_dir, exist_ok=True)
    
    image_urls = df['image_link'].tolist()
    download_images(image_urls, image_dir)
    print(f"Download process complete for {image_dir}.")

def run_download(dataset_type):
    """Main logic for downloading datasets."""
    if dataset_type == 'train' or dataset_type == 'all':
        print("--- Processing Training Set ---")
        download_for_set(TRAIN_CSV, TRAIN_IMAGES_DIR)
        
    if dataset_type == 'test' or dataset_type == 'all':
        print("\n--- Processing Test Set ---")
        download_for_set(TEST_CSV, TEST_IMAGES_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download images for the ML Challenge.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'dataset',
        choices=['train', 'test', 'all'], 
        help="Which dataset to download images for.\n'train': Download only training images.\n'test':  Download only test images.\n'all':   Download both training and test images."
    )
    args = parser.parse_args()
    run_download(args.dataset)