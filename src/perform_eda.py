import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration ---
TRAIN_CSV = 'student_resource/dataset/train.csv'
ARTIFACTS_DIR = 'artifacts'
EDA_PLOT_PATH = os.path.join(ARTIFACTS_DIR, 'eda_price_distribution.png')

def run_eda():
    """Generates and saves a plot of the price distribution if it doesn't exist."""
    print("\n--- Exploratory Data Analysis ---")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if os.path.exists(EDA_PLOT_PATH):
        print(f"EDA plot already exists at '{EDA_PLOT_PATH}'.")
        print("Please open the image file to view the analysis.")
        return

    print("Generating new EDA plot...")
    df = pd.read_csv(TRAIN_CSV)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot original price distribution
    sns.histplot(df['price'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribution of Price (Original)', fontsize=16)
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Frequency')

    # Plot log-transformed price distribution
    sns.histplot(np.log1p(df['price']), bins=50, kde=True, ax=ax2, color='green')
    ax2.set_title('Distribution of Price (Log-Transformed)', fontsize=16)
    ax2.set_xlabel('Log(1 + Price)')
    ax2.set_ylabel('Frequency')

    fig.suptitle('Price Distribution Analysis', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(EDA_PLOT_PATH)
    print(f"EDA plot saved successfully to '{EDA_PLOT_PATH}'.")

if __name__ == '__main__':
    run_eda()