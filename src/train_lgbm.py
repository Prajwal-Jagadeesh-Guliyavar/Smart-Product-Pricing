import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from processing import load_config, load_and_preprocess_data, get_text_embeddings, get_image_embeddings

def lgbm_smape(y_true, y_pred):
    """Custom SMAPE evaluation metric for LightGBM."""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    numerator = np.abs(y_pred_orig - y_true_orig)
    denominator = (np.abs(y_true_orig) + np.abs(y_pred_orig)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 'SMAPE', np.mean(ratio) * 100, False

def run_training():
    """Main logic for the training pipeline."""
    config = load_config()
    paths = config['paths']
    training_params = config['training']

    # --- Pre-flight checks ---
    if not os.path.exists(paths['train_csv']):
        print(f"\nFATAL ERROR: Training data not found at '{paths['train_csv']}'")
        sys.exit(1)

    if not os.path.exists(paths['train_images']) or not os.listdir(paths['train_images']):
        print(f"\nFATAL ERROR: Training images not found in '{paths['train_images']}'")
        sys.exit(1)

    os.makedirs(paths['artifacts'], exist_ok=True)

    df = load_and_preprocess_data(paths['train_csv'], paths['train_images'], is_train=True)
    text_embeddings = get_text_embeddings(df, config)
    image_embeddings = get_image_embeddings(df, config)
    ipq_features = df['ipq'].values.reshape(-1, 1)

    X = np.concatenate([text_embeddings, image_embeddings, ipq_features], axis=1)
    y = df['log_price'].values
    print(f"\nFinal feature matrix created with shape: {X.shape}")

    kf = KFold(n_splits=training_params['n_splits'], shuffle=True, random_state=training_params['lgbm_params']['random_state'])
    oof_smape_scores = []

    print(f"Starting {training_params['n_splits']}-fold cross-validation...\n")
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold+1}/{training_params['n_splits']} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        lgbm = lgb.LGBMRegressor(**training_params['lgbm_params'])
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=[lgbm_smape, 'rmse'], callbacks=[lgb.early_stopping(100, verbose=100)])

        val_preds = lgbm.predict(X_val)
        _, fold_smape_val, _ = lgbm_smape(y_val, val_preds)
        oof_smape_scores.append(fold_smape_val)
        print(f"Fold {fold+1} SMAPE: {fold_smape_val:.4f}%")

        joblib.dump(lgbm, os.path.join(paths['artifacts'], f'lgbm_model_fold_{fold+1}.pkl'))

    print(f"\nAverage SMAPE across all folds: {np.mean(oof_smape_scores):.4f}%")
    print("Training complete. All models saved to artifacts directory.")

if __name__ == '__main__':
    run_training()
