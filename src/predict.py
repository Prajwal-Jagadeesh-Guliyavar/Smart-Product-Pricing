import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import sys

from processing import load_config, load_and_preprocess_data, get_text_embeddings, get_image_embeddings

def run_prediction(input_csv, image_dir, output_csv):
    """Runs the full prediction pipeline on a given CSV file."""
    config = load_config()
    paths = config['paths']
    
    # --- Pre-flight checks ---
    if not os.path.exists(input_csv):
        print(f"\nFATAL ERROR: Input data not found at '{input_csv}'")
        sys.exit(1)

    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"\nFATAL ERROR: Images not found in '{image_dir}'")
        sys.exit(1)

    df_test = load_and_preprocess_data(input_csv, image_dir, is_train=False)
    test_text_embeddings = get_text_embeddings(df_test, config, force_regenerate=True)
    test_image_embeddings = get_image_embeddings(df_test, config, force_regenerate=True)
    test_ipq_features = df_test['ipq'].values.reshape(-1, 1)

    X_test = np.concatenate([test_text_embeddings, test_image_embeddings, test_ipq_features], axis=1)
    print(f"Final test feature matrix created with shape: {X_test.shape}")

    models = []
    for i in range(config['training']['n_splits']):
        model_path = os.path.join(paths['artifacts'], f'lgbm_model_fold_{i+1}.pkl')
        if not os.path.exists(model_path):
            print(f"\nFATAL ERROR: Trained model not found at {model_path}. Please run the training script first.")
            sys.exit(1)
        print(f"Loading model for fold {i+1}...")
        models.append(joblib.load(model_path))

    print(f"Generating predictions from the {len(models)} models...")
    all_predictions = np.array([model.predict(X_test) for model in models])
    avg_log_predictions = np.mean(all_predictions, axis=0)

    final_predictions = np.expm1(avg_log_predictions)
    final_predictions[final_predictions < 0] = 0

    submission_df = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': final_predictions})

    if len(submission_df) != len(df_test):
        print(f"\nFATAL ERROR: Output length ({len(submission_df)}) does not match input length ({len(df_test)}).")
        sys.exit(1)

    print(f"\nSuccessfully generated {len(submission_df)} predictions.")
    print(f"Creating submission file at {output_csv}...")
    submission_df.to_csv(output_csv, index=False)
    print("Submission file created successfully!")

if __name__ == '__main__':
    config = load_config()
    paths = config['paths']
    run_prediction(input_csv=paths['test_csv'], image_dir=paths['test_images'], output_csv=paths['submission'])