
import sys
import os
import pandas as pd

# Import the refactored functions from our scripts
import download_data
import train_lgbm
import predict
import perform_eda

# --- Configuration ---
TRAIN_CSV = 'student_resource/dataset/train.csv'
TEST_CSV = 'student_resource/dataset/test.csv'
TRAIN_IMAGES_DIR = 'student_resource/train_images'
TEST_IMAGES_DIR = 'student_resource/test_images'
ARTIFACTS_DIR = 'artifacts'

# --- Helper & Validation Functions ---

def get_user_choice(prompt, valid_choices):
    while True:
        choice = input(f"{prompt} {valid_choices}: ").lower().strip()
        if choice in valid_choices:
            return choice
        else:
            print(f"Invalid input. Please enter one of {valid_choices}.")

def confirm_action(prompt):
    return get_user_choice(prompt, ['y', 'n']) == 'y'

def check_training_data():
    if not os.path.exists(TRAIN_CSV):
        print(f"\nWARNING: Training CSV not found at '{TRAIN_CSV}'.")
        return False
    if not os.path.exists(TRAIN_IMAGES_DIR) or not os.listdir(TRAIN_IMAGES_DIR):
        print(f"\nWARNING: Training images not found in '{TRAIN_IMAGES_DIR}'.")
        if confirm_action("Would you like to download them now? (y/n)"):
            download_data.run_download('train')
            return os.path.exists(TRAIN_IMAGES_DIR) # Re-check after download
        return False
    return True

def check_test_data():
    if not os.path.exists(TEST_CSV):
        print(f"\nWARNING: Test CSV not found at '{TEST_CSV}'.")
        return False
    if not os.path.exists(TEST_IMAGES_DIR) or not os.listdir(TEST_IMAGES_DIR):
        print(f"\nWARNING: Test images not found in '{TEST_IMAGES_DIR}'.")
        if confirm_action("Would you like to download them now? (y/n)"):
            download_data.run_download('test')
            return os.path.exists(TEST_IMAGES_DIR) # Re-check after download
        return False
    return True

def check_models_exist():
    for i in range(5):
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, f'lgbm_model_fold_{i+1}.pkl')):
            print("\nWARNING: Trained models not found.")
            print("Please run the training process first (Option 3).")
            return False
    return True

def validate_custom_csv(file_path):
    try:
        df = pd.read_csv(file_path, nrows=5)
        required_columns = {'sample_id', 'catalog_content', 'image_link'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            print(f"\nERROR: The CSV file is missing required columns: {list(missing)}")
            return False
        return True
    except Exception as e:
        print(f"\nERROR: Could not read the CSV file. Details: {e}")
        return False

# --- Main Menu Logic ---

def main():
    while True:
        print("\n--- Smart Product Pricing Challenge Menu ---")
        print("1. Perform Data EDA")
        print("2. Download Datasets")
        print("3. Train Model")
        print("4. Generate Official Submission")
        print("5. Predict on Custom Data")
        print("6. View Project README")
        print("7. Exit")
        print("------------------------------------------")
        main_choice = get_user_choice("Enter your choice", ['1', '2', '3', '4', '5', '6', '7'])

        if main_choice == '1':
            if check_training_data():
                perform_eda.run_eda()

        elif main_choice == '2':
            download_choice = get_user_choice("\nDownload [train], [test], or [all]?", ['train', 'test', 'all'])
            download_data.run_download(download_choice)

        elif main_choice == '3':
            if check_training_data():
                print("\n--- Starting Model Training ---")
                train_lgbm.run_training()

        elif main_choice == '4':
            print("\n--- Generating Official Submission ---")
            if check_test_data() and check_models_exist():
                predict.run_prediction(predict.TEST_CSV, predict.TEST_IMAGES_DIR, predict.SUBMISSION_PATH)

        elif main_choice == '5':
            print("\n--- Predict on Custom Data ---")
            if not check_models_exist():
                continue
            
            while True:
                input_csv = input("\nEnter the full path to your custom CSV file: ").strip()
                if os.path.exists(input_csv) and validate_custom_csv(input_csv):
                    break
                else:
                    print("Invalid file or format. Please check the path and column names and try again.")
            
            while True:
                image_dir = input("Enter the full path to your custom images directory: ").strip()
                if os.path.exists(image_dir):
                    break
                else:
                    print(f"Error: Directory not found at '{image_dir}'. Please try again.")

            output_csv = input("Enter desired output file path (default: custom_predictions.csv): ").strip() or "custom_predictions.csv"
            predict.run_prediction(input_csv, image_dir, output_csv)

        elif main_choice == '6':
            if os.path.exists('README.md'):
                with open('README.md', 'r') as f: print(f.read())
            else: print("ERROR: README.md not found.")

        elif main_choice == '7':
            print("Exiting.")
            sys.exit(0)

if __name__ == '__main__':
    main()
