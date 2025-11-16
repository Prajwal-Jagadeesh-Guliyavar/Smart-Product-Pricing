import sys
import os
import pandas as pd
from processing import load_config
from download_data import run_download
from train_lgbm import run_training
from predict import run_prediction
from perform_eda import run_eda

def get_user_choice(prompt, valid_choices):
    """Gets a valid user choice from a prompt."""
    while True:
        choice = input(f"{prompt} {valid_choices}: ").lower().strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid input. Please enter one of {valid_choices}.")

def confirm_action(prompt):
    """Gets a yes/no confirmation from the user."""
    return get_user_choice(prompt, ['y', 'n']) == 'y'

def check_data(config, data_type='train'):
    """Checks for the existence of the specified dataset (train or test)."""
    paths = config['paths']
    csv_path = paths[f'{data_type}_csv']
    images_path = paths[f'{data_type}_images']

    if not os.path.exists(csv_path):
        print(f"\nWARNING: {data_type.capitalize()} CSV not found at '{csv_path}'.")
        return False
    if not os.path.exists(images_path) or not os.listdir(images_path):
        print(f"\nWARNING: {data_type.capitalize()} images not found in '{images_path}'.")
        if confirm_action("Would you like to download them now? (y/n)"):
            run_download(data_type)
            return os.path.exists(images_path)
        return False
    return True

def check_models_exist(config):
    """Checks if the trained model artifacts exist."""
    paths = config['paths']
    for i in range(config['training']['n_splits']):
        model_path = os.path.join(paths['artifacts'], f'lgbm_model_fold_{i+1}.pkl')
        if not os.path.exists(model_path):
            print("\nWARNING: Trained models not found. Please run the training process first (Option 3).")
            return False
    return True

def validate_custom_csv(file_path):
    """Validates the columns of a custom CSV file."""
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

def handle_eda(config):
    """Handler for the EDA option."""
    if check_data(config, 'train'):
        run_eda()

def handle_download(_):
    """Handler for the download option."""
    download_choice = get_user_choice("\nDownload [train], [test], or [all]?", ['train', 'test', 'all'])
    run_download(download_choice)

def handle_train(config):
    """Handler for the training option."""
    if check_data(config, 'train'):
        print("\n--- Starting Model Training ---")
        run_training()

def handle_submission(config):
    """Handler for generating the official submission."""
    print("\n--- Generating Official Submission ---")
    if check_data(config, 'test') and check_models_exist(config):
        paths = config['paths']
        run_prediction(paths['test_csv'], paths['test_images'], paths['submission'])

def handle_custom_prediction(config):
    """Handler for predicting on custom data."""
    print("\n--- Predict on Custom Data ---")
    if not check_models_exist(config):
        return

    while True:
        input_csv = input("\nEnter the full path to your custom CSV file: ").strip()
        if os.path.exists(input_csv) and validate_custom_csv(input_csv):
            break
        print("Invalid file or format. Please check the path and column names and try again.")

    while True:
        image_dir = input("Enter the full path to your custom images directory: ").strip()
        if os.path.exists(image_dir):
            break
        print(f"Error: Directory not found at '{image_dir}'. Please try again.")

    output_csv = input("Enter desired output file path (default: custom_predictions.csv): ").strip() or "custom_predictions.csv"
    run_prediction(input_csv, image_dir, output_csv)

def handle_readme(_):
    """Handler for viewing the README."""
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            print(f.read())
    else:
        print("ERROR: README.md not found.")

def main():
    """Main function to run the CLI menu."""
    # Ensure the script is run from the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    config = load_config()

    menu_options = {
        '1': ('Perform Data EDA', handle_eda),
        '2': ('Download Datasets', handle_download),
        '3': ('Train Model', handle_train),
        '4': ('Generate Official Submission', handle_submission),
        '5': ('Predict on Custom Data', handle_custom_prediction),
        '6': ('View Project README', handle_readme),
        '7': ('Exit', lambda _: sys.exit("Exiting."))
    }

    while True:
        print("\n--- Smart Product Pricing Challenge Menu ---")
        for key, (desc, _) in menu_options.items():
            print(f"{key}. {desc}")
        print("------------------------------------------")

        choice = get_user_choice("Enter your choice", menu_options.keys())
        _, handler = menu_options[choice]
        handler(config)

if __name__ == '__main__':
    main()
