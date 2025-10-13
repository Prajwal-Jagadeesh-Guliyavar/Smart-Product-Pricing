import sys
import os
import pandas as pd

# Import the refactored functions from our scripts
import download_data
import train_lgbm
import predict
import perform_eda

def display_menu():
    """Prints the main menu to the console."""
    print("\n--- Smart Product Pricing Challenge Menu ---")
    print("1. Perform Data EDA")
    print("2. Download Datasets")
    print("3. Train Model")
    print("4. Generate Official Submission")
    print("5. Predict on Custom Data")
    print("6. View Project README")
    print("7. Exit")
    print("------------------------------------------")

def get_user_choice(prompt, valid_choices):
    """Generic function to get a valid user choice."""
    while True:
        choice = input(f"{prompt} {valid_choices}: ").lower().strip()
        if choice in valid_choices:
            return choice
        else:
            print(f"Invalid input. Please enter one of {valid_choices}.")

def validate_custom_csv(file_path):
    """Checks if the custom CSV has the required columns."""
    try:
        df = pd.read_csv(file_path, nrows=5) # Read only a few rows for efficiency
        required_columns = {'sample_id', 'catalog_content', 'image_link'}
        actual_columns = set(df.columns)
        if required_columns.issubset(actual_columns):
            return True
        else:
            missing = required_columns - actual_columns
            print(f"\nERROR: The CSV file is missing the following required columns: {list(missing)}")
            return False
    except Exception as e:
        print(f"\nERROR: Could not read or process the CSV file. Details: {e}")
        return False

def main():
    """Main function to run the interactive menu."""
    while True:
        display_menu()
        main_choice = get_user_choice("Enter your choice", ['1', '2', '3', '4', '5', '6', '7'])

        if main_choice == '1':
            perform_eda.run_eda()

        elif main_choice == '2':
            print("\n--- Download Menu ---")
            download_choice = get_user_choice("Download [train], [test], or [all]?", ['train', 'test', 'all'])
            download_data.run_download(download_choice)

        elif main_choice == '3':
            print("\n--- Train Model ---")
            train_lgbm.run_training()

        elif main_choice == '4':
            print("\n--- Generate Official Submission ---")
            predict.run_prediction(
                input_csv=predict.TEST_CSV,
                image_dir=predict.TEST_IMAGES_DIR,
                output_csv=predict.SUBMISSION_PATH
            )

        elif main_choice == '5':
            print("\n--- Predict on Custom Data ---")
            while True:
                input_csv = input("\nEnter the full path to your custom CSV file: ").strip()
                if os.path.exists(input_csv):
                    if validate_custom_csv(input_csv):
                        break # File exists and is valid
                    else:
                        print("Please provide a file with the correct format.")
                else:
                    print(f"Error: File not found at '{input_csv}'. Please try again.")
            
            while True:
                image_dir = input("Enter the full path to your custom images directory: ").strip()
                if os.path.exists(image_dir):
                    break
                else:
                    print(f"Error: Directory not found at '{image_dir}'. Please try again.")

            output_csv = input("Enter the desired output file path (default: custom_predictions.csv): ").strip() or "custom_predictions.csv"

            print(f"\nRunning prediction on custom data...")
            predict.run_prediction(input_csv=input_csv, image_dir=image_dir, output_csv=output_csv)

        elif main_choice == '6':
            print("\n--- Project README ---")
            if os.path.exists('README.md'):
                with open('README.md', 'r') as f:
                    print(f.read())
            else:
                print("ERROR: README.md not found.")

        elif main_choice == '7':
            print("Exiting.")
            sys.exit(0)

if __name__ == '__main__':
    main()