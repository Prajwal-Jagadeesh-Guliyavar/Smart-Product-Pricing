
import sys
import os

# Import the refactored functions from our scripts
import download_data
import train_lgbm
import predict

def display_menu():
    """Prints the main menu to the console."""
    print("\n--- Smart Product Pricing Challenge Menu ---")
    print("1. Download Datasets")
    print("2. Train Model")
    print("3. Generate Official Submission")
    print("4. Predict on Custom Data")
    print("5. Exit")
    print("------------------------------------------")

def get_user_choice(prompt, valid_choices):
    """Generic function to get a valid user choice."""
    while True:
        choice = input(f"{prompt} {valid_choices}: ").lower().strip()
        if choice in valid_choices:
            return choice
        else:
            print(f"Invalid input. Please enter one of {valid_choices}.")

def main():
    """Main function to run the interactive menu."""
    while True:
        display_menu()
        main_choice = get_user_choice("Enter your choice", ['1', '2', '3', '4', '5'])

        # --- 1. Download Datasets ---
        if main_choice == '1':
            print("\n--- Download Menu ---")
            download_choice = get_user_choice("Download [train], [test], or [all]?", ['train', 'test', 'all'])
            download_data.run_download(download_choice)

        # --- 2. Train Model ---
        elif main_choice == '2':
            print("\n--- Train Model ---")
            train_lgbm.run_training()

        # --- 3. Generate Official Submission ---
        elif main_choice == '3':
            print("\n--- Generate Official Submission ---")
            predict.run_prediction(
                input_csv=predict.TEST_CSV,
                image_dir=predict.TEST_IMAGES_DIR,
                output_csv=predict.SUBMISSION_PATH
            )

        # --- 4. Predict on Custom Data ---
        elif main_choice == '4':
            print("\n--- Predict on Custom Data ---")
            print("INFO: Your custom CSV file must have the following columns:")
            print("  - sample_id: A unique identifier for each item.")
            print("  - catalog_content: The text description of the product.")
            print("  - image_link: A public URL for the product image.")
            print("\nIMPORTANT: The image files must be present in the specified image directory.")
            print("The name of each image file must match the end of its corresponding 'image_link' URL.")
            print("For example, for URL 'https://a.com/image123.jpg', the file must be named 'image123.jpg'.")
            
            while True:
                input_csv = input("\nEnter the full path to your custom CSV file: ").strip()
                if os.path.exists(input_csv):
                    break
                else:
                    print(f"Error: File not found at '{input_csv}'. Please try again.")
            
            while True:
                image_dir = input("Enter the full path to your custom images directory: ").strip()
                if os.path.exists(image_dir):
                    break
                else:
                    print(f"Error: Directory not found at '{image_dir}'. Please try again.")

            output_csv = input("Enter the desired output file path (default: custom_predictions.csv): ").strip() or "custom_predictions.csv"

            print(f"\nReading data from: {input_csv}")
            print(f"Looking for images in: {image_dir}")
            print(f"Saving results to: {output_csv}\n")

            predict.run_prediction(
                input_csv=input_csv,
                image_dir=image_dir,
                output_csv=output_csv
            )

        # --- 5. Exit ---
        elif main_choice == '5':
            print("Exiting.")
            sys.exit(0)

if __name__ == '__main__':
    main()
