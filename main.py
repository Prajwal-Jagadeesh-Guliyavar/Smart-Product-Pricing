import argparse
import sys
import os

# Import the refactored functions from our scripts
import download_data
import train_lgbm
import predict

def main():
    parser = argparse.ArgumentParser(
        description="Main menu for the Smart Product Pricing Challenge.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Download Command ---
    parser_download = subparsers.add_parser('download', help='Download image datasets.')
    parser_download.add_argument('dataset', choices=['train', 'test', 'all'], help='Which dataset to download.')

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help='Run the full training pipeline.')

    # --- Predict Command ---
    parser_predict = subparsers.add_parser('predict', help='Generate predictions for the official test set.')

    # --- Predict-Custom Command ---
    parser_custom = subparsers.add_parser(
        'predict-custom', 
        help='Generate predictions for a custom CSV file.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser_custom.add_argument('input_csv', type=str, help='Path to the custom input CSV file.')
    parser_custom.add_argument('image_dir', type=str, help='Path to the directory containing images for the custom CSV.')
    parser_custom.add_argument('--output_csv', type=str, default='custom_predictions.csv', help='Path to save the output predictions.')

    args = parser.parse_args()

    # --- Execute Commands ---
    if args.command == 'download':
        download_data.run_download(args.dataset)

    elif args.command == 'train':
        train_lgbm.run_training()

    elif args.command == 'predict':
        predict.run_prediction(
            input_csv=predict.TEST_CSV,
            image_dir=predict.TEST_IMAGES_DIR,
            output_csv=predict.SUBMISSION_PATH
        )

    elif args.command == 'predict-custom':
        print("--- Running Prediction on Custom Data ---")
        print("INFO: Your custom CSV file must have the following columns:")
        print("  - sample_id: A unique identifier for each item.")
        print("  - catalog_content: The text description of the product.")
        print("  - image_link: A public URL for the product image.")
        print("\nIMPORTANT: The image files must be present in the specified image directory.")
        print("The name of each image file must match the end of its corresponding 'image_link' URL.")
        print("For example, for URL 'https://a.com/image123.jpg', the file must be named 'image123.jpg'.")
        
        print(f"\nReading data from: {args.input_csv}")
        print(f"Looking for images in: {args.image_dir}")
        print(f"Saving results to: {args.output_csv}\n")
            
        predict.run_prediction(
            input_csv=args.input_csv,
            image_dir=args.image_dir,
            output_csv=args.output_csv
        )

if __name__ == '__main__':
    main()