import pandas as pd
import argparse
from datetime import datetime
import os
import sys
import csv

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils
from src.lib.colors import ColorConverter

class ScreenshotValidator:
    def __init__(self,
                 input_folder: str,
                 output_file: str):
        self.input_folder = input_folder
        self.output_file = output_file

        # Initialize utils and logger (assuming Utils is defined elsewhere)
        self.utils = Utils()  # Replace with actual initialization if needed
        self.logger = Utils.set_colorful_logging('ScreenshotValidator')  # Replace with actual logging setup

        # Prepare the output CSV file
        self.output_file = os.path.join('data', 'processed', 'screenshots.csv')
        # Create the CSV file and write the header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['website', 'screenshot', 'is_same', 
                                 'has_overlay', 'has_error', 
                                 'has_minor_diff', 'has_major_diff', 
                                 'is_shopping'])

    def get_user_input(self, prompt):
        while True:
            response = input(prompt)
            if response == '':
                return '0'  # Default to '0' if no input is provided
            elif response in ['0', '1']:
                return response
            else:
                print("Invalid input. Please enter 0 for No or 1 for Yes.")

    def run(self):
        # Check if the output CSV file exists and is not empty
        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            existing_results_df = pd.read_csv(self.output_file)
        else:
            existing_results_df = pd.DataFrame(columns=['website', 'screenshot', 'is_same', 
                                                        'has_overlay', 'has_error', 
                                                        'has_minor_diff', 'has_major_diff', 
                                                        'is_shopping'])

        # Read websites from input file
        websites_df = self.utils.get_websites()
        total_websites = len(websites_df)
        self.logger.info(f"📋 Found {total_websites} websites to process")

        # Iterate through each row in the DataFrame
        for idx, row in websites_df.iterrows():
            url = row['website']

            # Check if the website is already processed
            if url in existing_results_df['website'].values:
                self.logger.warning(f"[{idx+1}/{total_websites}] ✅ Skipping already processed website: {url}")
                continue

            clean_url = Utils.get_clean_url(url)
            screenshot_file = f'{clean_url}.png'
            screenshot_path = os.path.join(self.input_folder, screenshot_file)

            # Check if the screenshot file exists
            if not os.path.exists(screenshot_path):
                self.logger.warning(f"[{idx+1}/{total_websites}] ⚠️ Screenshot not found for URL: {url}")
                continue

            # Format the screenshot URL
            screenshot_url = f"file://{os.path.abspath(screenshot_path)}"
            self.logger.info(f"[{idx+1}/{total_websites}] {url} - {screenshot_url}")

            # Initialize variables for validation inputs
            is_same = has_overlay = has_error = has_minor_diff = has_major_diff = is_shopping = None

            while True:
                # Prompt for validation inputs
                is_same = self.get_user_input("Is the screenshot the same? (0 for No, 1 for Yes): ")
                has_overlay = self.get_user_input("Does it have an overlay? (0 for No, 1 for Yes): ")
                has_error = self.get_user_input("Does it have an error? (0 for No, 1 for Yes): ")
                has_minor_diff = self.get_user_input("Does it have minor differences? (0 for No, 1 for Yes): ")
                has_major_diff = self.get_user_input("Does it have major differences? (0 for No, 1 for Yes): ")
                is_shopping = self.get_user_input("Is this a shopping website? (0 for No, 1 for Yes): ")

                # Create a DataFrame to display the inputs
                inputs_df = pd.DataFrame({'website': [url], 'screenshot': [screenshot_file], 
                                         'is_same': [is_same], 'has_overlay': [has_overlay], 
                                         'has_error': [has_error], 'has_minor_diff': [has_minor_diff], 
                                         'has_major_diff': [has_major_diff], 'is_shopping': [is_shopping]})
                # Display the inputs DataFrame
                print(inputs_df)

                # Show the inputs and ask for confirmation
                save_option = input("Press 'e' to re-enter inputs or any other key to save: ")

                if save_option.lower() == 'e':
                    continue  # Re-enter inputs
                else:
                    # Store the results in the CSV file immediately
                    with open(self.output_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([url, screenshot_file, is_same, 
                                         has_overlay, has_error, 
                                         has_minor_diff, has_major_diff, 
                                         is_shopping])
                    break  # Exit the input loop

            print("\n" + "="*50 + "\n")

    def summary(self):
        # Generate a summary from the output file
        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            results_df = pd.read_csv(self.output_file)
            summary_df = results_df[['is_same', 'has_overlay', 'has_error', 
                                      'has_minor_diff', 'has_major_diff', 
                                      'is_shopping']].apply(pd.Series.value_counts).fillna(0)

            print("\nSummary of Features:")
            print(summary_df)
        else:
            print("No results to summarize.")

def main():
    parser = argparse.ArgumentParser(description='Process dataset and extract features')
    parser.add_argument('--input', type=str, default='./data/raw/screenshots/',
                       help='Input dataset file')
    parser.add_argument('--output', type=str, default='./data/processed/screenshots.csv',
                       help='Output processed file')
    args = parser.parse_args()

    validator = ScreenshotValidator(
        input_folder=args.input,
        output_file=args.output,
    )

    start_time = datetime.now()
    validator.run()
    duration = datetime.now() - start_time

    print("\n" + "="*50)
    print(f"📊 Extraction Summary")
    print(f"  • Processing time: {duration}")
    print("="*50 + "\n")

    # Generate and display the summary
    validator.summary()

if __name__ == "__main__":
    main()