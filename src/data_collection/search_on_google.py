import csv
import json
import os
import re
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

import sys
 # Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils  # Import the Utils class

load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')  # Ensure you have your Google API key set in the environment
CX = os.getenv('CX')  # Load the Custom Search Engine ID from the environment
RESULTS_PER_PAGE = 10  # Constant for the number of results per page


pattern = re.compile(r'(?<!^)([ ])')

class GoogleSearch:
    def __init__(self, input_file: str, output_dir: str, max_page: int, overwrite: int, exceptions_file: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.max_page = max_page
        self.overwrite = overwrite
        self.exceptions_file = exceptions_file
        self.utils = Utils()  # Instantiate Utils
        self.logger = self.utils.set_colorful_logging("Google Custom Search")  # Set up logging
        self.keywords = pd.read_csv(self.input_file, low_memory=False)  # Load keywords as DataFrame
        self.service = build("customsearch", "v1", developerKey=API_KEY)  # Build the Google API service

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for index, row in self.keywords.iterrows():
            q = row['keywords']  
            self.logger.info(f"{index}: Query: {q}")
            
            for page in range(1, self.max_page + 1):
                filename = self.utils.clean_filename(q, page) + '.json'
                output_file_path = os.path.join(self.output_dir, filename)
                
                # Check if output file already exists
                if not self.overwrite and os.path.exists(output_file_path):
                    self.logger.warning(f"{index}: File {output_file_path} already exists. Skipping page {page}.")
                    continue  # Skip this page if the file exists

                try:
                    self.search(q, page, index)
                except Exception as ex:
                    self.log_exception(q, str(ex), index)
                time.sleep(2)  # To avoid hitting the API too quickly

    def search(self, q: str, page: int, index: int):
        
        filename = self.utils.clean_filename(q, page) + '.json'
        start_index = (page - 1) * RESULTS_PER_PAGE + 1  # Calculate the start index for the API call

        # Call the Google Custom Search API
        res = (
            self.service.cse()
            .list(
                q=q,
                cx=CX,
                start=start_index
            )
            .execute()
        )

        # Save results to JSON file
        fn = os.path.join(self.output_dir, filename)
        if self.overwrite or not os.path.exists(fn):
            with open(fn, 'w') as f:
                json.dump(res, f)
            self.logger.info(f"{index}: Saved results for {q} on page {page} to {fn}")
        else:
            self.logger.warning(f"{index}: File {fn} already exists. Use --overwrite to overwrite.")

    def log_exception(self, website: str, error_message: str, index: int):
        """Log the exception to the exceptions CSV file."""
        with open(self.exceptions_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([website, error_message])
        self.logger.error(f"{index}: Logged exception for {website}: {error_message}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search on Google Custom Search API.')
    parser.add_argument('--input', type=str, required=True, help='CSV file containing keywords')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the results as JSON')
    parser.add_argument('--max_page', type=int, default=1, help='Maximum number of pages to iterate (default is 1)')
    parser.add_argument('--overwrite', type=int, default=0, help='Overwrite existing files (default is 0)') 
    parser.add_argument('--exceptions', type=str, default='exceptions.csv', help='CSV file to save exceptions')
    args = parser.parse_args()
    
    google_search = GoogleSearch(args.input, args.output, args.max_page, args.overwrite, args.exceptions)
    google_search.run()
