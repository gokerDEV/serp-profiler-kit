import csv
import json
import os
import re
import time
import requests
import argparse
from dotenv import load_dotenv
import sys
import pandas as pd
 # Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils  # Import the Utils class

load_dotenv()

API_KEY = os.getenv('AZURE_KEY')
ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
RESULTS_PER_PAGE = 20

class BingSearch:
    def __init__(self, overwrite: int, input_file: str, output_dir: str, max_page: int, exceptions_file: str):
        self.overwrite = overwrite
        self.input_file = input_file
        self.output_dir = output_dir
        self.max_page = max_page
        self.exceptions_file = exceptions_file
        self.utils = Utils()  # Instantiate Utils
        self.logger = self.utils.set_colorful_logging("Bing Search")  # Set up logging
        self.websites = pd.read_csv(self.input_file, low_memory=False)

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for index, row in self.websites.iterrows():
            website = row[0]  # Assuming the website URL is in the first column
            q = 'site:' + re.sub(r'^https?:\/\/', '', website)
            self.logger.info(f"{index}: Query: {q}")

            for page in range(1, self.max_page + 1):
                output_file_path = os.path.join(self.output_dir, f"pages_{re.sub(r'^site:', '', q)}_{page}.json")
                
                # Check if output file already exists
                if not self.overwrite and os.path.exists(output_file_path):
                    self.logger.warning(f"{index}: File {output_file_path} already exists. Skipping page {page}.")
                    continue  # Skip this page if the file exists

                try:
                    self.search(q, page, index)
                except Exception as ex:
                    self.log_exception(website, str(ex), index)
                time.sleep(2)  # To avoid hitting the API too quickly

    def search(self, q: str, page: int, index: int):
        params = {
            'q': q,
            'count': RESULTS_PER_PAGE,  # Number of results per page (max is 20)
            'offset': (page - 1) * 20,  # Offset for pagination
            'mkt': 'en-US'
        }
        headers = {'Ocp-Apim-Subscription-Key': API_KEY}

        # Call the API
        response = requests.get(ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        res = response.json()

        # Save results to JSON file
        fn = os.path.join(self.output_dir, f"{re.sub(r'^site:', '', q)}_page_{page}.json")
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
    parser = argparse.ArgumentParser(description='Search Bing for websites and save results.')
    parser.add_argument('--overwrite', type=int, default=0, help='Overwrite existing files (1 to overwrite, 0 to skip)')
    parser.add_argument('--input', type=str, required=True, help='CSV file containing website URLs')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the results as JSON')
    parser.add_argument('--max-page', type=int, default=1, help='Maximum number of pages to iterate (default is 1)')
    parser.add_argument('--exceptions', type=str, default='exceptions/search_bing.csv', help='CSV file to log exceptions')
    
    args = parser.parse_args()
    
    bing_search = BingSearch(args.overwrite, args.input, args.output, args.max_page, args.exceptions)
    bing_search.run()
