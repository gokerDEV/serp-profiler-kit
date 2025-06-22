import pandas as pd
import argparse
import re
from typing import List
from datetime import datetime
import os
import sys
 # Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class WebsiteExtractor:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.unique_websites = set()  # Use a set to store unique website URLs

    def extract_website(self, url: str) -> str:
        """Extract the website from a given URL."""
        pattern = re.compile(r'(https?://[a-zA-Z0-9.-]+)')
        match = pattern.match(url)
        return match.group(0) if match else None

    def process_file(self):
        """Process the input CSV file and extract unique website URLs."""
        try:
            df = pd.read_csv(self.input_file)
            if 'link' not in df.columns:
                raise ValueError("Input CSV must contain a 'link' column.")
            
            # Extract unique websites from the links
            for link in df['link'].dropna().unique():
                website = self.extract_website(link)
                if website:
                    self.unique_websites.add(website)
        except Exception as e:
            print(f"❌ Error processing {self.input_file}: {str(e)}")

    def save_results(self):
        """Save the unique websites to the output file."""
        unique_websites_list = list(self.unique_websites)
        df = pd.DataFrame(unique_websites_list, columns=['website'])
        df.to_csv(self.output_file, index=False)
        print(f"💾 Saved {len(unique_websites_list)} unique websites to: {self.output_file}")

    def show_summary(self):
        """Show a summary of the results."""
        total_results = len(self.unique_websites)
        print("\n" + "="*50)
        print("📊 Summary of Website Extraction:")
        print(f"  • Total unique websites extracted: {total_results}")
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Extract unique website URLs from a CSV file.')
    parser.add_argument('--input', type=str, required=True, default='./data/processed/trends.csv',
                        help='Input CSV file containing URLs')
    parser.add_argument('--output', type=str, default='./data/processed/websites.csv',
                        help='Output CSV file for unique websites')
    args = parser.parse_args()

    extractor = WebsiteExtractor(input_file=args.input, output_file=args.output)

    start_time = datetime.now()
    extractor.process_file()
    extractor.save_results()
    extractor.show_summary()
    duration = datetime.now() - start_time
    print(f"⏱️ Processing time: {duration}")

if __name__ == "__main__":
    main() 