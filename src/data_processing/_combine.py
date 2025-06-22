import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, Optional
import os
import random
import sys
import csv

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils
from src.lib.colors import ColorConverter

class Combine:
    def __init__(self,
                 input_file: str,
                 output_folder: str,
                 exceptions_file: str,
                 target_size: int = None):
        self.input_file = input_file
        self.output_folder = output_folder
        self.exceptions_file = exceptions_file
        self.target_size = target_size

        # Initialize utils and logger
        self.utils = Utils()
        self.logger = Utils.set_colorful_logging('Combine')

    def process_dataset(self):
        """Process the entire dataset in chunks"""
        try:
            self.logger.info(f"📖 Processing dataset from: {self.input_file}")
            df = pd.read_csv(self.input_file)
            # Drop the 6th page of Bing
            # It came from the optimized api call for Bing. 
            # we get 20 results per page, so the 6th page is 51-60
            df = df[~((df['bing_position'] >= 51) & (df['bing_position'] <= 60))]
            grouped_by_website = df.groupby('website')

            labeled_links = []
            unlabeled_links = []
            exceptions = []

            total_websites = len(grouped_by_website)
            self.logger.info(f"🌐 Found {total_websites:,} unique websites in the dataset.")

            for i, (website, chunk) in enumerate(grouped_by_website):
                if len(chunk) > 0:
                    # Split into labeled and unlabeled
                    is_labeled = (chunk['google_position'] < 100) | (chunk['bing_position'] < 100)
                    labeled_links.extend(chunk[is_labeled].to_dict('records'))
                    unlabeled_links.extend(chunk[~is_labeled].to_dict('records'))

                    # Check if there are any labeled samples for this website
                    if not any(is_labeled):
                        exceptions.append({'website': website, 'reason': 'No labeled samples found'})
                        self.logger.warning(f"⚠️ No labeled samples found for website: {website}")

                # Log progress
                if (i + 1) % 100 == 0:
                    self.logger.info(f"✅ Processed {i + 1:,}/{total_websites:,} websites, {len(labeled_links) + len(unlabeled_links):,} samples...")

            self.logger.info(f"✅ Finished processing all {total_websites:,} websites.")

            # Record exceptions in the exceptions file
            if exceptions:
                pd.DataFrame(exceptions).to_csv(self.exceptions_file, index=False)
                self.logger.info(f"⚠️ Recorded {len(exceptions)} exceptions in: {self.exceptions_file}")



            # Combine
            all_links = labeled_links + unlabeled_links

            # Calculate total websites in the balanced dataset
            final_total_websites = len(set(link['website'] for link in all_links))

            # Create timestamped output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            self.logger.info(f"🔍 Found {final_total_websites:,} unique websites and {len(all_links):,} samples after excluding exceptions.")
        
            # Create separate datasets for each label column
            self.create_datasets(pd.DataFrame(all_links), timestamp)

            total_processed = len(all_links)
            return total_processed, final_total_websites, len(labeled_links)

        except Exception as e:
            self.logger.error(f"❌ Error processing dataset: {str(e)}")
            return 0, 0, 0

    def _balancing_log(self, min_count: int):
        # Check if target size is valid
        if self.target_size and self.target_size > min_count:
            self.logger.warning(f"❌ Target size {self.target_size:,} exceeds available the smallest class size {min_count:,}")
            self.target_size = min_count
            self.logger.warning(f"Target size reduced to {self.target_size:,}")

        # Determine final size
        target_size = self.target_size if self.target_size is not None else min_count
        self.logger.info(f"Balancing dataset to {target_size:,} samples per class")

    def create_datasets(self, df: pd.DataFrame, timestamp: str):
        """Create separate datasets for google_position and bing_position"""
        for label_column in ['google_position', 'bing_position']:
            self._create_two_class_datasets(df.copy(), label_column, timestamp)
            self._create_grouped_datasets(df.copy(), label_column, timestamp)
            self._create_position_datasets(df.copy(), label_column, timestamp)

    def _create_two_class_datasets(self, df: pd.DataFrame, label_column: str, timestamp: str):
        """Create datasets based on the specified label column"""
        
        df['is_labeled'] = df[label_column].apply(lambda x: 1 if x < 100 else 0)  # Assuming < 100 is labeled
        
        # Labeled and Unlabeled
        labeled = df[df[label_column] < 100]  # Assuming < 100 is labeled
        unlabeled = df[df[label_column] >= 100]  # Assuming >= 100 is unlabeled

        # Balance the datasets
        min_count = min(len(labeled), len(unlabeled))
        labeled = labeled.sample(min_count, random_state=42)
        unlabeled = unlabeled.sample(min_count, random_state=42)

        self._balancing_log(min_count)

        provider = label_column.replace('_position', '')
        file_name = f"{self.output_folder}/dataset_{provider}_two_class_{timestamp}.csv"
        merged = pd.concat([labeled, unlabeled])
        # Save labeled and unlabeled datasets
        merged.to_csv(file_name, index=False, header=True, quoting=csv.QUOTE_ALL, encoding='utf-8')
        
        self.logger.info(f"✅ {len(merged)} samples are saved from {label_column} to {file_name}")


    def _create_grouped_datasets(self, df: pd.DataFrame, label_column: str, timestamp: str):
        """Create datasets grouped by page number"""
        df['page_number'] = ((df[label_column] - 1) // 10) + 1  # Calculate page number

        # Group by page number and create datasets
        labels = df['page_number'].unique()
        min_count = len(df)

        for label in labels:
            min_count = min(min_count, len(df[df['page_number'] == label]))

        self._balancing_log(min_count)

        provider = label_column.replace('_position', '')
        file_name = f"{self.output_folder}/dataset_{provider}_paged_{timestamp}.csv"
        for label in labels:
            group = df[df['page_number'] == label]
            group = group.sample(min_count, random_state=42)
            group.to_csv(file_name, index=False, mode='a',
                         header=not os.path.exists(file_name),
                         encoding='utf-8', quoting=csv.QUOTE_ALL)

        group_size = len(labels)
        self.logger.info(f"✅ {min_count * group_size} samples and {group_size} classes from {label_column} are saved to {file_name}")


    def _create_position_datasets(self, df: pd.DataFrame, label_column: str, timestamp: str):
        """Create datasets using position directly"""
        # Balance the datasets
        df['position_number'] = df[label_column]
        labels = df['position_number'].unique()
        min_count = len(df)

        for label in labels:
            min_count = min(min_count, len(df[df[label_column] == label]))

        self._balancing_log(min_count)
        provider = label_column.replace('_position', '')
        file_name = f"{self.output_folder}/dataset_{provider}_positioned_{timestamp}.csv"
        for label in labels:
            group = df[df[label_column] == label]
            group = group.sample(min_count, random_state=42)
            group.to_csv(file_name, index=False, mode='a',
                         header=not os.path.exists(file_name),
                         encoding='utf-8', quoting=csv.QUOTE_ALL)
            
        group_size = len(labels)
        self.logger.info(f"✅ {min_count * group_size} samples and {group_size} classes from {label_column} are saved to {file_name}")

def main():
    parser = argparse.ArgumentParser(description='Combine data')
    parser.add_argument('--input', type=str, default='data/processed/labeled_links.csv',
                       help='Input dataset file')
    parser.add_argument('--output', type=str, default='data/processed/combined.csv',
                       help='Output processed file')
    parser.add_argument('--exceptions', type=str, default='data/exceptions/combine.csv',
                       help='File to store websites without labeled links')
    parser.add_argument('--target-size', type=int, default=None,
                       help='Target size per class (must not exceed available labeled links)')
    args = parser.parse_args()

    processor = Combine(
        input_file=args.input,
        output_folder=args.output,
        exceptions_file=args.exceptions,
        target_size=args.target_size
    )

    start_time = datetime.now()
    total_links, total_websites, labeled_links = processor.process_dataset()
    duration = datetime.now() - start_time

    print("\n" + "="*50)
    print(f"📊 Dataset Summary: ")
    print(f"  • Included websites: {total_websites:,}")
    print(f"  • Total links: {total_links:,}")
    print(f"  • Total labeled links (position < 100): {labeled_links:,}")
    print(f"  • Processing time: {duration}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()