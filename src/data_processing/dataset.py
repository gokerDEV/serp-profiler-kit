import pandas as pd
import sys
import os
from datetime import datetime
import argparse

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

class DatasetCreator:
    def __init__(self,
                 input_file: str,
                 output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.logger = Utils.set_colorful_logging('Dataset Creator')
        
        # Initialize stats
        self.stats = {
            'initial_rows': 0,
            'removed_rows': 0,
            'final_rows': 0,
            'removed_columns': [],
            'missing_lighthouse_rows': 0
        }
        
    def run(self):
        """Clean the dataset"""
        # Read the dataset
        self.logger.info("Reading dataset...")
        df = pd.read_csv(self.input_file)
        self.stats['initial_rows'] = len(df)
        
        # Extract hostname from link
        self.logger.info("Extracting hostnames...")
        if 'link' in df.columns:
            df['hostname'] = df['link'].apply(lambda x: Utils.get_hostname(x) if pd.notnull(x) else None)
        
        # Remove PWA score column if exists
        if 'pwa_score' in df.columns:
            self.logger.info("Removing PWA score column...")
            df = df.drop('pwa_score', axis=1)
            self.stats['removed_columns'].append('pwa_score')
        
        # Count rows with missing Lighthouse data
        lighthouse_columns = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        missing_lighthouse = df[lighthouse_columns].isnull().any(axis=1)
        self.stats['missing_lighthouse_rows'] = missing_lighthouse.sum()
        
        # Remove rows with missing Lighthouse data
        self.logger.info("Removing rows with missing Lighthouse data...")
        df_cleaned = df[~missing_lighthouse]
        
        # Calculate removed rows
        self.stats['removed_rows'] = len(df) - len(df_cleaned)
        self.stats['final_rows'] = len(df_cleaned)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Save cleaned dataset
        self.logger.info("Saving cleaned dataset...")
        df_cleaned.to_csv(self.output_file, index=False)
        
    def show_summary(self):
        """Display a summary of the cleaning process"""
        print("\n" + "="*50)
        print("🧹 Dataset Summary")
        
        print("\n📊 Row Statistics:")
        print(f"  • Initial rows: {self.stats['initial_rows']:,}")
        print(f"  • Removed rows: {self.stats['removed_rows']:,}")
        print(f"  • Final rows: {self.stats['final_rows']:,}")
        
        print("\n🔍 Cleaning Details:")
        print(f"  • Removed columns: {', '.join(self.stats['removed_columns'])}")
        print(f"  • Rows with missing Lighthouse data: {self.stats['missing_lighthouse_rows']:,}")
        
        if self.stats['initial_rows'] > 0:
            retention_rate = (self.stats['final_rows'] / self.stats['initial_rows']) * 100
            print(f"\n📈 Data Retention Rate: {retention_rate:.1f}%")
        
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Clean dataset by removing PWA scores and incomplete rows')
    parser.add_argument('--input', type=str, default='data/processed/combined.csv',
                       help='Input dataset CSV file')
    parser.add_argument('--output', type=str, default='data/datasets/dataset.csv',
                       help='Output cleaned dataset CSV file')
    args = parser.parse_args()

    creator = DatasetCreator(
        input_file=args.input,
        output_file=args.output
    )
    
    start_time = datetime.now()
    creator.run()
    creator.show_summary()
    duration = datetime.now() - start_time
    print(f"⏱️ Processing time: {duration}")

if __name__ == "__main__":
    main() 