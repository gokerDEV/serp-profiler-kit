import asyncio
import csv
import os
import re
import json
from typing import List, Dict, Set
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from pyppeteer.launcher import launch
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

@dataclass
class Result:
    url: str
    file_name: str
    success: bool
    error: str = None

class Dataset:
    def __init__(self, 
                 features_file: str,
                 pagespeed_file: str,
                 output_file: str,
                 exceptions_file: str
               ):
        # Initialize paths
        self.features_file = features_file
        self.pagespeed_file = pagespeed_file
        self.output_file = output_file
        self.exceptions_file = exceptions_file

        # Log configuration
        self._log_initialization()
        
    def run(self):        
        # Load the features data
        self.features = pd.read_csv(self.features_file)
        
        # Load the PageSpeed data
        self.pagespeed = pd.read_csv(self.pagespeed_file)
        
        # Initialize stats
        self.stats = {
            'total_features': len(self.features),
            'total_pagespeed': len(self.pagespeed),
            'final_dataset': 0,
        }
        
        exceptions = []
        
        # Get files with PageSpeed data
        pagespeed_files = set(self.pagespeed['file_name'])
        
        # Find features entries that don't have PageSpeed data
        missing_pagespeed = ~self.features['file_name'].isin(pagespeed_files)
        missing_entries = self.features[missing_pagespeed]
        
        # Log exceptions
        for _, row in missing_entries.iterrows():
            exceptions.append({
                "url": row['url'],
                "file_name": row['file_name'],
                "message": "PageSpeed data not found"
            })
            self.logger.error(f"❌ PageSpeed data not found for URL: {row['url']}")
        
        # Save exceptions
        exceptions_df = pd.DataFrame(exceptions)
        exceptions_df.to_csv(self.exceptions_file, index=False)
        
        # Filter features to keep only entries with PageSpeed data
        dataset = self.features[~missing_pagespeed].copy()
        
        # Merge with PageSpeed data
        dataset = pd.merge(
            dataset,
            self.pagespeed,
            on='file_name',
            how='inner'  # Only keep records that exist in both dataframes
        )
        
        # Save the final dataset
        dataset.to_csv(self.output_file, index=False)
        
        # Update stats
        self.stats['final_dataset'] = len(dataset)
        
        # Calculate additional statistics
        if 'url' in dataset.columns:
            dataset['hostname'] = dataset['url'].apply(Utils.get_hostname)
            unique_hostnames = dataset['hostname'].dropna().unique()
            self.stats['hostnames'] = len(unique_hostnames)
            self.stats['unique_urls'] = len(dataset['url'].unique())
        
        if 'engine' in dataset.columns:
            # Calculate position distribution by engine
            position_counts_by_engine = {}
            
            if 'google' in dataset['engine'].values:
                self.stats['google'] = len(dataset[dataset['engine'] == 'google'])
                
                # Get Google distribution if serp_position exists
                if 'serp_position' in dataset.columns:
                    google_data = dataset[dataset['engine'] == 'google']
                    position_counts = google_data['serp_position'].value_counts().sort_index()
                    position_counts_by_engine['google'] = {f"p{pos}": count for pos, count in position_counts.items()}
                    self.stats['google_distribution'] = json.dumps(position_counts_by_engine.get('google', {}), indent=4)
            
            if 'bing' in dataset['engine'].values:
                self.stats['bing'] = len(dataset[dataset['engine'] == 'bing'])
                
                # Get Bing distribution if serp_position exists
                if 'serp_position' in dataset.columns:
                    bing_data = dataset[dataset['engine'] == 'bing']
                    position_counts = bing_data['serp_position'].value_counts().sort_index()
                    position_counts_by_engine['bing'] = {f"p{pos}": count for pos, count in position_counts.items()}
                    self.stats['bing_distribution'] = json.dumps(position_counts_by_engine.get('bing', {}), indent=4)

    def _log_initialization(self):
        # Setup logging
        self.logger = Utils.set_colorful_logging("Data Check")
        """Log initialization parameters"""
        params = {
            "Features File": self.features_file,
            "PageSpeed File": self.pagespeed_file,
            "Output File": self.output_file,
            "Exceptions File": self.exceptions_file,
        }
        
        self.logger.info("="*50)
        self.logger.info("🔧 Initialization Parameters:")
        for key, value in params.items():
            self.logger.info(f"• {key}: {value}")
        self.logger.info("="*50)

    
    def show_stats(self):
        excluded = self.stats['total_features'] - self.stats['final_dataset']
        inclusion_rate = (self.stats['final_dataset'] / self.stats['total_features'] * 100) if self.stats['total_features'] > 0 else 0
        
        print("\n" + "="*50)
        print("📊 Statistics:")
        print("\n📈 Summary:")
        print(f"  • Total entries in features file: {self.stats['total_features']}")
        print(f"  • Total entries with PageSpeed data: {self.stats['final_dataset']}")
        print(f"  • Inclusion rate: {inclusion_rate:.1f}%")
        
        if hasattr(self.stats, 'unique_urls'):
            print(f"  • Unique URLs: {self.stats['unique_urls']}")
        
        if hasattr(self.stats, 'hostnames'):
            print(f"  • Hostnames: {self.stats['hostnames']}")
        
        if hasattr(self.stats, 'google'):
            print(f"  • Google: {self.stats['google']}")
            if hasattr(self.stats, 'google_distribution'):
                print(f"  • Google Distribution: {self.stats['google_distribution']}")
        
        if hasattr(self.stats, 'bing'):
            print(f"  • Bing: {self.stats['bing']}")
            if hasattr(self.stats, 'bing_distribution'):
                print(f"  • Bing Distribution: {self.stats['bing_distribution']}")
        
        print(f"  • Excluded (no PageSpeed data): {excluded}")
        print("="*50 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Combine features with PageSpeed data')
    parser.add_argument('--features', type=str, default='data/processed/features.csv', help='Features CSV file')
    parser.add_argument('--pagespeed', type=str, default='data/processed/pagespeed.csv',
                        help='PageSpeed scores CSV file')
    parser.add_argument('--output', type=str, default='data/processed/combined.csv', help='Output CSV file')
    parser.add_argument('--exceptions', type=str, default='data/exceptions/combine.csv', 
                        help='CSV file to log exceptions')
    args = parser.parse_args()

    dataset = Dataset(  
        features_file=args.features,
        pagespeed_file=args.pagespeed,
        output_file=args.output,
        exceptions_file=args.exceptions
    )

    dataset.run()
    dataset.show_stats()
if __name__ == "__main__":
    main()
