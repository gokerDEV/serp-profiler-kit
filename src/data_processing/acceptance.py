import asyncio
import csv
import os
import re
from typing import List, Dict, Set
from dataclasses import dataclass
import logging
from datetime import datetime
import pandas as pd
from pyppeteer.launcher import launch
import sys
from pathlib import Path
from trafilatura import baseline
from urllib.parse import urlparse

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

# List of well-known websites to filter out
WELL_KNOWN_WEBSITES = {
    # Social Media
    'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com', 'pinterest.com',
    'reddit.com', 'linkedin.com', 'tumblr.com', 'snapchat.com',
    
    # Video/Streaming
    'youtube.com', 'vimeo.com', 'twitch.tv', 'netflix.com', 'dailymotion.com',
    
    # E-commerce
    'amazon.com', 'ebay.com', 'walmart.com', 'aliexpress.com', 'etsy.com',
    'shopify.com', 'target.com',
    
    # Tech
    'github.com', 'stackoverflow.com', 'microsoft.com', 'apple.com', 'google.com',
    
    # Content/News
    'medium.com', 'wordpress.com', 'blogger.com', 'cnn.com', 'bbc.com',
    'nytimes.com', 'wikipedia.org',
    
    # Others
    'quora.com', 'yelp.com', 'craigslist.org', 'imdb.com', 'spotify.com'
}

@dataclass
class Result:
    url: str
    file_name: str
    success: bool
    error: str = None

class DataCheck:
    def __init__(self, 
                 input_file: str,
                 input_dir: str,
                 output_file: str,
                 exceptions_file: str,
               ):
        # Initialize paths
        self.input_file = input_file
        self.input_dir = input_dir
        self.output_file = output_file
        self.exceptions_file = exceptions_file

        # Log configuration
        self._log_initialization()
        
    def get_base_domain(self, url: str) -> str:
        """Extract base domain from URL, removing subdomains"""
        try:
            parsed = urlparse(url if url.startswith('http') else f'http://{url}')
            domain_parts = parsed.netloc.split('.')
            # Handle cases like co.uk, com.br, etc.
            if len(domain_parts) > 2 and domain_parts[-2] in {'co', 'com', 'org', 'gov', 'edu'}:
                return '.'.join(domain_parts[-3:])
            return '.'.join(domain_parts[-2:])
        except:
            return None
    
    def _url_to_filename(self, url: str, dir: str = None) -> str:
        """Convert URL to a safe filename and join with directory path"""
        safe_name = Utils.url_to_safe_filename(url)
        target_dir = dir if dir is not None else self.output_dir
        return os.path.join(target_dir, safe_name)

    def run(self):
        # Initialize URLs - Read from serps.csv
        self.df = pd.read_csv(self.input_file)
        
        # Check for URL column name variations
        url_col = None
        for col in ['url', 'URL', 'link', 'Link']:
            if col in self.df.columns:
                url_col = col
                break
        
        if not url_col:
            self.logger.error(f"No URL column found in {self.input_file}. Available columns: {list(self.df.columns)}")
            raise ValueError(f"No URL column found in {self.input_file}")
        
        # Check for query column
        query_col = None
        for col in ['query', 'Query', 'keyword', 'Keyword', 'search_term', 'SearchTerm']:
            if col in self.df.columns:
                query_col = col
                break
        
        if not query_col:
            self.logger.warning(f"No query column found in {self.input_file}. Available columns: {list(df.columns)}")
            self.logger.warning("Proceeding without query information")

        # Count total entries before deduplication
        total_entries = len(self.df)     
        
        # Count duplicate URLs
        # url_counts = self.df[url_col].value_counts()
        # duplicate_urls = url_counts[url_counts > 1]
        # num_duplicate_urls = len(duplicate_urls)
        # duplicate_entries = sum(duplicate_urls) - num_duplicate_urls
        
        # self.logger.info(f"Found {num_duplicate_urls} URLs with duplicates, accounting for {duplicate_entries} duplicate entries")
        
        # Extract unique URLs and keep track of their queries
        # Use drop_duplicates to get only unique URLs with their first occurrence
        # self.df = df.drop_duplicates(subset=[url_col]).copy()
        # unique_count = len(self.df)
        
        # self.logger.info(f"Processing {unique_count} unique URLs instead of {total_entries} total entries")
        # self.logger.info(f"Deduplication reduced processing by {(total_entries - unique_count) / total_entries * 100:.1f}%")
        
        exceptions = []
        accepted = []
        self.stats = {
            'total_entries': total_entries,
            # 'total': unique_count,  # Total is now equal to unique URLs count
            # 'duplicate_urls': num_duplicate_urls,
            # 'duplicate_entries': duplicate_entries,
            'has_html': 0,
            'has_ss': 0,
            'has_content': 0,
            'is_acceptable': 0,
            'well_known_sites': 0,
            'non_coupon_sites': 0,
            'well_known_breakdown': {}
        }
        
        # Process only unique URLs
        for index, row in self.df.iterrows():
            try:
                url = row[url_col]
                query = row[query_col] if query_col and pd.notna(row[query_col]) else ""
                engine = row['engine'] if 'engine' in row else ""
                position = row['serp_position'] if 'serp_position' in row else ""
                
                # Generate proper filename using utils
                file_name = Utils.url_to_safe_filename(url)
                base_domain = self.get_base_domain(url)
                
                # Check if it's a well-known website
                if base_domain in WELL_KNOWN_WEBSITES:
                    self.stats['well_known_sites'] += 1
                    self.stats['well_known_breakdown'][base_domain] = self.stats['well_known_breakdown'].get(base_domain, 0) + 1
                    raise Exception(403, f"Well-known website detected: {base_domain}")
                
                html_file = os.path.join(self.input_dir, 'pages', f"{file_name}.html")
                if not os.path.exists(html_file):
                    raise Exception(404, "HTML file does not exist")
                self.stats['has_html'] += 1
                
                ss_file = os.path.join(self.input_dir, 'screenshots', f"{file_name}.png")
                if not os.path.exists(ss_file):
                    raise Exception(404, "Screenshot file does not exist")
                self.stats['has_ss'] += 1
                
                self.stats['has_content'] += 1
                
                html = open(html_file, 'r').read()
                
                if 'coupon' in html.lower():
                    self.stats['is_acceptable'] += 1
                    self.logger.info(f"[{index}] ✅ {url} file: {file_name}")
                    accepted.append({
                        "query": query,
                        "engine": engine,
                        "position": position,
                        "url": url, 
                        "base_domain": base_domain,
                        "file_name": file_name,
                    })
                else:
                    self.stats['non_coupon_sites'] += 1  # Count non-coupon sites
                    raise Exception(406, "Content is not a coupon website")
                
            except Exception as e:
                error_code = e.args[0] if hasattr(e, 'args') and len(e.args) >= 2 else 500
                error_msg = e.args[1] if hasattr(e, 'args') and len(e.args) >= 2 else str(e)
                
                exceptions.append({
                    "url": url if 'url' in locals() else row[url_col],
                    "code": error_code,
                    "message": error_msg,
                    "base_domain": base_domain if 'base_domain' in locals() else None,
                    "query": query if 'query' in locals() else None
                })
                self.logger.error(f"[{index}] ❌ {error_msg} for URL: {row[url_col]}")
                continue
         
        # Save results
        exceptions_df = pd.DataFrame(exceptions)
        exceptions_df.to_csv(self.exceptions_file, index=False)
        
        # NEVER write back to the input serps.csv file
        # Only write the accepted list to the output file
        accepted_df = pd.DataFrame(accepted)
        # No need to deduplicate here as we've already processed unique URLs
        accepted_df.to_csv(self.output_file, index=False)
        
        # Update hostname stats
        accepted_df['hostname'] = accepted_df['url'].apply(Utils.get_hostname)
        unique_hostnames = accepted_df['hostname'].dropna().unique()
        self.stats['hostnames'] = len(unique_hostnames)

    def show_stats(self):
        failed = self.stats['total_entries'] - (self.stats['is_acceptable'])
        success_rate = ((self.stats['has_html'] - failed) / self.stats['total_entries'] * 100) if self.stats['total_entries'] > 0 else 0
        
        print("\n" + "="*50)
        print("📊 Statistics:")
        print("\n📈 Summary:")
        print(f"  • Total entries in input file: {self.stats['total_entries']}")
        print(f"  • HTML files: {self.stats['has_html']}")
        print(f"  • Screenshots: {self.stats['has_ss']}")
        print(f"  • Content: {self.stats['has_content']}")
        print(f"  • Acceptable: {self.stats['is_acceptable']}")
        print(f"  • Hostnames: {self.stats['hostnames']}")
        print(f"  • Well-known sites filtered: {self.stats['well_known_sites']}")
        print(f"  • Non-coupon sites: {self.stats['non_coupon_sites']}")
        print(f"  • Failed: {failed}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        
        if self.stats['well_known_breakdown']:
            print("\n🌐 Well-known Sites Breakdown:")
            for domain, count in sorted(self.stats['well_known_breakdown'].items(), key=lambda x: x[1], reverse=True):
                print(f"  • {domain}: {count}")
        
        print("="*50 + "\n")

    def _log_initialization(self):
        # Setup logging
        self.logger = Utils.set_colorful_logging("Data Check")
        """Log initialization parameters"""
        params = {
            "Input File": self.input_file,
            "Input Directory": self.input_dir,
            "Exceptions File": self.exceptions_file,
        }
        
        self.logger.info("="*50)
        self.logger.info("🔧 Initialization Parameters:")
        for key, value in params.items():
            self.logger.info(f"• {key}: {value}")
        self.logger.info("="*50)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check data files')
    parser.add_argument('--input', type=str, default='data/processed/serps.csv', help='Input CSV file with search queries')
    parser.add_argument('--input_dir', type=str, default='data/raw/', help='Data directory')
    parser.add_argument('--output', type=str, default='data/processed/accepted.csv', help='Output CSV file')
    parser.add_argument('--exceptions', type=str, default='data/exceptions/data_check.csv', 
                        help='CSV file to log exceptions')
    args = parser.parse_args()

    data_check = DataCheck(      
        input_file=args.input,
        input_dir=args.input_dir,
        output_file=args.output,
        exceptions_file=args.exceptions,
    )

    data_check.run()
    data_check.show_stats()
if __name__ == "__main__":
    main()
