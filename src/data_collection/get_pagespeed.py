import os
import pandas as pd
import sys
import json
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import time
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

load_dotenv()

API_KEY = os.getenv('GOOGLE_PAGESPEED_API_KEY')
domains = ['pinterest.com']

class GetPagespeed:
    def __init__(self, 
                 input_file: str,
                 output_dir: str,
                 exceptions_file: str,
               ):
        # Initialize paths
        self.input_file = input_file
        self.output_dir = output_dir
        self.exceptions_file = exceptions_file

        # Log configuration
        self._log_initialization()
        
        try:
            self.service = build("pagespeedonline", "v5", developerKey=API_KEY, static_discovery=False)
        except Exception as e:
            self.logger.error(f"Failed to initialize Pagespeed API: {str(e)}")
            raise
        
    def run(self):        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.urls = pd.read_csv(self.input_file)
        print(len(self.urls))
        # Remove duplicates from accepted dataset, keeping first occurrence
        self.urls = self.urls.drop_duplicates(subset=['url'], keep='first')
        print(len(self.urls))
        
        # Initialize stats
        self.stats = {
            'total': len(self.urls),
            'success': 0,
            'failed': 0
        }
        
        self.urls['file_name'] = self.urls['file_name'].str.replace('.csv', '')
        
        # Create exceptions DataFrame with headers
        if not os.path.exists(self.exceptions_file):
            pd.DataFrame(columns=['url', 'code', 'message']).to_csv(self.exceptions_file, index=False)
        
        for index, row in self.urls.iterrows():
            try:
                domain_pattern = '|'.join(map(re.escape, domains))
                if re.search(domain_pattern, row['url'], re.IGNORECASE):
                    self.logger.warning(f"[{index}] ❌ {row['url']} - matched domain pattern")
                    self.stats['failed'] += 1
                    continue
                
                file_path = os.path.join(self.output_dir,f"{row['file_name']}.json")
                if os.path.exists(file_path):
                    self.logger.info(f"[{index}] ✅ {row['url']} - already exists")
                    self.stats['success'] += 1
                    continue
                
                # Make the API call
                categories_to_include = ['ACCESSIBILITY', 'BEST_PRACTICES', 'PERFORMANCE', 'PWA', 'SEO']
                result = self.service.pagespeedapi().runpagespeed(url=row['url'], category=categories_to_include).execute()
                
                # Save the result as json file
                with open(file_path, "w") as f:
                    json.dump(result, f)
                
                self.logger.info(f"[{index}] ✅ {row['url']}")
                self.stats['success'] += 1
                
            except HttpError as e:
                error_msg = f"HTTP Error: {e.resp.status} {e.content.decode()}"
                exception_data = {"url": row['url'], "code": e.resp.status, "message": error_msg}
                pd.DataFrame([exception_data]).to_csv(self.exceptions_file, mode='a', header=False, index=False)
                self.logger.error(f"[{index}] ❌ {row['url']} - {error_msg}")
                self.stats['failed'] += 1
                
            except Exception as e:
                exception_data = {"url": row['url'], "code": 500, "message": str(e)}
                pd.DataFrame([exception_data]).to_csv(self.exceptions_file, mode='a', header=False, index=False)
                self.logger.error(f"[{index}] ❌ {row['url']} - {str(e)}")
                self.stats['failed'] += 1
                
            time.sleep(2)  # Rate limiting

    def _log_initialization(self):
        # Setup logging
        self.logger = Utils.set_colorful_logging("Pagespeed")
        """Log initialization parameters"""
        params = {
            "Input File": self.input_file,
            "Output Dir": self.output_dir,
            "Exceptions File": self.exceptions_file,
        }
        
        self.logger.info("="*50)
        self.logger.info("🔧 Initialization Parameters:")
        for key, value in params.items():
            self.logger.info(f"• {key}: {value}")
        self.logger.info("="*50)

    def show_stats(self):
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        
        print("\n" + "="*50)
        print("📊 Statistics:")
        print("\n📈 Summary:")
        print(f"  • Total items in input file: {self.stats['total']}")
        print(f"  • Success: {self.stats['success']}")
        print(f"  • Failed: {self.stats['failed']}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        print("="*50 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check data files')
    parser.add_argument('--input', type=str, default='data/processed/accepted.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='data/raw/pagespeed', help='Output folder')
    parser.add_argument('--exceptions', type=str, default='data/exceptions/pagespeed.csv', 
                        help='CSV file to log exceptions')
    args = parser.parse_args()

    get_pagespeed = GetPagespeed(  
        input_file=args.input,
        output_dir=args.output,
        exceptions_file=args.exceptions,
    )

    get_pagespeed.run()
    get_pagespeed.show_stats()
if __name__ == "__main__":
    main()
