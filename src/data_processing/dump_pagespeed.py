import json
import pandas as pd
import os
import sys
from datetime import datetime
import argparse

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

class PageSpeedDumper:
    def __init__(self,
                 input_dir: str,
                 output_file: str,
                 exceptions_file: str):
        self.input_dir = input_dir
        self.output_file = output_file
        self.exceptions_file = exceptions_file
        
        # Initialize stats
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0
        }
        
        # Initialize utils and logger
        self.logger = Utils.set_colorful_logging('PageSpeedDumper')
        
    def run(self):
        """Process all PageSpeed files and combine results"""
        if not os.path.exists(self.input_dir):
            self.logger.error(f"❌ Input directory not found: {self.input_dir}")
            return

        all_results = []
        exceptions = []
        
        # Get list of all JSON files
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        self.stats['total_files'] = len(json_files)
        
        for file_name in json_files:
            try:
                file_path = os.path.join(self.input_dir, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract scores
                scores = {}
                if 'lighthouseResult' in data:
                    categories = data['lighthouseResult']['categories']
                    for category in ['performance', 'accessibility', 'best-practices', 'seo', 'pwa']:
                        if category in categories:
                            scores[f'{category}_score'] = categories[category]['score'] * 100
                        else:
                            scores[f'{category}_score'] = None
                
                    # Add file name (will be used to match with dataset)
                    scores['file_name'] = os.path.splitext(file_name)[0]
                    all_results.append(scores)
                    self.stats['processed'] += 1
                    
                    if len(all_results) % 100 == 0:
                        self.logger.info(f"Processed {len(all_results)} files...")
                
            except Exception as e:
                self.stats['failed'] += 1
                exception_data = {
                    "file_name": file_name,
                    "error": str(e)
                }
                exceptions.append(exception_data)
                self.logger.error(f"❌ Error processing {file_name}: {str(e)}")

        # Save results
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(self.output_file, index=False)
            self.logger.info(f"💾 Saved {len(df)} results to: {self.output_file}")

        if exceptions:
            pd.DataFrame(exceptions).to_csv(self.exceptions_file, index=False)
            self.logger.info(f"⚠️ Saved {len(exceptions)} exceptions to: {self.exceptions_file}")

    def show_stats(self):
        """Show processing statistics"""
        print("\n" + "="*50)
        print("📊 Processing Summary:")
        
        # Calculate success rate
        success_rate = (self.stats['processed'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
        
        print(f"  • Total files: {self.stats['total_files']}")
        print(f"  • Successfully processed: {self.stats['processed']}")
        print(f"  • Failed: {self.stats['failed']}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump PageSpeed scores into a single CSV file')
    parser.add_argument('--input', type=str, default='./data/raw/pagespeed',
                       help='Input directory containing PageSpeed JSON files')
    parser.add_argument('--output', type=str, default='./data/processed/pagespeed.csv',
                       help='Output CSV file')
    parser.add_argument('--exceptions', type=str, default='./data/exceptions/pagespeed_dump.csv',
                       help='Exceptions output file')
    args = parser.parse_args()

    dumper = PageSpeedDumper(
        input_dir=args.input,
        output_file=args.output,
        exceptions_file=args.exceptions
    )
    
    start_time = datetime.now()
    dumper.run()
    dumper.show_stats()
    duration = datetime.now() - start_time
    print(f"⏱️ Processing time: {duration}") 