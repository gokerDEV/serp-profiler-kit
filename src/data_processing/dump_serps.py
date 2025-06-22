import json
import pandas as pd
import os
from typing import List, Dict, Tuple
from datetime import datetime
import argparse
import sys
 # Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils  # Now you can import Utils

from src.data_collection.site_search_bing import RESULTS_PER_PAGE as BING_RESULTS_PER_PAGE
from src.data_collection.site_search_google import RESULTS_PER_PAGE as GOOGLE_RESULTS_PER_PAGE

class SerpDumper:
    def __init__(self,  
                 input_file: str ,
                 output_file: str ,
                 invalids_file: str):
        self.input_file = input_file
        self.input_folder = 'data/raw/search_results'
        self.output_file = output_file
        self.invalids_file = invalids_file
        self.keywords = pd.read_csv(self.input_file, low_memory=False) 
        
        #  Initialize output
        self.all_results = []
        self.invalid_queries = []
        self.processed_count = 0
        self.total_results = 0
        self.unique_links = 0
        self.unique_hostnames = 0
        
        # Initialize utils and logger
        self.utils = Utils()
        self.logger = Utils.set_colorful_logging('SerpDumper')
        
    def run(self):
        """Process all Google SERP files and combine results"""
        if not os.path.exists(self.input_folder):
            self.logger.error(f"❌ Input folder not found: {self.input_folder}")
            return
        if not os.path.exists(self.input_folder+'/google'):
            self.logger.error(f"❌ Google folder not found: {self.input_folder}")
            return
        if not os.path.exists(self.input_folder+'/bing'):
            self.logger.error(f"❌ Bing folder not found: {self.input_folder}")
            return

       
        # Read keywords from input file
        for index, row in self.keywords.iterrows():
            q = row['keywords']  
            file_name = Utils.clean_filename(q)
            self.logger.info(f"{index}: Query: {q} - File: {file_name}")
            
            google_results = self._dump_google(file_name)
            bing_results = self._dump_bing(file_name)
            
            self.all_results.extend(google_results)
            self.all_results.extend(bing_results)
            self.processed_count += 1
            
            # if index > 2:
            #     break

        # Save results
        if self.all_results:
            df = pd.DataFrame(self.all_results)
            df.to_csv(self.output_file, index=False)
            self.unique_links = df['link'].nunique()
            df['hostname'] = df['link'].apply(Utils.get_hostname)
            unique_hostnames = df['hostname'].dropna().unique()
            self.unique_hostnames = len(unique_hostnames)
            if(df['hostname'].isnull().sum() > 0):
                self.logger.warning(f"❌ {df['hostname'].isnull().sum()} hostnames are null")
            self.logger.info(f"💾 Saved {len(df)} results to: {self.output_file}")

        if self.invalid_queries:
            pd.DataFrame(self.invalid_queries).to_csv(self.invalids_file, index=False)
            self.logger.info(f"⚠️ Saved {len(self.invalid_queries)} invalid queries to: {self.invalids_file}")


    def _dump_google(self, file_name: str):
        """Dump Google SERP results"""
        files = self._get_json_files(file_name, 'google')
        results = []
        for index, file_path in enumerate(files):
            # Google has start index in JSON results
            results.extend(self._process_google_file(file_path))
        return results

    def _dump_bing(self, file_name: str):
        """Dump Bing SERP results"""
        files = self._get_json_files(file_name, 'bing')
        results = []
        for index, file_path in enumerate(files):
            results.extend(self._process_bing_file(file_path, index * BING_RESULTS_PER_PAGE))
        return results

    def _get_json_files(self, base_name: str, engine: str) -> List[str]:
        """Get all available JSON files (1-5) for a given base name"""
        files = []
        for i in range(1, 6):  # Check files 1-5
            file_path = os.path.join(self.input_folder, engine, f"{base_name}_{i}.json")
            if os.path.exists(file_path):
                files.append(file_path)
        return files

    def _process_google_file(self, file_path: str) -> List[Dict]:
        """Process a single JSON file and return list of results"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
               
                query = data['queries']['request'][0]['searchTerms']
                start = data['queries']['request'][0]['startIndex']

                if int(data['searchInformation']['totalResults']) > 0:
                    results = []
                    for i, r in enumerate(data['items'], 0):
                        # print(f"URL {i}: {r['link']}")
                        results.append({
                            'engine': 'google',
                            'query': query,
                            'link': r['link'],
                            'serp_position': start + i
                        })
                    self.total_results += len(results)
                    return results
                else:
                    self.logger.error(f"❌ No results found in {file_path}")
                    self.invalid_queries.append(query)
                    return []
                
        except Exception as e:
            self.logger.error(f"❌ Error processing {file_path}: {str(e)}")
        return []


    def _process_bing_file(self, file_path: str, start: int = 1) -> List[Dict]:
        """Process a single JSON file and return list of results"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                query = data['queryContext']['originalQuery']

                if 'webPages' in data and int(data['webPages']['totalEstimatedMatches']) > 0:
                    results = []
                    for i, r in enumerate(data['webPages']['value'], 1):
                        # print(f"URL {i}: {r['url']}")
                        results.append({
                            'engine': 'bing',
                            'query': query,
                            'link': r['url'],
                            'serp_position': start + i
                        })
                    self.total_results += len(results)
                    return results
                else:
                    self.logger.error(f"❌ No results found in {file_path}")
                    self.invalid_queries.append(query)
                    return []
                
        except Exception as e:
            self.logger.error(f"❌ Error processing {file_path}: {str(e)}")
        return []


    def show_stats(self):
        """Show processing statistics"""
        print("\n" + "="*50)
        print("📊 Processing Summary:")
        
        # Calculate success rate
        success_rate = (self.processed_count / len(self.keywords) * 100) if len(self.keywords) > 0 else 0
        
        print(f"  • Total keywords in input: {len(self.keywords)}")
        print(f"  • Successfully processed: {self.processed_count}")
        print(f"  • Failed queries: {len(self.invalid_queries)}")
        print(f"  • Total results dumped: {self.total_results:,}")
        print(f"  • Unique links: {self.unique_links:,}")   
        print(f"  • Unique hostnames: {self.unique_hostnames:,}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        if self.processed_count > 0:
            print(f"  • Average results per query: {self.total_results/self.processed_count:.1f}")
        print("="*50 + "\n")




if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='Dump all Google, Bing SERP results into a single CSV file')
    parser.add_argument('--input', type=str, default='./data/raw/keywords/keywords.csv',
                       help='Input CSV file')
    parser.add_argument('--output', type=str, default='./data/processed/serps.csv',
                       help='Output CSV file')
    parser.add_argument('--invalids', type=str, default='./data/exceptions/serps.csv',
                       help='Invalid queries output file')
    args = parser.parse_args()

    dumper = SerpDumper(
        input_file=args.input,
        output_file=args.output,
        invalids_file=args.invalids,
    )
    
    start_time = datetime.now()
    dumper.run()
    dumper.show_stats()
    duration = datetime.now() - start_time
    print(f"⏱️ Processing time: {duration}")

