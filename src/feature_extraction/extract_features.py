import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, Optional
import os
import glob
import sys
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils
from src.lib.colors import ColorConverter

class FeatureExtractor:
    def __init__(self,
                 input_folder: str,
                 output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Initialize utils and logger
        self.utils = Utils()
        self.logger = Utils.set_colorful_logging('FeatureExtractor')
        
    def run(self):
        dataset = glob.glob(os.path.join(self.input_folder, '*.csv'))
        for input_file in dataset:
            output_file = os.path.join(self.output_folder, os.path.basename(input_file).replace('dataset', 'extracted'))
            self.process_dataset(input_file, output_file)

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data"""
        processed_rows = []

        for _, row in chunk.iterrows():
            try:
                count = len(chunk[chunk['href'] == row['href']])
                features = self.extract_features(row, count)
                if features:  # Only add if features were successfully extracted
                    processed_rows.append(features)
            except Exception as e:
                self.logger.error(f"Error processing row: {str(e)}")
                continue

        return pd.DataFrame(processed_rows)

    def process_dataset(self, input_file: str, output_file: str):
        """Process the entire dataset in chunks"""
        try:
            self.logger.info(f"📖 Processing dataset from: {input_file}")
            
            # order,url,href,innerText,innerHTML,title,ariaLabeledText,fontWeight,underlined,isVisible,isKeyboardAccessible,x,y,width,height,viewportWidth,viewportHeight,color,actualBackgroundColor,google_position,bing_position,website

            df = pd.read_csv(input_file)
            
            grouped_by_website = df.groupby('website')
            total_websites = len(grouped_by_website)
            self.logger.info(f"🌐 Found {total_websites:,} unique websites in the dataset.")
            
            dataset = []

            for i, (website, chunk) in enumerate(grouped_by_website):
                processed_chunk = self.process_chunk(chunk)

                if len(processed_chunk) > 0:
                    dataset.extend(processed_chunk.to_dict('records'))

                # Log progress
                if (i + 1) % 100 == 0:
                    self.logger.info(f"✅ Processed {i + 1:,}/{total_websites:,} websites, {len(dataset):,} samples...")

            self.logger.info(f"✅ Finished processing all {total_websites:,} websites.")

            df = pd.DataFrame(dataset)
            df.to_csv(output_file, index=False)

            self.logger.info(f"💾 Saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"❌ Error processing dataset: {str(e)}")

    def extract_features(self, row: pd.Series, count: int) -> Optional[Dict]:
        """Extract features from a single row"""
        try:
            # Safely get values
            is_labeled = row.get('is_labeled')
            page_number = row.get('page_number')
            position_number = row.get('position_number')

            # Initialize label with a default value
            label = 0  # Default value if all are None

            # Check if is_labeled is valid
            if is_labeled is not None:
                label = int(is_labeled)
            # If is_labeled is not valid, check page_number
            elif page_number is not None:
                label = int(page_number)
            # If page_number is also not valid, check position_number
            elif position_number is not None:
                label = int(position_number)

            text = row.get('innerText') or row.get('title') or row.get('ariaLabeledText') or ''

            return {
                # LABELS
                'label': label,

                # FEATURES
                'has_text': 1 if text else 0,
                'has_html': 1 if row.get('innerText') != row.get('innerHTML') else 0,
                'is_visual': 1 if float(row.get('isVisual', 0)) == 1 else 0,
                'is_bold': 1 if float(row.get('fontWeight', 0)) >= 700 else 0,
                'is_underlined': 1 if str(row.get('underlined')) == 'True' else 0,
                'is_visible': 1 if str(row.get('isVisible')) == 'True' else 0,
                'is_keyboard_accessible': 1 if str(row.get('isKeyboardAccessible')) == 'True' else 0,
                'is_unique': 1 if count == 1 else 0,
                'in_viewport': Utils.calculate_visible_area(
                    float(row.get('x', 0)), float(row.get('y', 0)),
                    float(row.get('width', 0)), float(row.get('height', 0)),
                    float(row.get('viewportWidth', 1)), float(row.get('viewportHeight', 1))
                ),
                'text_clarity': Utils.measure_clarity(text),
                'contrast_ratio': ColorConverter.calculate_contrast_ratio(row.get('color', ''), row.get('actualBackgroundColor', '')),
                
                # CONTROL DATA
                'google_position': float(row.get('google_position', 100)),
                'bing_position': float(row.get('bing_position', 100)),
                'website': Utils.get_domain(row['href']),
                'href': row['href'],
            }
        except Exception as e:
            try:
                row_dict = row.to_dict()
                # Restructure the data for the DataFrame
                data = {'key': [], 'value': []}
                for key, value in row_dict.items():
                    data['key'].append(key)
                    data['value'].append(value)
                df = pd.DataFrame(data)
                self.logger.error(f"Error extracting features: {str(e)}\nRow data:\n{df}")
            except Exception as inner_e:
                self.logger.error(f"Error extracting features: {str(e)} | Error formatting row data: {inner_e} | Row data: {row.to_dict()}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Process dataset and extract features')
    parser.add_argument('--input', type=str, default='./data/datasets/',
                       help='Input dataset file')
    parser.add_argument('--output', type=str, default='./data/extracted/',
                       help='Output processed file')
    args = parser.parse_args()

    extractor = FeatureExtractor(
        input_folder=args.input,
        output_folder=args.output,
    )

    start_time = datetime.now()
    extractor.run()
    duration = datetime.now() - start_time

    print("\n" + "="*50)
    print(f"📊 Extraction Summary")
    print(f"  • Processing time: {duration}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
