import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, Optional
import os
import glob
import sys
import cv2
import hashlib
import numpy as np
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils
from src.lib.colors import ColorConverter

class ImageExtractor:
    def __init__(self,
                 input_folder: str,
                 output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Initialize utils and logger
        self.utils = Utils()
        self.logger = Utils.set_colorful_logging('ImageExtractor')
        
    def run(self):
        dataset = glob.glob(os.path.join(self.input_folder, '*.csv'))
        for input_file in dataset:
            output_file = os.path.join(self.output_folder, os.path.basename(input_file).replace('dataset', 'extracted_img'))
            self.process_dataset(input_file, output_file)

    def process_chunk(self, website: str, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data"""
        processed_rows = []
        
        # try:    
        #     screenshot_path = os.path.join('data/raw/screenshots', f'{Utils.get_clean_url(website)}.png')
        #     screenshot = cv2.imread(screenshot_path, cv2.IMREAD_UNCHANGED)  # Load the image
        #     if screenshot is None or screenshot.size == 0:
        #         self.logger.error(f"Failed to load screenshot: {screenshot_path}")
        #         return pd.DataFrame(processed_rows)
        # except Exception as e:
        #     self.logger.error(f"Error reading screenshot: {str(e)}")
        #     return pd.DataFrame(processed_rows)

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
                processed_chunk = self.process_chunk(website, chunk)

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
        try:
            # Safely get values
            is_labeled = row.get('is_labeled')
            page_number = row.get('page_number')
            position_number = row.get('position_number')
            website = Utils.get_clean_url(Utils.get_domain(row['href']))

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

            try:
                screenshot_path = os.path.join('data/raw/screenshots', f'{Utils.get_clean_url(website)}.png')
                screenshot = cv2.imread(screenshot_path, cv2.IMREAD_UNCHANGED)  # Load the image
                if screenshot is None or screenshot.size == 0:
                    self.logger.error(f"Failed to load screenshot: {screenshot_path}")
                    return None

                # Convert the image from RGBA to RGB if it has 4 channels
                if screenshot.shape[2] == 4:
                    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2RGB)

                ss_width = screenshot.shape[1]
                ss_height = screenshot.shape[0]
                
                # Ensure that the coordinates are integers
                x1 = int(row.get('x', 0))
                y1 = int(row.get('y', 0))
                x2 = int(row.get('x', 0) + row.get('width', 0))
                y2 = int(row.get('y', 0) + row.get('height', 0))

                # Validate cropping coordinates
                if x1 < 0 or y1 < 0 or x2 > ss_width or y2 > ss_height:
                    self.logger.error(f"Invalid cropping coordinates: ({x1}, {y1}, {x2}, {y2}) for image size: ({ss_width}, {ss_height})")
                    return None

                # Create a mask with the same dimensions as the screenshot
                mask = np.zeros((ss_height, ss_width), dtype="uint8")  # Create a mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill link areas in mask

                # Create a flat color image (e.g., gray) with the same number of channels as the screenshot
                flat_color = (31, 212, 0)  # Example gray color
                flat_image = np.full((ss_height, ss_width, 3), flat_color, dtype=np.uint8)  # Create flat color image

                # Keep only the area defined by the mask from the original screenshot
                img = cv2.bitwise_and(screenshot, screenshot, mask=mask)  # Keep only the area in the mask

                # Invert the mask to create a blank area
                blank_mask = cv2.bitwise_not(mask)

                # Create an image filled with the flat color for the unmasked area
                flat_area = cv2.bitwise_and(flat_image, flat_image, mask=blank_mask)

                # Combine the masked area with the flat color area
                masked_image = cv2.add(img, flat_area)  # Combine the original area with the flat color area
                
                # Check if masked_image is empty
                if masked_image is None or masked_image.size == 0:
                    self.logger.error("Masked image is empty.")
                    return None

                # Convert the masked image to bytes and calculate the hash value
                image_bytes = masked_image.tobytes()  # Convert the numpy array to bytes
                hash_value = hashlib.sha256(image_bytes).hexdigest()
                
                # Save the processed image
                masked_image_name =  f'{website}_masked_{hash_value}.png'
                masked_image_path = os.path.join('data/processed/masked/', masked_image_name)
                cv2.imwrite(masked_image_path, masked_image)  # Save the masked image

                # # Save the unmasked area separately
                # unmasked_area = cv2.bitwise_and(screenshot, screenshot, mask=blank_mask)  # Extract the unmasked area
                # unmasked_image_path = os.path.join('data/processed/images/', f'{website}_unmasked_{hash_value}.png')
                # cv2.imwrite(unmasked_image_path, unmasked_area)  # Save the unmasked area

                # Crop the specified area from the screenshot
                cropped_image = screenshot[y1:y2, x1:x2]  # Crop the image using the coordinates
                if cropped_image.size == 0:
                    self.logger.error("Cropped area is empty.")
                    return None
                
                  # Convert the masked image to bytes and calculate the hash value
                image_bytes = cropped_image.tobytes()  # Convert the numpy array to bytes
                hash_value = hashlib.sha256(image_bytes).hexdigest()
                cropped_image_name =  f'{website}_cropped_{hash_value}.png'
                cropped_image_path = os.path.join('data/processed/cropped/', cropped_image_name)
                cv2.imwrite(cropped_image_path, cropped_image)  # Save the cropped area

            except Exception as e:
                self.logger.error(f"Error image processing: {row['href']} | {str(e)}")
                return None
            
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
                
                # ADDITIONAL DATA
                'ss_width': ss_width,
                'ss_height': ss_height,
                'x': float(row.get('x', 0)),
                'y': float(row.get('y', 0)),
                'width': float(row.get('width', 0)),
                'height': float(row.get('height', 0)),
                'viewport_width': float(row.get('viewportWidth', 1)),
                'viewport_height': float(row.get('viewportHeight', 1)),
                
                # IMAGES
                'masked_image': masked_image_name,
                'cropped_image': cropped_image_name,
                
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

    extractor = ImageExtractor(
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
