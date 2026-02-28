import pandas as pd
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.helpers.paths import normalize_folder

def investigate_outlier():
    parquet_path = "data/features/D/text_readability.parquet"
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Filter for extreme outlier
    # The report showed -3568378.52
    outliers = df[df['flesch_reading_ease'] < -10000].sort_values('flesch_reading_ease')
    
    if outliers.empty:
        print("No extreme outliers found (< -10000).")
        return

    print(f"Found {len(outliers)} extreme outliers.")
    print(outliers[['record_id', 'flesch_reading_ease', 'word_count', 'sentence_count', 'char_count']].head().to_string())
    
    # Drill down into the worst one
    worst_record = outliers.iloc[0]
    rid = worst_record['record_id']
    print(f"\nInvestigating worst record: {rid}")
    
    # We need to find the file path.
    # We need the 'folder' from index or just search based on knowledge.
    # Let's load index to find the folder for this record_id
    index_path = "data/index.parquet"
    print(f"Loading index to find file path...")
    index_df = pd.read_parquet(index_path)
    
    # record_id in features came from index record_id
    record = index_df[index_df['record_id'] == rid]
    
    if record.empty:
        print("Record not found in index.")
        return
        
    folder = normalize_folder(record.iloc[0]['folder'])
    filename = record.iloc[0]['file_name']
    
    # Extracted JSON path
    # Extraction root usually /Volumes/KIOXIA/extraction/
    extraction_root = "/Volumes/KIOXIA/extraction/"
    json_path = os.path.join(extraction_root, folder, f"{filename}.json")
    
    print(f"Checking extracted content at: {json_path}")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            content = data.get('content', '')
            print(f"Content Length: {len(content)}")
            print("--- Content Snippet (First 500 chars) ---")
            print(content[:500])
            print("--- Content Snippet (Last 500 chars) ---")
            print(content[-500:])
            
            # Why is Flesch so low?
            # 206.835 - 1.015 (words/sentences) - 84.6 (syllables/words)
            # If sentences is 0? textstat handles it usually.
            # If word is huge and syllables huge?
            import textstat
            print("\n--- Re-calculating Metrics ---")
            print(f"Sentences: {textstat.sentence_count(content)}")
            print(f"Lexicon (Words): {textstat.lexicon_count(content, removepunct=True)}")
            print(f"Syllables: {textstat.syllable_count(content)}")
    else:
        print("Extracted JSON file not found.")

if __name__ == "__main__":
    investigate_outlier()
