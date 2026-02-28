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

def investigate_similarity_outliers():
    parquet_path = "data/features/E/semantic_similarity.parquet"
    print(f"\n\n=== Investigating Similarity Outliers ===")
    print(f"Loading {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print("File not found.")
        return

    # Look for lowest similarity in content (most irrelevant content)
    # Even if they are not statistical outliers, they are semantic outliers.
    print("Lowest sim_content scores (Least relevant):")
    lowest = df.sort_values('sim_content').head(5)
    print(lowest[['record_id', 'sim_content', 'sim_title', 'sim_h1']].to_string())
    
    if lowest.empty:
        return

    # Pick the worst one
    worst_record = lowest.iloc[0]
    rid = worst_record['record_id']
    score = worst_record['sim_content']
    print(f"\nInvestigating record with lowest similarity ({score:.4f}): {rid}")
    
    # Load index to find file
    index_path = "data/index.parquet"
    extraction_root = "/Volumes/KIOXIA/extraction/"
    
    index_df = pd.read_parquet(index_path)
    record = index_df[index_df['record_id'] == rid]
    
    if record.empty:
        print("Record not found in index.")
        return
        
    folder = normalize_folder(record.iloc[0]['folder'])
    filename = record.iloc[0]['file_name']
    search_term = record.iloc[0]['search_term']
    
    print(f"Search Term: '{search_term}'")
    
    json_path = os.path.join(extraction_root, folder, f"{filename}.json")
    print(f"Checking extracted content at: {json_path}")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            title = data.get('title', '')
            content = data.get('content', '')
            
            print(f"Title: {title}")
            print(f"Content Length: {len(content)}")
            print("--- Content Snippet (First 500 chars) ---")
            print(content[:500])
    else:
        print("Extracted JSON file not found.")

if __name__ == "__main__":
    investigate_similarity_outliers()
