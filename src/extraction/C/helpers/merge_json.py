import pandas as pd
from urllib.parse import urlparse
import os
import sys
import json
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Add project root to path
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.helpers.paths import normalize_folder, resolve_path
from src.helpers.data_loader import load_valid_index

def read_json_row(row, json_root):
    """
    Reads a single JSON file based on index row and returns a row for the DataFrame.
    """
    record_id = row['record_id']
    folder = normalize_folder(row.get('folder'))
    file_name = row.get('file_name')
    
    # Resolve JSON path robustly
    json_path = resolve_path(json_root, folder, file_name, ".json")
    
    try:
        if not os.path.exists(json_path):
            # In Pull model, we report missing files as 'miss'
            return {'record_id': record_id, 'step_status': 'miss', 'step_reason': 'file_not_found'}

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Respect the status recorded in local JSON
        json_status = data.get('status', 'ok')
        
        return {
            'record_id': record_id, 
            'title': data.get('title'),
            'description': data.get('description'),
            'word_count': data.get('word_count'),
            'content_length': data.get('content_length'),
            'tag_count': data.get('tag_count'),
            'link_count': data.get('link_count'),
            'image_count': data.get('image_count'),
            'script_count': data.get('script_count'),
            'h1_count': data.get('h_counts', {}).get('h1'),
            'h2_count': data.get('h_counts', {}).get('h2'),
            'h3_count': data.get('h_counts', {}).get('h3'),
            'has_schema': data.get('has_schema'),
            'has_published_time': data.get('has_published_time'),
            'has_author': data.get('has_author'),
            'canonical': data.get('canonical'),
            'extractor_used': data.get('extractor_used'),
            'text_quality_flag': data.get('text_quality_flag'),
            'step_status': json_status,
            'step_reason': data.get('status_reason') or data.get('error')
        }
    except Exception as e:
        return {'record_id': record_id, 'step_status': 'fail', 'step_reason': str(e)}

def merge_json_files(json_root, out_file, index_path=None, max_workers=None):
    """
    Index-driven merge: Iterates over the index and pulls data from JSONs.
    Ensures that multiple records pointing to the same file are all populated.
    """
    if not index_path:
        print("Error: Merge now requires --index to correctly map records.", file=sys.stderr)
        return

    print(f"Loading index for merge: {index_path}", file=sys.stderr)
    df = load_valid_index(index_path)
    records = df.to_dict('records')
    
    print(f"Merging {len(records)} records using data from {json_root}...", file=sys.stderr)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = 100
        reader_func = partial(read_json_row, json_root=json_root)
        
        results_iterator = executor.map(reader_func, records, chunksize=chunk_size)
        
        for res in tqdm.tqdm(results_iterator, total=len(records), desc="Merging JSONs"):
            if res:
                results.append(res)
                
    # Create final DataFrame
    df_merged = pd.DataFrame(results)
    
    # Validation: Ensure no data loss against index
    if len(df_merged) < len(df):
        print(f"Warning: Merged records ({len(df_merged)}) less than index ({len(df)})", file=sys.stderr)

    # Ensure output directory exists 
    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Saving merged parquet to {out_file} (Records: {len(df_merged)})...", file=sys.stderr)
    df_merged.to_parquet(out_file, index=False)
    print("Merge complete.", file=sys.stderr)
