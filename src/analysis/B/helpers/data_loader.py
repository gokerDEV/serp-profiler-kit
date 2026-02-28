import pandas as pd
import numpy as np
import os
import sys

# Default paths, can be overridden if needed
FEATURE_PATHS = {
    'A': 'data/features/A/runtime_metrics.parquet',
    'B': 'data/features/B/accessibility_metrics.parquet',
    'C': 'data/features/C/html_structure.parquet',
    'D': 'data/features/D/text_readability.parquet',
    'E': 'data/features/E/semantic_similarity.parquet'
}

def load_data(index_path):
    print(f"Loading Index from {index_path}...", file=sys.stderr)
    try:
        index_df = pd.read_parquet(index_path)
    except FileNotFoundError:
        print(f"Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Total Index Records: {len(index_df)}", file=sys.stderr)
    
    # Filter only status=ok for meaningful analysis
    ok_df = index_df[index_df['status'] == 'ok'].copy()
    print(f"Status=OK Records: {len(ok_df)}", file=sys.stderr)
    
    # Ensure record_id is unique
    if 'record_id' in ok_df.columns:
        if not ok_df['record_id'].is_unique:
             dupes = ok_df['record_id'].duplicated().sum()
             print(f"Warning: Index record_id is not unique! Duplicates found: {dupes}", file=sys.stderr)
             print("Dropping duplicates for analysis purposes (keeping first)...", file=sys.stderr)
             ok_df = ok_df.drop_duplicates(subset='record_id', keep='first')
             
        ok_df.set_index('record_id', inplace=True)
        
    return ok_df

def merge_features(base_df, feature_paths_map=FEATURE_PATHS):
    merged_df = base_df.copy()
    
    for key, path in feature_paths_map.items():
        if os.path.exists(path):
            print(f"Loading Feature Set {key} from {path}...", file=sys.stderr)
            try:
                feat_df = pd.read_parquet(path)
                
                # Deduplicate feature set too
                if 'record_id' in feat_df.columns:
                     if not feat_df['record_id'].is_unique:
                         feat_df = feat_df.drop_duplicates(subset='record_id', keep='first')
                     feat_df.set_index('record_id', inplace=True)
                
                # Rename status columns
                if 'step_status' in feat_df.columns:
                    if 'step_reason' not in feat_df.columns:
                        feat_df['step_reason'] = None
                    feat_df.rename(columns={'step_status': f'status_{key}', 'step_reason': f'reason_{key}'}, inplace=True)
                
                # Drop overlapping columns - latest overwrites existing
                overlap = [c for c in feat_df.columns if c in merged_df.columns]
                if overlap:
                    merged_df.drop(columns=overlap, inplace=True)
                
                # Merge
                merged_df = merged_df.join(feat_df, how='left')
                
            except Exception as e:
                print(f"Error loading {path}: {e}", file=sys.stderr)
        else:
            print(f"Feature Set {key} not found (skipping): {path}", file=sys.stderr)
            
    return merged_df
