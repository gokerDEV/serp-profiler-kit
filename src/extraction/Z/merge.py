import pandas as pd
import numpy as np
import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.helpers.schema_validator import validate_columns
from src.extraction.Z.helpers.report import generate_dataset_report

# Feature paths
FEATURE_NAMES = ['A', 'B', 'C', 'D', 'E']

def load_index(index_path):
    print(f"Loading Index from {index_path}...", file=sys.stderr)
    try:
        df = pd.read_parquet(index_path)
    except FileNotFoundError:
        print(f"Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)
            
    df.set_index('record_id', inplace=True)
            
    return df

def merge_features(base_df, feature_paths, ignore_outliers=True):
    merged_df = base_df.copy()
    
    # 1. Outlier filtering
    if 'Outliers' in feature_paths and os.path.exists(feature_paths['Outliers']):
        outlier_df = pd.read_parquet(feature_paths['Outliers'])
        if 'record_id' in outlier_df.columns:
            outlier_df.set_index('record_id', inplace=True)
            
        merged_df = merged_df.join(outlier_df, how='left')
        
        if ignore_outliers:
            print("Filtering records based on outliers...", file=sys.stderr)
            outlier_cols = [c for c in outlier_df.columns if c.startswith('is_outlier_')]
            if outlier_cols:
                is_outlier = (merged_df[outlier_cols] == 'outlier').any(axis=1)
                before_filter = len(merged_df)
                merged_df = merged_df[~is_outlier]
                print(f"Removed {before_filter - len(merged_df)} outlier records.", file=sys.stderr)

    # 2. Sequential Merging
    for key in FEATURE_NAMES:
        path = feature_paths.get(key)
        if path and os.path.exists(path):
            print(f"Merging Feature Set {key}...", file=sys.stderr)
            feat_df = pd.read_parquet(path)
            
            if 'record_id' in feat_df.columns:
                if not feat_df['record_id'].is_unique:
                    feat_df = feat_df.drop_duplicates(subset='record_id', keep='first')
                feat_df.set_index('record_id', inplace=True)
            
            if 'step_status' in feat_df.columns:
                 if 'step_reason' not in feat_df.columns:
                     feat_df['step_reason'] = None
                 feat_df.rename(columns={'step_status': f'status_{key}', 'step_reason': f'reason_{key}'}, inplace=True)
            
            # Generic Update & Join: Subsequent steps overwrite previous data if columns overlap
            overlap = merged_df.columns.intersection(feat_df.columns)
            if not overlap.empty:
                # Update existing columns with non-NA values from current feature set
                merged_df.update(feat_df[overlap])
            
            # Join remaining new columns
            new_cols = feat_df.columns.difference(merged_df.columns)
            if not new_cols.empty:
                merged_df = merged_df.join(feat_df[new_cols], how='left')
            # Update global status: If this step failed, the record is no longer 'ok'
            if f'status_{key}' in merged_df.columns:
                # We only mark as 'fail' if it's not 'ok' and not NaN (NaN means it wasn't in this feature set)
                # But actually, if it's missing from a feature set it should probably also be a fail? 
                # For now, let's be strict: anything not 'ok' -> fail
                fail_mask = (merged_df[f'status_{key}'].notna()) & (merged_df[f'status_{key}'] != 'ok')
                if fail_mask.any():
                    merged_df.loc[fail_mask, 'status'] = 'fail'

        else:
            print(f"Feature Set {key} not found at {path}", file=sys.stderr)

    # Final Cleanup: Convert empty strings to NaN for consistent reporting
    merged_df = merged_df.replace(r'^\s*$', None, regex=True)

    return merged_df

def apply_acceptance_criteria(df, config_path):
    if not os.path.exists(config_path):
        return df, {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        return df, {}
        
    filters = config.get('filters', [])
    current_df = df.copy()
    initial_count = len(current_df)
    stats = {}
    
    for f_cfg in filters:
        col, op, val = f_cfg.get('column'), f_cfg.get('op'), f_cfg.get('value')
        desc = f_cfg.get('description', f"{col} {op} {val}")
        if col not in current_df.columns: continue
        
        before = len(current_df)
        if op == 'eq': mask = (current_df[col] == val)
        elif op == 'noteq': mask = (current_df[col] != val)
        elif op == 'gt': mask = (current_df[col] > val)
        elif op == 'lt': mask = (current_df[col] < val)
        elif op in ['ge', 'min']: mask = (current_df[col] >= val)
        elif op in ['le', 'max']: mask = (current_df[col] <= val)
        elif op == 'notnull': mask = current_df[col].notnull()
        elif op == 'isnull': mask = current_df[col].isnull()
        else: continue
            
        current_df = current_df[mask]
        stats[desc] = before - len(current_df)
        
    return current_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index.parquet")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--include-outliers", action="store_true")
    parser.add_argument("--feature-base-dir", default="data/features")
    parser.add_argument("--outliers-path", default="data/outliers.parquet")
    parser.add_argument("--acceptance-config", default="src/extraction/Z/acceptance.json")
    args = parser.parse_args()
    
    feature_paths = {k: os.path.join(args.feature_base_dir, f"{k}/{'runtime_metrics' if k=='A' else 'accessibility_metrics' if k=='B' else 'html_structure' if k=='C' else 'text_readability' if k=='D' else 'semantic_similarity'}.parquet") for k in FEATURE_NAMES}
    feature_paths['Outliers'] = args.outliers_path

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"dataset-{timestamp}.parquet")
    report_path = os.path.join("data/reports", f"dataset-{timestamp}.md")
    
    df = load_index(args.index)
    merged_df = merge_features(df, feature_paths, ignore_outliers=not args.include_outliers)
        
    final_df, acceptance_stats = apply_acceptance_criteria(merged_df, args.acceptance_config)
    
    validate_columns(final_df, strict=False)
    final_df.reset_index().to_parquet(out_path, index=False)
    
    params = {'Index Source': args.index, 'Ignore Outliers': not args.include_outliers, 'Acceptance Config': args.acceptance_config}
    generate_dataset_report(final_df, report_path, params, acceptance_stats=acceptance_stats)
    print(f"Dataset: {out_path}\nReport: {report_path}")

if __name__ == "__main__":
    main()
