import argparse
import sys
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import datetime
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from linearmodels.panel import PanelOLS

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Helpers (Reusing D helpers)
from src.analysis.D.helpers.stat_utils import run_r_wrapper, prep_model_data, standardize_variables, validate_and_prep_data
from src.helpers.data_loader import load_analysis_metrics

metrics = load_analysis_metrics()
BLOCK_SEMANTICS = metrics.get("semantic", [])
BLOCK_READABILITY = metrics.get("readability", [])
BLOCK_PERFORMANCE = metrics.get("performance", [])
BLOCK_ACCESSIBILITY = metrics.get("accessibility", [])

# Core predictors for difficulty interaction
DIFFICULTY_TARGET_PREDICTORS = BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY

def assign_difficulty_bands(df, metric='sim_content', n_bands=3):
    """
    Assigns query difficulty bands based on the mean semantic similarity
    of the top results for each query.
    """
    print("Calculating query difficulty metrics...", file=sys.stderr)
    
    # 1. Calculate query-level dispersion (std) of sim_content
    query_stats = df.groupby('search_term')[metric].agg(['std', 'mean']).reset_index()
    query_stats.columns = ['search_term', 'q_dispersion', 'q_mean_sim']
    
    # Handle NaN
    query_stats = query_stats.dropna()
    
    if len(query_stats) == 0:
        return df, None
        
    # 2. Create Bands
    try:
        labels = ['High_Difficulty', 'Medium_Difficulty', 'Low_Difficulty']
        query_stats['difficulty_band'] = pd.qcut(query_stats['q_dispersion'], q=n_bands, labels=labels)
    except ValueError:
        query_stats['difficulty_band'] = 'Medium_Difficulty'
        
    # Merge back
    df_merged = df.merge(query_stats[['search_term', 'difficulty_band', 'q_dispersion']], on='search_term', how='left')
    
    return df_merged, query_stats

def main():
    parser = argparse.ArgumentParser(description="Analysis F (RQ7): Query Difficulty (R Wrapper)")
    parser.add_argument("--dataset", default="data/dataset.parquet", help="Path to merged dataset")
    parser.add_argument("--out-dir", default="data/analysis/F", help="Output directory")
    parser.add_argument("--dataset-variant", default="clean", help="Variant tag")
    parser.add_argument("--code-version", default="unknown", help="Git SHA")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read dataset directly (prevent double prep & multiplier application)
    df = pd.read_parquet(args.dataset)
    
    # 0. Assign Bands (Must happen in Python to generate the signal, or just for reporting)
    df_banded, q_stats = assign_difficulty_bands(df, metric='sim_content')
    
    # Filter out rows where difficulty could not be assigned
    if 'difficulty_band' in df_banded.columns:
        df_banded = df_banded.dropna(subset=['difficulty_band'])
    
    print(f"Assigned difficulty bands. Valid N={len(df_banded)}", file=sys.stderr)
    
    if q_stats is not None:
        q_stats.to_csv(f"{args.out_dir}/query_difficulty_stats.csv", index=False)
        
    # Write temp dataset with just raw data + bands.
    temp_dataset = os.path.join(args.out_dir, "dataset_with_bands.parquet")
    df_banded.to_parquet(temp_dataset, index=False)
    
    print("Running RQ7 Query Difficulty Analysis (via R)...", file=sys.stderr)
    
    r_script = os.path.join(project_root, "src/analysis/F/query_difficulty.R")
    # Pass the TEMP dataset to R
    success = run_r_wrapper(r_script, ["--dataset", temp_dataset, "--out-dir", args.out_dir])
    
    if os.path.exists(temp_dataset):
        os.remove(temp_dataset)
    
    if not success:
         sys.exit("R script execution failed.")
         
    print(f"\nSaved results to {args.out_dir}", file=sys.stderr)  
    
if __name__ == "__main__":
    main()
