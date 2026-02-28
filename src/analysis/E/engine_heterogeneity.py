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

# Import Helpers (Reusing D helpers as they are compatible stats tools)
# Import Helpers (Reusing D helpers)
from src.analysis.D.helpers.stat_utils import validate_and_prep_data, run_r_wrapper
from src.helpers.data_loader import load_analysis_metrics

# --- CONFIGURATION (Locked per Plan v10) ---

metrics = load_analysis_metrics()
BLOCK_SEMANTICS = metrics.get("semantic", [])
BLOCK_READABILITY = metrics.get("readability", [])
BLOCK_PERFORMANCE = metrics.get("performance", [])
BLOCK_ACCESSIBILITY = metrics.get("accessibility", [])

# Primary predictors
HETEROGENEITY_PREDICTORS = BLOCK_SEMANTICS + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY + BLOCK_READABILITY

def generate_replication_grid(coeffs, predictors):
    """
    Analyzes engine-stratified coefficients to generate a Replication Grid.
    Metrics:
    - Sign Agreement (All signs match?)
    - CI Overlap (Do 95% CIs overlap across all engines?)
    - Practical Significance Stability (Is practical_flag consistent?)
    """
    grid_stats = []
    
    for term in predictors:
        # Get coeffs for this term from engine models
        relevant = [c for c in coeffs if c['term'] == term and str(c.get('model_id', '')).startswith('RQ6_Stratified_')]
        
        if not relevant: continue
        
        signs = [np.sign(r.get('effect_size', r.get('coef'))) for r in relevant]
        pvals = [r.get('p_raw', r.get('pval')) for r in relevant]
        cis = [(r.get('ci_lower_95', r.get('ci_lower')), r.get('ci_upper_95', r.get('ci_upper'))) for r in relevant]
        
        # Agreement: All signs same?
        all_same_sign = len(set(signs)) == 1
        
        # Significant Agreement: All significant AND same sign?
        sig_agreement = all(p < 0.05 for p in pvals) and all_same_sign
        
        # CI Overlap
        max_lower = max([c[0] for c in cis])
        min_upper = min([c[1] for c in cis])
        ci_overlap = max_lower <= min_upper
        
        stats = {
            'term': term,
            'n_engines': len(relevant),
            'all_sign_agreement': all_same_sign,
            'significant_agreement': sig_agreement,
            'ci_overlap': ci_overlap,
            'median_coef': np.median([r.get('effect_size', r.get('coef')) for r in relevant]),
            'mad_coef': np.median(np.abs(np.array([r.get('effect_size', r.get('coef')) for r in relevant]) - np.median([r.get('effect_size', r.get('coef')) for r in relevant])))
        }
        grid_stats.append(stats)
        
    return grid_stats

def main():
    parser = argparse.ArgumentParser(description="Analysis E (RQ6): Engine Heterogeneity (R Wrapper)")
    parser.add_argument("--dataset", default="data/dataset.parquet", help="Path to merged dataset")
    parser.add_argument("--out-dir", default="data/analysis/E", help="Output directory")
    parser.add_argument("--dataset-variant", default="clean", help="Variant tag for dataset")
    parser.add_argument("--code-version", default="unknown", help="Git SHA or version tag")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Validation only (R script assumes clean data mostly, but we validate path)
    if not os.path.exists(args.dataset):
        sys.exit(f"Dataset not found: {args.dataset}")
        
    print("Running RQ6 Engine Heterogeneity Analysis (via R)...", file=sys.stderr)
    
    r_script = os.path.join(project_root, "src/analysis/E/engine_heterogeneity.R")
    success = run_r_wrapper(r_script, ["--dataset", args.dataset, "--out-dir", args.out_dir])
    
    if not success:
        sys.exit("R script execution failed.")
        
    # --- Post-Processing: Generate Replication Grid ---
    r_out_csv = os.path.join(args.out_dir, "heterogeneity_coeffs_r.csv")
    if os.path.exists(r_out_csv):
        print("Processing R output for Replication Grid...", file=sys.stderr)
        try:
            df_coeffs = pd.read_csv(r_out_csv)
            coeffs_list = df_coeffs.to_dict('records')
            
            # Find unique subsets for stratified models
            stratified_coeffs = [c for c in coeffs_list if str(c.get('model_id', '')).startswith('RQ6_Stratified_')]
            # Subset values look like: Engine_google_Full, Engine_google_NoSource
            # Let's extract the actual data subset (Full, NoSource, Source)
            # The model_id usually ends with _Full, _NoSource, or _Source
            import re
            
            subsets = set()
            for c in stratified_coeffs:
                m_id = str(c.get('model_id', ''))
                match = re.search(r'_(Full|NoSource|Source)$', m_id)
                if match:
                    subsets.add(match.group(1))
            
            if not subsets:
                subsets = {'Full'} # Fallback
                
            all_grid_stats = []
            for subset in subsets:
                # Filter coeffs for this specific subset
                subset_coeffs = [c for c in stratified_coeffs if str(c.get('model_id', '')).endswith(f"_{subset}")]
                if not subset_coeffs: continue
                
                grid_stats = generate_replication_grid(subset_coeffs, HETEROGENEITY_PREDICTORS)
                for g in grid_stats:
                    g['subset'] = subset
                all_grid_stats.extend(grid_stats)
            
            if all_grid_stats:
                pd.DataFrame(all_grid_stats).to_csv(f"{args.out_dir}/replication_grid.csv", index=False)
                print("Generated replication_grid.csv", file=sys.stderr)
            
        except Exception as e:
            print(f"Error processing replication grid: {e}", file=sys.stderr)
    else:
        print("Warning: R output CSV not found. Skipping replication grid.", file=sys.stderr)

    print(f"\nSaved results to {args.out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
