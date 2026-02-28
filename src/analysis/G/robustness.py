import argparse
import sys
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import datetime
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Helpers
from src.analysis.D.helpers.stat_utils import validate_and_prep_data, run_r_wrapper, prep_model_data, run_logit_model, run_panel_ols
from src.helpers.data_loader import load_analysis_metrics
from src.analysis.D.helpers.report_utils import extract_coeffs
import statsmodels.formula.api as smf

# --- CONFIGURATION ---

metrics = load_analysis_metrics()
BLOCK_SEMANTICS = metrics.get("semantic", [])
BLOCK_READABILITY = metrics.get("readability", [])
BLOCK_PERFORMANCE = metrics.get("performance", [])
BLOCK_ACCESSIBILITY = metrics.get("accessibility", [])

FDR_WHITELIST = set(BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY)

def checks_influence(df, formula, model_id):
    """
    Computes Cook's Distance-like influence measures.
    (Keeping this in Python as R script doesn't implement it yet)
    """
    clean_df = prep_model_data(df, formula, model_id, [])
    if clean_df is None or len(clean_df) == 0: return None
    
    # Remove EntityEffects from formula for OLS proxy
    ols_formula = formula.replace('+ EntityEffects', '').replace(' + EntityEffects', '')
    
    try:
        mod = smf.ols(ols_formula, data=clean_df)
        res = mod.fit()
        influence = res.get_influence()
        cooks_d = influence.cooks_distance[0]
        
        threshold = 4 / len(clean_df)
        outliers = np.where(cooks_d > threshold)[0]
        
        return {
            'n_outliers': len(outliers),
            'outlier_pct': len(outliers) / len(clean_df) * 100,
            'max_cooks': np.max(cooks_d)
        }
    except Exception as e:
        print(f"Influence check failed: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Analysis G (RQ8): Robustness Checks (R Wrapper + Python Extras)")
    parser.add_argument("--dataset", default="data/dataset.parquet")
    parser.add_argument("--out-dir", default="data/analysis/G")
    parser.add_argument("--dataset-variant", default="clean")
    parser.add_argument("--code-version", default="unknown")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df, dataset_meta = validate_and_prep_data(args.dataset, args.dataset_variant, args.code_version)
    
    # --- Part 1: Main Robustness Checks (via R) ---
    # Covers: Baseline, NoSource, Winsorized (FE Models)
    # --- Part 1: Main Robustness Checks (via R) ---
    # Covers: Baseline, NoSource, Winsorized (FE Models)
    # Baseline, NoSource, Winsorized (FE Models)
    print("Running RQ8 Main Robustness Checks (via R)...", file=sys.stderr)
    
    r_script = os.path.join(project_root, "src/analysis/G/robustness.R")
    success = run_r_wrapper(r_script, ["--dataset", args.dataset, "--out-dir", args.out_dir])
    
    if not success:
        print("Warning: R script execution failed. Some results may be missing.", file=sys.stderr)

    # --- Part 2: Additional Checks (Python) ---
    # Influence Checks (Cook's D) & Top-k Logits (not in R script yet)
    
    print("Running Additional Robustness Checks (Python)...", file=sys.stderr)
    results = []
    
    f_rq2 = f"recip_rank ~ {' + '.join(BLOCK_SEMANTICS)} + C(search_engine, Treatment(reference='google')) + EntityEffects + 1"
    
    # --- Check 4: Influence / Outliers (Cook's D) ---
    print("Running Influence Checks...", file=sys.stderr)
    infl_stats = checks_influence(df, f_rq2, "RQ2_Influence")
    if infl_stats:
        with open(f"{args.out_dir}/influence_stats.json", "w") as f:
            json.dump(infl_stats, f, indent=2)

    # --- Check 5: Outcome Families (Top-k Logits) ---
    print("Running Top-k Robustness...", file=sys.stderr)
    # We run on Full and NoSource
    source_variants = {
        'Full': df,
        'NoSource': df[~df['is_source_domain']] if 'is_source_domain' in df.columns else pd.DataFrame()
    }
    
    from src.helpers.data_loader import get_rank_tiers
    top_k_thresholds = get_rank_tiers()['logit_cut_points']
    
    for variant, v_df in source_variants.items():
        if v_df.empty or len(v_df) < 50: continue
        
        for k_val in top_k_thresholds:
            target = f"is_top{k_val}"
            if target not in v_df.columns:
                 v_df[target] = (v_df['rank'] <= k_val).astype(int)
                 
            f_logit = f"{target} ~ {' + '.join(BLOCK_SEMANTICS)} + C(search_engine, Treatment(reference='google'))"
            res_l = run_logit_model(v_df, f_logit, f"RQ2_Logit_{variant}_Top{k_val}", [])
            if res_l:
                rows = extract_coeffs(res_l, f"RQ2_Logit_{variant}_Top{k_val}", f"Top-{k_val} Logit", dataset_meta, FDR_WHITELIST)
                for r in rows: r['robustness_check'] = 'Source_x_OutcomeFamily'
                results.extend(rows)

    # Note: 2-Way Clustering removed from Python as it's computationally heavy and R likely handles clustering better if updated.
    # For now we skip it or assume R will handle it in future updates.

    # Save Python results
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df.to_csv(f"{args.out_dir}/robustness_coeffs_python_extras.csv", index=False)
        print(f"Saved python extra results to {args.out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
