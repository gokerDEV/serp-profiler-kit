import argparse
import sys
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import datetime
from statsmodels.stats.multitest import multipletests

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Helpers
# Import Helpers
from src.analysis.D.helpers.stat_utils import run_panel_ols, run_logit_model, prep_model_data, standardize_variables, apply_winsorization, validate_and_prep_data, check_and_mitigate_collinearity, run_r_wrapper
from src.helpers.data_loader import load_analysis_metrics
from src.analysis.D.helpers.report_utils import extract_coeffs


metrics = load_analysis_metrics()
BLOCK_SEMANTICS = metrics.get("semantic", [])
BLOCK_READABILITY = metrics.get("readability", [])
BLOCK_PERFORMANCE = metrics.get("performance", [])
BLOCK_ACCESSIBILITY = metrics.get("accessibility", [])

# FDR Whitelist 
FDR_WHITELIST = set(BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY)
FDR_FAMILY_MODE = "per_model"

def main():
    parser = argparse.ArgumentParser(description="Analysis D (RQ2-5)")
    parser.add_argument("--dataset", default="data/dataset.parquet", help="Path to merged dataset")
    parser.add_argument("--out-dir", default="data/analysis/D", help="Output directory")
    parser.add_argument("--dataset-variant", default="clean", help="Variant tag for dataset (e.g. clean, winsorized)")
    parser.add_argument("--code-version", default="unknown", help="Git SHA or version tag of the code")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load & Validate
    df, dataset_meta = validate_and_prep_data(args.dataset, args.dataset_variant, args.code_version)
    
    print(f"FDR Configuration: {FDR_FAMILY_MODE} (Whitelist={len(FDR_WHITELIST)} vars)", file=sys.stderr)
    from src.helpers.data_loader import get_rank_tiers
    top_k_thresholds = get_rank_tiers()['logit_cut_points']
    from src.analysis.D.helpers.confirmatory import run_confirmatory_models
    from src.analysis.D.helpers.supplementary import run_supplementary_models

    results_conf = []
    results_supp = []
    model_eligibility = []
    model_status = {}

    # --- Stratification: Source Domain ---
    subsets = {
        'Full': df,
        'NoSource': df[~df['is_source_domain']] if 'is_source_domain' in df.columns else pd.DataFrame()
    }
    
    for subset_name, current_df in subsets.items():
        if current_df.empty: continue
        print(f"\n--- Processing Subset: {subset_name} ---", file=sys.stderr)
        
        # Confirmatory Models
        c_res, c_elig, c_status = run_confirmatory_models(
            current_df, subset_name, dataset_meta, top_k_thresholds,
            BLOCK_SEMANTICS, BLOCK_READABILITY, BLOCK_PERFORMANCE, BLOCK_ACCESSIBILITY,
            FDR_WHITELIST
        )
        results_conf.extend(c_res)
        model_eligibility.extend(c_elig)
        model_status.update(c_status)
        
        # Supplementary Models
        s_res, s_elig, s_status = run_supplementary_models(
            current_df, subset_name, dataset_meta, top_k_thresholds,
            BLOCK_SEMANTICS, BLOCK_READABILITY, BLOCK_PERFORMANCE, BLOCK_ACCESSIBILITY,
            FDR_WHITELIST
        )
        results_supp.extend(s_res)
        model_eligibility.extend(s_elig)
        model_status.update(s_status)

    def process_and_save(res, fname):
        res_df = pd.DataFrame(res)
        if not res_df.empty:
            res_df['p_fdr'] = np.nan
            res_df['fdr_significant'] = False
            for m_id in res_df['model_id'].unique():
                mask = (res_df['model_id'] == m_id) & (res_df['is_whitelist'])
                pvals = res_df.loc[mask, 'pval']
                if not pvals.empty:
                    from statsmodels.stats.multitest import multipletests
                    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                    res_df.loc[mask, 'p_fdr'] = pvals_corrected
                    res_df.loc[mask, 'fdr_significant'] = reject
            
            res_df = res_df.rename(columns={
                'coef': 'effect_size', 
                'ci_lower': 'ci_lower_95', 
                'ci_upper': 'ci_upper_95'
            })
            res_df.to_csv(f"{args.out_dir}/{fname}", index=False)

    def generate_comparison_tables(res, out_dir):
        res_df = pd.DataFrame(res)
        if res_df.empty or 'subset' not in res_df.columns: return
        
        # Calculate FDR manually for completeness if skipped
        base_df = res_df.copy()
        base_df['base_model_id'] = base_df['model_id'].str.replace(r'_(Full|NoSource)$', '', regex=True)
        
        pivot_df = base_df.pivot_table(
            index=['base_model_id', 'term'],
            columns='subset',
            values=['coef', 'pval'],
            aggfunc='first'
        ).reset_index()
        
        pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns]
        
        if 'coef_Full' in pivot_df.columns and 'coef_NoSource' in pivot_df.columns:
            pivot_df['Delta (%)'] = ((pivot_df['coef_NoSource'] - pivot_df['coef_Full']) / pivot_df['coef_Full'].abs() * 100).fillna(0)
            
            comp_table = pivot_df[['term', 'base_model_id', 'coef_Full', 'coef_NoSource', 'Delta (%)', 'pval_NoSource']].copy()
            comp_table.rename(columns={
                'term': 'Predictor',
                'coef_Full': 'Estimate (Full)',
                'coef_NoSource': 'Estimate (NoSource)',
                'pval_NoSource': 'P-Value (NoSource)'
            }, inplace=True)
            comp_table.to_csv(f"{out_dir}/confirmatory_table_comparison.csv", index=False)
            
            # Stability Table (Sign Flip)
            stability = pivot_df[
                (pivot_df['coef_Full'] * pivot_df['coef_NoSource'] < 0)
            ].copy()
            
            if not stability.empty:
                stability['Flag'] = '🚩 Sign Flip'
                stab_out = stability[['term', 'base_model_id', 'coef_Full', 'coef_NoSource', 'Flag']]
                stab_out.rename(columns={'term': 'Predictor', 'coef_Full': 'Estimate (Full)', 'coef_NoSource': 'Estimate (NoSource)'}, inplace=True)
                stab_out.to_csv(f"{out_dir}/stability_table.csv", index=False)
            else:
                pd.DataFrame(columns=['Predictor', 'base_model_id', 'Estimate (Full)', 'Estimate (NoSource)', 'Flag']).to_csv(f"{out_dir}/stability_table.csv", index=False)

    process_and_save(results_conf, "confirmatory_coeffs.csv")
    generate_comparison_tables(results_conf, args.out_dir)
    process_and_save(results_supp, "supplementary_coeffs.csv")

    pd.DataFrame(model_eligibility).to_csv(f"{args.out_dir}/model_eligibility_report.csv", index=False)
    with open(f"{args.out_dir}/model_status.json", "w") as f:
        json.dump(model_status, f, indent=2)
        
    # --- Run R Implementation (Validation) ---
    print("Running R Confirmatory Models (Validation)...", file=sys.stderr)
    
    r_script = os.path.join(project_root, "src/analysis/D/confirmatory_models.R")
    run_r_wrapper(r_script, ["--dataset", args.dataset, "--out-dir", args.out_dir])

    print("Running R Supplementary Models (Validation)...", file=sys.stderr)
    r_supp_script = os.path.join(project_root, "src/analysis/D/supplementary_models.R")
    run_r_wrapper(r_supp_script, ["--dataset", args.dataset, "--out-dir", args.out_dir])
    
    print("Running R Nested Model Fit Comparisons...", file=sys.stderr)
    r_nested_script = os.path.join(project_root, "src/analysis/D/helpers/nested_model_fit.R")
    run_r_wrapper(r_nested_script, ["--dataset", args.dataset, "--out-dir", args.out_dir])

    print(f"\nSaved results to {args.out_dir}", file=sys.stderr)
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
