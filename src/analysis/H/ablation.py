import argparse
import sys
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from linearmodels.panel import PanelOLS

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Helpers
from src.analysis.D.helpers.stat_utils import validate_and_prep_data, run_r_wrapper
from src.helpers.data_loader import load_analysis_metrics


metrics = load_analysis_metrics()
BLOCK_SEMANTICS = metrics.get("semantic", [])
BLOCK_READABILITY = metrics.get("readability", [])
BLOCK_PERFORMANCE = metrics.get("performance", [])
BLOCK_ACCESSIBILITY = metrics.get("accessibility", [])

def train_evaluate_ltr_and_stability(df, feature_cols, target_col='recip_rank', k=10, run_permutation=False, permutation_repeats=5):
    """
    Corroborative LTR Evaluation & Stability.
    
    1. CV NDCG@10 Evaluation.
    2. Rank Stability (Permutation Importance) [Optional, on Full Train].
       - Planda: "grouped-by-query evaluation... permutation/SHAP rank stability"
       - We run permutation importance on the hold-out sets or a separate validation?
       - Standard: Run Permutation Importance on the whole set (or CV) to get feature importance.
    """
    groups = df['search_term']
    gkf = GroupKFold(n_splits=5)
    
    ndcg_scores = []
    importances = []
    query_ndcgs = {}
    
    for train_idx, test_idx in gkf.split(df, y=df[target_col], groups=groups):
        # Prepare Train
        X_train = df.iloc[train_idx][feature_cols].fillna(0) 
        y_train = df.iloc[train_idx][target_col]
        
        # Prepare Test
        X_test_full = df.iloc[test_idx].copy()
        X_test = X_test_full[feature_cols].fillna(0)
        y_test = X_test_full[target_col]
        
        # Train RF
        model = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Calculate NDCG per query
        X_test_full['predicted_score'] = preds
        X_test_full['relevance'] = y_test
        
        # --- NDCG Calc ---
        fold_ndcg = _mean_group_ndcg_at_k(
            df_eval=X_test_full,
            pred_col='predicted_score',
            rel_col='relevance',
            group_col='search_term',
            k=k,
        )
        ndcg_scores.append(fold_ndcg)
        
        for q, group in X_test_full.groupby('search_term'):
            if len(group) < 2: continue
            
            # Sort by predicted score desc
            pred_sorted = group.sort_values('predicted_score', ascending=False)
            # Sort by ideal relevance desc
            ideal_sorted = group.sort_values('relevance', ascending=False)
            
            # DCG
            dcg = 0.0
            for i, rel in enumerate(pred_sorted['relevance'].head(k)):
                dcg += (np.power(2.0, float(rel)) - 1.0) / np.log2(i + 2)
                
            # IDCG
            idcg = 0.0
            for i, rel in enumerate(ideal_sorted['relevance'].head(k)):
                idcg += (np.power(2.0, float(rel)) - 1.0) / np.log2(i + 2)
                
            if idcg > 0:
                query_ndcgs[q] = dcg / idcg
            else:
                query_ndcgs[q] = 0.0
            
        # --- Permutation Importance (Rank Stability) ---
        if run_permutation:
            # Build group lookup aligned with X_test index
            test_groups = X_test_full['search_term'].copy()

            def ndcg_permutation_scorer(
                estimator: RandomForestRegressor,
                X_perm: pd.DataFrame,
                y_true: pd.Series | np.ndarray,
            ) -> float:
                # estimator.predict on permuted feature matrix
                preds_perm = estimator.predict(X_perm)

                # Ensure y_true is aligned as Series with X index
                if isinstance(y_true, pd.Series):
                    y_series = y_true.reindex(X_perm.index)
                else:
                    y_series = pd.Series(y_true, index=X_perm.index)

                eval_df = pd.DataFrame(
                    {
                        'search_term': test_groups.reindex(X_perm.index).values,
                        'relevance': y_series.values,
                        'predicted_score': preds_perm,
                    },
                    index=X_perm.index,
                )

                return _mean_group_ndcg_at_k(
                    df_eval=eval_df,
                    pred_col='predicted_score',
                    rel_col='relevance',
                    group_col='search_term',
                    k=k,
                )

            perm_result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=5,           # increase later if compute allows
                random_state=42,
                n_jobs=-1,
                scoring=ndcg_permutation_scorer,
            )
            importances.append(perm_result.importances_mean)

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    avg_importances = {}
    if run_permutation and importances:
        # Average across folds
        avg_imp_vals = np.mean(importances, axis=0)
        avg_importances = dict(zip(feature_cols, avg_imp_vals))
        
    return mean_ndcg, avg_importances, query_ndcgs


def _mean_group_ndcg_at_k(
    df_eval: pd.DataFrame,
    pred_col: str,
    rel_col: str,
    group_col: str,
    k: int = 10,
) -> float:
    """
    Compute mean query-grouped NDCG@k.
    Expects df_eval to contain prediction, relevance, and group columns.
    """
    ndcgs: list[float] = []

    for _, group in df_eval.groupby(group_col):
        if len(group) < 2:
            continue

        pred_sorted = group.sort_values(pred_col, ascending=False)
        ideal_sorted = group.sort_values(rel_col, ascending=False)

        dcg = 0.0
        for i, rel in enumerate(pred_sorted[rel_col].head(k)):
            dcg += (np.power(2.0, float(rel)) - 1.0) / np.log2(i + 2)

        idcg = 0.0
        for i, rel in enumerate(ideal_sorted[rel_col].head(k)):
            idcg += (np.power(2.0, float(rel)) - 1.0) / np.log2(i + 2)

        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    if not ndcgs:
        return 0.0

    return float(np.mean(ndcgs))

def main():
    parser = argparse.ArgumentParser(description="Analysis H (RQ9): Ablation Study")
    parser.add_argument("--dataset", default="data/dataset.parquet")
    parser.add_argument("--out-dir", default="data/analysis/H")
    parser.add_argument("--dataset-variant", default="clean")
    parser.add_argument("--code-version", default="unknown")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Reuse updated validate_and_prep_data
    df, meta = validate_and_prep_data(args.dataset, args.dataset_variant, args.code_version)
    
    # --- Status Tracking ---
    status_report = {'dataset_id': meta['dataset_id'], 'models': {}}
    model_eligibility = []
    
    # Baseline NDCG + Rank Stability (Permutation)
    print("Running Inferential Models (via R)...", file=sys.stderr)
    
    r_script = os.path.join(project_root, "src/analysis/H/ablation.R")
    success = run_r_wrapper(r_script, ["--dataset", args.dataset, "--out-dir", args.out_dir])
    
    if success:
        status_report['models']['Inferential_R'] = "success"
    else:
        status_report['models']['Inferential_R'] = "failed"
        print("Warning: R inferential steps failed.", file=sys.stderr)

    # Note: R script saves to ablation_inferential_r.csv directly

    
    # --- Part 2: Corroborative LTR Ablation (Engine Stratified) ---
    print("Running Corroborative LTR Ablation (Step F)...", file=sys.stderr)
    
    predictive_results = []
    # ... (skipping re-definition if not changing) ...
    importance_results = []
    
    # Define Subsets: Full + Each Engine + NoSource
    subsets = {'Full': df}
    if 'is_source_domain' in df.columns:
        subsets['NoSource'] = df[~df['is_source_domain']].copy()
        
    for eng in df['search_engine'].unique():
        subsets[eng] = df[df['search_engine'] == eng]
        
    for subset_name, sub_df in subsets.items():
        if len(sub_df) < 50: continue
        
        # Performance Guard: Skip LTR for massive subsets (e.g. Full > 100k) to avoid hanging
        if len(sub_df) > 100000:
             print(f"  Skipping LTR for {subset_name} (Size {len(sub_df)} > 100k) to avoid timeouts. Using Stratified results.", file=sys.stderr)
             continue
             
        print(f"  Processing Subset: {subset_name} (n={len(sub_df)})...", file=sys.stderr)
        
        # Full Feature Set
        features_full = BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY
        
        # Baseline NDCG + Rank Stability (Permutation)
        ndcg_full, importances, ndcgs_full_dict = train_evaluate_ltr_and_stability(sub_df, features_full, run_permutation=True)
        
        predictive_results.append({
            'subset': subset_name,
            'set_name': 'Full Model',
            'ndcg_mean': ndcg_full,
            'effect_size': 0.0, # Baseline
            'ci_lower_95': 0.0,
            'ci_upper_95': 0.0,
            'model_family': 'LTR',
            'evidence_tier': 'corroborative',
            'analysis_tier': 'ablation',
            'dataset_id': meta['dataset_id'],
            'generated_at': meta['generated_at'],
            'missing_policy': 'fillna_zero'
        })
        
        # Store Importances
        for feat, imp in importances.items():
            importance_results.append({
                'subset': subset_name,
                'feature': feat,
                'importance_mean': imp
            })
        
        # Ablations (Remove one block)
        ablations = [
            ("- Semantics", BLOCK_READABILITY + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY),
            ("- Readability", BLOCK_SEMANTICS + BLOCK_PERFORMANCE + BLOCK_ACCESSIBILITY),
            ("- Performance", BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_ACCESSIBILITY),
            ("- Accessibility", BLOCK_SEMANTICS + BLOCK_READABILITY + BLOCK_PERFORMANCE)
        ]
        
        for name_ab, feats in ablations:
            # Quick Eval (no perm needed)
            score, _, ndcgs_ab_dict = train_evaluate_ltr_and_stability(sub_df, feats, run_permutation=False)
            delta = score - ndcg_full
            
            # Query-cluster bootstrap for CI of the delta
            deltas_list = []
            for q in ndcgs_full_dict:
                if q in ndcgs_ab_dict:
                    deltas_list.append(ndcgs_ab_dict[q] - ndcgs_full_dict[q])
                    
            if len(deltas_list) > 1:
                rng = np.random.default_rng(42)
                d_arr = np.array(deltas_list)
                n_queries = len(d_arr)
                # 1000 bootstrap resamples
                boot_samples = rng.choice(d_arr, size=(1000, n_queries), replace=True)
                boot_means = boot_samples.mean(axis=1)
                ci_lower = np.percentile(boot_means, 2.5)
                ci_upper = np.percentile(boot_means, 97.5)
            else:
                ci_lower, ci_upper = delta, delta

            predictive_results.append({
                'subset': subset_name,
                'set_name': name_ab,
                'ndcg_mean': score,
                'effect_size': delta, # Delta NDCG as effect size
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper,
                'model_family': 'LTR',
                'evidence_tier': 'corroborative',
                'analysis_tier': 'ablation',
                'dataset_id': meta['dataset_id'],
                'generated_at': meta['generated_at'],
                'missing_policy': 'fillna_zero'
            })
        
    pd.DataFrame(predictive_results).to_csv(f"{args.out_dir}/ablation_predictive.csv", index=False)
    pd.DataFrame(importance_results).to_csv(f"{args.out_dir}/rank_stability_importance.csv", index=False)
    
    # Comparison Table Delta NDCG
    pred_df = pd.DataFrame(predictive_results)
    if 'set_name' in pred_df.columns and 'subset' in pred_df.columns:
        comp_df = pred_df[pred_df['subset'].isin(['Full', 'NoSource'])].copy()
        pivot_abl = comp_df.pivot_table(index='set_name', columns='subset', values='effect_size', aggfunc='first').reset_index()
        if 'Full' in pivot_abl.columns and 'NoSource' in pivot_abl.columns:
            pivot_abl.rename(columns={'set_name': 'Ablated Block', 'Full': 'Delta-NDCG (Full)', 'NoSource': 'Delta-NDCG (NoSource)'}, inplace=True)
            pivot_abl = pivot_abl[pivot_abl['Ablated Block'] != 'Full Model']
            pivot_abl.to_csv(f"{args.out_dir}/ablation_comparison.csv", index=False)
            
    
    # Save Status
    with open(f"{args.out_dir}/ablation_status.json", "w") as f:
        json.dump(status_report, f, indent=2)
        
    print(f"Saved ablation results to {args.out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
