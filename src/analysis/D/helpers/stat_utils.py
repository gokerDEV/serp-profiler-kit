import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import sys
import os
import datetime
from pathlib import Path


def prep_model_data(df, formula, model_id, model_eligibility, FDR_FAMILY_MODE="per_model"):
    """
    Checks missingness and prepares dataframe for modeling.
    """
    clean_df = df.copy()
    
    # Simple check based on columns present in formula
    # We need to parse formula terms to check for NaNs in specific columns?
    # Or just dropna on the subset of columns used.
    # Patsy does this automatically usually, but we want to track loss.
    
    # 1. Identify columns
    import re
    # Very rough extraction of variable names
    vars_in_formula = set(re.findall(r"[\w_]+", formula)) - {'C', 'Treatment', 'reference', 'EntityEffects', 'Intercept'}
    cols_to_check = [c for c in vars_in_formula if c in df.columns]
    
    N_pre = len(df)
    clean_df = df.dropna(subset=cols_to_check)
    N_complete = len(clean_df)
    
    loss_pct = (N_pre - N_complete) / N_pre * 100 if N_pre > 0 else 0
    
    # Cluster info
    n_clusters = clean_df['search_term'].nunique() if 'search_term' in clean_df.columns else 0
    cluster_warning = n_clusters < 30
    
    status = {
        'model_id': model_id,
        'N_pre': N_pre,
        'N_complete_case': N_complete,
        'loss_pct': round(loss_pct, 2),
        'indicator_sensitivity_triggered': (loss_pct > 15),
        'status': 'ok' if N_complete > 0 else 'fail',
        'n_clusters_search_term': n_clusters,
        'cluster_warning_flag': cluster_warning,
        'fdr_family_mode': FDR_FAMILY_MODE
    }
    model_eligibility.append(status)
    
    return clean_df

def run_panel_ols(df, formula, model_id, model_eligibility):
    clean_df = prep_model_data(df, formula, model_id, model_eligibility)
    if clean_df is None or len(clean_df) == 0:
        return None

    # Set index for PanelOLS (Entity=search_term, Time=row_id)
    if 'search_term' in clean_df.columns and 'row_id' in clean_df.columns:
        clean_df_panel = clean_df.set_index(['search_term', 'row_id'])
    else:
            return None
    
    try:
        mod = PanelOLS.from_formula(formula, clean_df_panel, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        return res
    except Exception as e:
        print(f"PanelOLS Failed for {model_id}: {e}")
        return None


def check_variance(df, predictors):
    valid_predictors = []
    dropped_predictors = []
    
    for p in predictors:
        if p in df.columns and df[p].nunique() > 1:
            valid_predictors.append(p)
        else:
            dropped_predictors.append(p)
            
    return valid_predictors, dropped_predictors

def run_logit_model(df, formula, model_id, model_eligibility):
    clean_df = prep_model_data(df, formula, model_id, model_eligibility)
    if clean_df is None or len(clean_df) == 0:
        return None
        
    import re
    dependent_var = formula.split('~')[0].strip()
    all_vars = re.findall(r"[\w_]+", formula.split('~')[1])
    predictors = [v for v in all_vars if v in clean_df.columns]

    valid_predictors, dropped = check_variance(clean_df, predictors)
    
    if dropped:
        print(f"  [Warning] Dropping zero-variance predictors for {model_id}: {dropped}")
        formula = f"{dependent_var} ~ {' + '.join(valid_predictors)}"

        
    try:
        mod = smf.logit(formula, clean_df)
        # res = mod.fit(disp=0, cov_type='cluster', cov_kwds={'groups': clean_df['search_term']})
        return mod.fit(method='bfgs', maxiter=500, disp=False, cov_type='cluster', cov_kwds={'groups': clean_df['search_term']})

    except Exception as e:
        print(f"BFGS Failed for {model_id}, trying lbfgs or newton with regularization: {e}")
        try:
            return mod.fit_regularized(method='l1', alpha=0.1, disp=0, cov_type='cluster', cov_kwds={'groups': clean_df['search_term']})
        except Exception as e2:
            print(f"Logit Regularized Failed for {model_id}: {e2}")
            return None

def standardize_variables(df, cols):
    """
    Returns a copy of df with specified columns standardized (z-score)
    """
    df_std = df.copy()
    for col in cols:
        if col in ['rank', 'recip_rank']:
            continue
            
        if col in df_std.columns:
            mean = df_std[col].mean()
            std = df_std[col].std()
            if std != 0:
                df_std[col] = (df_std[col] - mean) / std
            else:
                df_std[col] = 0
    return df_std

def apply_winsorization(df, cols, limits=(0.01, 0.01)):
    """
    Returns a copy of df with specified columns winsorized.
    """
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            # winsorize returns a masked array, cast back
            df_out[col] = winsorize(df_out[col], limits=limits).data
    return df_out

def validate_and_prep_data(path: str, dataset_variant: str, code_version: str):
    from src.helpers.data_loader import load_analysis_dataset
    df = load_analysis_dataset(path)
    
    required_cols = ['search_term', 'search_engine', 'rank']
    if any(c not in df.columns for c in required_cols):
        sys.exit(f"Missing required columns: {required_cols}")
        
            
    df = df[ (df['rank'] > 0) & (df['rank'].notna()) ]
    df['recip_rank'] = 1.0 / df['rank']
    
    from src.helpers.data_loader import get_rank_tiers
    tiers = get_rank_tiers()
    for point in tiers['logit_cut_points']:
        df[f'is_top{point}'] = (df['rank'] <= point).astype(int)
    df['search_engine'] = df['search_engine'].astype(str)
    
    if 'is_source_domain' not in df.columns:
        df['is_source_domain'] = False 
    else:
        df['is_source_domain'] = df['is_source_domain'].fillna(False).astype(bool)

    df['original_index'] = df.index
    df['row_id'] = np.arange(len(df), dtype=np.int64)
    
    dataset_meta = {
        'dataset_id': os.path.basename(path),
        'dataset_variant': dataset_variant,
        'code_version': code_version,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    return df, dataset_meta

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, predictors):
    """
    Calculates VIF for a set of predictors.
    Returns a DataFrame with 'variable' and 'vif' columns.
    """
    # 1. Prepare data (drop missing, add constant if needed)
    # statsmodels VIF requires constant or at least 2 vars
    # We assume 'df' has the predictors clean and numeric.
    
    X = df[predictors].dropna()
    if X.empty: 
        return pd.DataFrame()
        
    X = X.copy()
    X['const'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["variable"] = predictors
    
    # Calculate VIF for each predictor
    # VIF function expects array, index
    vifs = []
    for i, col in enumerate(predictors):
        # Index in X including const is i (since predictors come first? No.)
        # X columns: [p1, p2, ..., pn, const]
        # variance_inflation_factor(exog, exog_idx)
        idx = X.columns.get_loc(col)
        try:
            val = variance_inflation_factor(X.values, idx)
        except Exception:
            val = np.inf
        vifs.append(val)
        
    vif_data["vif"] = vifs
    return vif_data

def check_and_mitigate_collinearity(df, predictors, threshold=10, keep_always=None):
    """
    Iteratively checks VIF and drops variables if VIF > threshold.
    'keep_always' variables are never dropped.
    Returns: (keep_predictors, drop_predictors, vif_history)
    """
    if keep_always is None: keep_always = []
    
    current_predictors = list(predictors)
    drop_list = []
    history = []
    
    while True:
        vif_df = calculate_vif(df, current_predictors)
        if vif_df.empty: break
        
        # Log history
        history.append(vif_df.copy())
        
        # Check max VIF
        max_vif_row = vif_df.loc[vif_df['vif'].idxmax()] if not vif_df.empty else None
        
        if max_vif_row is None or max_vif_row['vif'] <= threshold:
            break
            
        # Identify candidate to drop
        candidate = max_vif_row['variable']
        
        if candidate in keep_always:
            # If highest VIF is a protected var, find next highest?
            # Or just stop and warn?
            # Standard mitigation: Try to drop the OTHER variable causing correlation.
            # For simplicity: Drop the non-protected variable with highest VIF.
            candidates = vif_df[~vif_df['variable'].isin(keep_always)].sort_values('vif', ascending=False)
            if candidates.empty:
                # All remaining are protected
                print(f"Warning: Multicollinearity in protected variables {keep_always} (VIF={max_vif_row['vif']:.2f}). Cannot drop.", file=sys.stderr)
                break
            candidate = candidates.iloc[0]['variable']
            
        print(f"Dropping {candidate} due to VIF={vif_df.loc[vif_df['variable']==candidate, 'vif'].values[0]:.2f}", file=sys.stderr)
        current_predictors.remove(candidate)
        drop_list.append(candidate)
        
    return current_predictors, drop_list, history

def run_bootstrap_ci(model_func, df, formula, n_boot=1000, cluster_col='search_term'):
    """
    Generic bootstrap function for coefficient CIs.
    model_func: function that takes (df, formula) and returns a result object with .params
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Cluster column {cluster_col} not found")
        
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    
    boot_coeffs = []
    
    print(f"Starting Bootstrap (B={n_boot})...", file=sys.stderr)
    
    for i in range(n_boot):
        # Resample clusters
        sampled_clusters = np.random.choice(clusters, n_clusters, replace=True)
        
        # Create sampled DF (simple concatenation)
        # This can be slow for large DF. Optimization: use indexing?
        # df is indexed by search_term?
        # Better: df.set_index(cluster_col).loc[sampled_clusters]
        # But indices must be unique for loc to behave as expected? 
        # Actually loc with duplicates in index returns all matches. 
        # If clustered sampled is [A, A, B], loc[A] returns A rows twice. Correct.
        
        # Ensure df is indexed for speed if not already
        if df.index.name != cluster_col:
            # We don't want to mutate input df permanently
            # df_indexed = df.set_index(cluster_col)
            # But making a copy every loop is bad.
            # Strategy: Pass pre-indexed df if possible.
            pass
            
        # Fallback slow way for safety first implementation
        # sampled_df = df[df[cluster_col].isin(sampled_clusters)] # This is wrong, it doesn't duplicate
        
        # Right way:
        # 1. Get indices of all rows for each cluster
        # 2. Concatenate specific standard indices
        
        # Optimization: Pre-compute cluster indices
        # Doing it inside loop for now.
        
        # Fast resampling:
        current_sample = []
        # This is the bottleneck. 
        # Let's use a simpler approach for now or optimize later.
        # Actually, let's just do it cleanly.
        
        # Re-indexed approach
        # df_sorted = df.sort_values(cluster_col)
        # ... this is becoming complex for a utility.
        
        # Vectorized way:
        # Map cluster -> list of indices
        cluster_map = df.groupby(cluster_col).indices # dict: cluster -> int array of indices
        
        # Sample clusters
        sampled_clusters = np.random.choice(list(cluster_map.keys()), n_clusters, replace=True)
        
        # Collect indices
        sampled_indices = np.concatenate([cluster_map[c] for c in sampled_clusters])
        
        # Construct DF
        sampled_df = df.iloc[sampled_indices].copy()
        
        # Run Model
        try:
            # model_eligibility dummy list
            res = model_func(sampled_df, formula)
            if res is not None:
                boot_coeffs.append(res.params)
        except Exception:
            pass
            
    if not boot_coeffs:
        return None
        
    # Aggregate
    boot_df = pd.DataFrame(boot_coeffs)
    ci_lower = boot_df.quantile(0.025)
    ci_upper = boot_df.quantile(0.975)
    
    return pd.DataFrame({'ci_lower': ci_lower, 'ci_upper': ci_upper})

import subprocess
import shutil

def run_r_wrapper(script_path, args_list, timeout=None):
    """
    Executes an R script via Rscript.
    """
    if not shutil.which("Rscript"):
        print("Error: Rscript not found in PATH.", file=sys.stderr)
        return False
        
    cmd = ["Rscript", script_path] + args_list
    
    print(f"Executing R script: {script_path}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing R script {script_path}: {e}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout executing R script {script_path}", file=sys.stderr)
        return False

