import numpy as np
import pandas as pd

def calculate_concentration_metrics(ranks: pd.Series, units: pd.Series, prefix: str = "domain") -> dict:
    """
    Computes concentration metrics for a list of ranks and units (e.g., domains or URLs) based on ANALYSIS_PLAN_RQs.md Addendum B.
    
    Args:
        ranks (pd.Series): List of rank positions (integers).
        units (pd.Series): List of associated units (str) - usually domains or URLs.
        prefix (str): Prefix for metric names ('domain' or 'url').
        
    Returns:
        dict: {
            'n_results': int,
            'n_{prefix}s': int,
            '{prefix}_entropy_raw': float,
            '{prefix}_entropy_norm': float,
            '{prefix}_gini': float, 
            '{prefix}_is_monopoly': bool,
            'status': str
        }
    """
    # Create a DataFrame for processing
    df = pd.DataFrame({'rank': ranks, 'unit': units})
    
    # Filter valid ranks > 0
    df = df[df['rank'] > 0].sort_values('rank')
    
    n_results = len(df)
    
    # B.4 Minimum observation rule check
    status = "ok"
    if n_results < 5:
        status = "insufficient_data"
        
    if n_results == 0:
        return {
            'n_results': 0,
            f'n_{prefix}s': 0,
            f'{prefix}_entropy_raw': np.nan,
            f'{prefix}_entropy_norm': np.nan,
            f'{prefix}_gini': np.nan,
            f'{prefix}_is_monopoly': False,
            'status': "insufficient_data"
        }
    
    # B.1.1 Individual Visibility (Continuous Re-ranking 1..n)
    ranks_continuous = np.arange(1, n_results + 1)
    # v_r = 1/r
    item_visibility = 1.0 / ranks_continuous
    
    # Assign visibility back to items
    df['visibility'] = item_visibility
    
    # B.1.2 Unit Aggregation
    unit_visibility = df.groupby('unit')['visibility'].sum()
    n_units = len(unit_visibility)
    is_monopoly = (n_units == 1)
    
    # B.1.3 Unit Shares
    total_visibility = unit_visibility.sum()
    unit_shares = unit_visibility / total_visibility
    
    # B.2 Entropy
    # H = -sum(p_i * log2(p_i))
    shares_safe = unit_shares[unit_shares > 0]
    entropy_raw = -np.sum(shares_safe * np.log2(shares_safe))
    
    # H_norm = H / log2(|U|)
    if n_units > 1:
        entropy_norm = entropy_raw / np.log2(n_units)
    else:
        entropy_norm = 0.0
        
    # B.3 Gini
    if n_units <= 1:
        gini = np.nan
    else:
        # Sort shares ascending
        shares_asc = unit_shares.sort_values(ascending=True).values
        indices = np.arange(1, n_units + 1)
        weighted_sum = np.sum(indices * shares_asc)
        gini = (2 * weighted_sum) / n_units - (n_units + 1) / n_units
    
    return {
        'n_results': n_results,
        f'n_{prefix}s': n_units,
        f'{prefix}_entropy_raw': float(entropy_raw),
        f'{prefix}_entropy_norm': float(entropy_norm),
        f'{prefix}_gini': float(gini),
        f'{prefix}_is_monopoly': bool(is_monopoly),
        'status': status
    }

def calculate_semantic_dispersion(sim_scores: pd.Series, prefix: str = "sim") -> dict:
    """
    Computes distributional statistics for visualization and analysis.
    """
    clean_scores = sim_scores.dropna()
    if len(clean_scores) == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_p25': np.nan,
            f'{prefix}_p50': np.nan,
            f'{prefix}_p75': np.nan,
            f'{prefix}_skew': np.nan
        }
        
    stats = {
        f'{prefix}_mean': float(clean_scores.mean()),
        f'{prefix}_std': float(clean_scores.std()),
        f'{prefix}_p25': float(clean_scores.quantile(0.25)),
        f'{prefix}_p50': float(clean_scores.median()),
        f'{prefix}_p75': float(clean_scores.quantile(0.75)),
        f'{prefix}_skew': float(clean_scores.skew()) if len(clean_scores) > 2 else 0.0
    }
    return stats

def calculate_rank_binned_trends(df_group: pd.DataFrame, metric_col: str, prefix: str = "sim") -> dict:
    if metric_col not in df_group.columns:
        return {}
        
    from src.helpers.data_loader import get_rank_tiers
    tiers = get_rank_tiers()
    
    res = {}
    last_point = 0
    for point in tiers['logit_cut_points']:
        val = df_group[df_group['rank'] <= point][metric_col].mean()
        res[f'{prefix}_top{point}'] = float(val) if not pd.isna(val) else np.nan
        last_point = point
        
    val_rest = df_group[df_group['rank'] > last_point][metric_col].mean()
    res[f'{prefix}_rest'] = float(val_rest) if not pd.isna(val_rest) else np.nan
    
    return res
