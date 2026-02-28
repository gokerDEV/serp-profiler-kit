import pandas as pd
import numpy as np
import sys

def analyze_health(df, keys):
    print("\n## Data Health & Integrity")
    print("\nConsistency check between Index(Status=OK) and Feature Extraction(Status).")
    print("If a record is OK in index but 'skipped'/'fail'/'miss' in feature, it indicates an extraction issue.\n")
    
    health_stats = []
    
    for key in keys:
        col = f'status_{key}'
        if col in df.columns:
            counts = df[col].fillna('missing_record').value_counts().to_dict()
            total = len(df)
            
            # Calculate match rate (ok / total)
            ok_count = counts.get('ok', 0)
            ok_pct = (ok_count / total) * 100
            
            row = {'Feature Set': key, 'Match Rate': f"{ok_pct:.1f}%", **counts}
            health_stats.append(row)
            
    if health_stats:
        health_df = pd.DataFrame(health_stats)
        print("```text")
        print(health_df.fillna(0).to_string(index=False))
        print("```\n")
    else:
        print("No feature status columns found.\n")

def tag_outliers_iqr(df, metrics, threshold=3.0):
    """
    Returns a boolean Series where True indicates an outlier based on IQR.
    """
    if not metrics:
        return pd.Series(False, index=df.index)
        
    is_outlier = pd.Series(False, index=df.index)
    
    for col in metrics:
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        if data.empty:
            continue
            
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
        
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
        # Identify indices of outliers
        outlier_indices = data[(data < lower) | (data > upper)].index
        is_outlier.loc[outlier_indices] = True
        
    return is_outlier

def calculate_outlier_flags(df, metric_groups, threshold=3.0):
    """
    Generates outlier flags for provided metric groups.
    Returns a DataFrame with columns: is_outlier_{group_name}
    """
    flags_list = []
    
    for group, metrics in metric_groups.items():
        mask = tag_outliers_iqr(df, metrics, threshold=threshold)
        flag_series = pd.Series(np.where(mask, 'outlier', 'ok'), index=df.index, name=f'is_outlier_{group}')
        flags_list.append(flag_series)
        
    flags_df = pd.concat(flags_list, axis=1)
    return flags_df

def report_distribution_comparison(df, flags_df, metric_groups):
    print("\n## Distribution Analysis (With vs Without Outliers)")
    print("Comparing statistical properties. 'Clean' dataset excludes records flagged as 'outlier' in ANY category (Extreme Outliers > 3*IQR).")
    
    # Determine "Clean" subset (Records that are OK in ALL categories)
    # Using 'ok' string check
    is_clean = (flags_df == 'ok').all(axis=1)
    clean_df = df[is_clean]
    
    print(f"\nTotal Records: {len(df)}")
    print(f"Clean Records: {len(clean_df)} ({len(clean_df)/len(df) if len(df)>0 else 0:.1%})")
    print(f"Outliers Removed: {len(df) - len(clean_df)}")
    
    for group, metrics in metric_groups.items():
        print(f"\n### {group.capitalize()} Metrics")
        
        stats_list = []
        for col in metrics:
            if col not in df.columns: continue
            
            # Full Stats
            full_s = df[col].dropna()
            clean_s = clean_df[col].dropna()
            
            if full_s.empty: continue

            # Append rows for table
            stats_list.append({
                'Metric': col,
                'Scope': 'All',
                'Count': len(full_s),
                'Mean': full_s.mean(),
                'Median': full_s.median(),
                'Std': full_s.std(),
                'Max': full_s.max()
            })
            if not clean_s.empty:
                stats_list.append({
                    'Metric': col,
                    'Scope': 'Clean',
                    'Count': len(clean_s),
                    'Mean': clean_s.mean(),
                    'Median': clean_s.median(),
                    'Std': clean_s.std(),
                    'Max': clean_s.max()
                })
            else:
                 stats_list.append({
                    'Metric': col,
                    'Scope': 'Clean',
                    'Count': 0,
                    'Mean': 0, 'Median': 0, 'Std': 0, 'Max': 0
                })
        
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            # Reorder columns
            cols = ['Metric', 'Scope', 'Count', 'Mean', 'Median', 'Std', 'Max']
            stats_df = stats_df[cols]
            
            print("```text")
            print(stats_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x) if isinstance(x, (float, int)) else x))
            print("```")

# Legacy function kept for compatibility but updated to be useful
def analyze_outliers(df, cols_to_check):
    print("\n## Initial Outlier Detection (Quick Scan)")
    print("(See detailed distribution comparison below)")
    pass 
