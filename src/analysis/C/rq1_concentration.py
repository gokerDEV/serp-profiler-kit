import argparse
import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
import datetime

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import helper
from src.analysis.C.helpers.metrics import calculate_concentration_metrics

def generate_report(df: pd.DataFrame, qa_stats: dict, output_path: str, effect_df: pd.DataFrame):
    """
    Generates a human-readable markdown report for RQ1 Analysis (Domain-Level).
    """
    print(f"Generating report: {output_path}...", file=sys.stderr)
    
    valid_df = df[df['status'] == 'ok'].copy()
    
    # Engine Summary
    # Note: Gini mean will ignore NaN automatically (which is good, Monopolies excluded from inequality calc)
    agg_dict = {
        'domain_gini': ['count', 'mean', 'std', 'min', 'max'],
        'domain_entropy_norm': ['mean', 'std'],
        'n_domains': ['mean', 'min', 'max'],
        'domain_is_monopoly': ['sum', 'mean'] # Count of monopolies and rate
    }
    
    # Add semantic stats if available
    if 'sem_mean' in valid_df.columns:
        agg_dict.update({
            'sem_mean': ['mean', 'std'],
            'sem_std': ['mean'],
            'sem_skew': ['mean']
        })
        
    # Add rank trends if available
    from src.helpers.data_loader import get_rank_tiers
    tiers = get_rank_tiers()
    for point in tiers['logit_cut_points']:
        col = f'sem_top{point}'
        if col in valid_df.columns:
            agg_dict[col] = ['mean']
    if 'sem_rest' in valid_df.columns:
        agg_dict['sem_rest'] = ['mean']

    engine_stats = valid_df.groupby('search_engine').agg(agg_dict).round(4)
    
    # Flatten columns
    engine_stats.columns = ['_'.join(col).strip() for col in engine_stats.columns.values]
    engine_stats = engine_stats.reset_index()
    
    # Correlation Checks (QA B.5)
    # 1. Gini vs n_results
    corr1 = valid_df['domain_gini'].corr(valid_df['n_results'])
    # 2. Gini vs n_domains
    corr2 = valid_df['domain_gini'].corr(valid_df['n_domains'])
    # 3. Entropy Norm vs n_domains
    corr3 = valid_df['domain_entropy_norm'].corr(valid_df['n_domains'])
    
    # Extreme Examples
    # High Concentration = Low Entropy (Monopoly/Oligopoly)
    # Start with Monopoly (n_domains=1, Entropy=0) - sort by entropy_norm ASC
    top_conc = valid_df.nsmallest(10, 'domain_entropy_norm')[['search_term', 'search_engine', 'n_results', 'n_domains', 'domain_gini', 'domain_entropy_norm', 'domain_is_monopoly']]
    
    # High Diversity = High Entropy (Uniform distribution across many domains) - sort by entropy_norm DESC
    low_conc = valid_df.nlargest(10, 'domain_entropy_norm')[['search_term', 'search_engine', 'n_results', 'n_domains', 'domain_gini', 'domain_entropy_norm', 'domain_is_monopoly']]

    md = []
    md.append(f"# RQ1 Domain-Level Concentration Report")
    md.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Methodology:** Domain-aggregated visibility shares ($p_d$). Gini is NA for Monopolies ($|D|=1$).")
    md.append(f"**Normalization:** $H_{{norm}} = -\\sum p_i \\log_2(p_i) / \\log_2(|D|)$ where $|D|$ is count of active domains.\n")

    
    md.append(f"## 1. QA Summary")
    md.append(f"- **Total Groups (Query x Engine):** {qa_stats['total_groups']}")
    md.append(f"- **Valid Groups (>=5 results):** {qa_stats['valid_count']}")
    md.append(f"- **Insufficient Data (<5 results):** {qa_stats['insufficient_count']} ({qa_stats['insufficient_pct']:.2f}%)")
    
    md.append(f"\n### QA Correlations (|r| should be < 0.85)")
    def check_corr(val, name):
         status = "✅ Safe" if abs(val) < 0.85 else "⚠️ WARNING High Dependency"
         return f"- **{name}:** {val:.4f} ({status})"
         
    md.append(check_corr(corr1, "Gini vs n_results"))
    md.append(check_corr(corr2, "Gini vs n_domains"))
    md.append(check_corr(corr3, "Entropy Norm vs n_domains"))

    md.append(f"\n## 2. Distribution by Search Engine")
    md.append(f"*Note: Gini statistics exclude Monopolies (n_domains=1). Monopoly rate is shown in 'is_monopoly_mean'.*")
    md.append(engine_stats.to_markdown(index=False))
    
    if not effect_df.empty:
        md.append(f"\n## 3. Statistical Comparison (Effect Sizes)")
        md.append(f"Cohen's d values comparing engine distributions (Query-Weighted Mean Differences).")
        md.append(effect_df.round(4).to_markdown(index=False))
    
    md.append(f"\n## 4. Extreme Examples")
    
    md.append(f"\n### Top 10 Most Concentrated (Low Entropy - Monopoly/Oligopoly)")
    md.append(f"Sorted by `entropy_norm` ASC. Monopolies appear first (Entropy=0).")
    md.append(top_conc.to_markdown(index=False))
    
    md.append(f"\n### Top 10 Least Concentrated (High Entropy - Diverse)")
    md.append(f"Sorted by `entropy_norm` DESC. Queries with balanced visibility among multiple domains.")
    md.append(low_conc.to_markdown(index=False))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

def main():
    parser = argparse.ArgumentParser(description="Analysis Step C (RQ1): Rank Concentration & Semantic Dispersion")
    parser.add_argument("--dataset", default="data/dataset.parquet", help="Path to merged dataset")
    parser.add_argument("--out", default="data/analysis/C/rq1_concentration.parquet", help="Output path")
    parser.add_argument("--report", default=None, help="Output path for markdown report")
    parser.add_argument("--skip-r", action="store_true", help="Skip R plotting step")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset}...", file=sys.stderr)
    from src.helpers.data_loader import load_analysis_dataset
    try:
        df = load_analysis_dataset(args.dataset)
    except FileNotFoundError:
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)
        
    required_cols = ['search_term', 'search_engine', 'rank', 'domain'] # domain required now
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing} (Ensure 'domain' is in dataset)", file=sys.stderr)
        sys.exit(1)
        
    print(f"Processing {len(df)} records...", file=sys.stderr)
    
    import tldextract
      
    subsets = {'Full': df}
    if 'is_source_domain' in df.columns:
        subsets['NoSource'] = df[~df['is_source_domain']].copy()
        subsets['Source'] = df[df['is_source_domain']].copy()

    all_results = []
    
    from src.analysis.C.helpers.metrics import calculate_semantic_dispersion, calculate_rank_binned_trends
    from src.helpers.data_loader import get_analysis_features, get_feature_categories, get_rank_tiers
    analysis_feats = get_analysis_features()
    
    def apply_metrics(group):
        ranks = group['rank']
        domains = group['domain']
        
        # 1. Structural Concentration (Domain-Level - Primary)
        metrics = calculate_concentration_metrics(ranks, domains, prefix="domain")
        
        # 1b. Robustness: URL-Level Concentration (Dedup/Normalization)
        # We treat each unique URL as a "domain" equivalent for this check.
        # This checks if concentration is driven by domain aggregation or pure rank slot inequality.
        # Ideally, we should use normalized URLs. Assuming 'url' or similar is the unit.
        # If 'url' is not available, we skip.
        if 'url' in group.columns:
            url_metrics = calculate_concentration_metrics(ranks, group['url'], prefix="url")
            metrics.update(url_metrics)
        else:
             pass
        
        # 2. Features Dispersion (all continuous features)
        for feat in analysis_feats:
            if feat in group.columns:
                f_stats = calculate_semantic_dispersion(group[feat], prefix=feat)
                metrics.update(f_stats)
                t_stats = calculate_rank_binned_trends(group, feat, prefix=feat)
                metrics.update(t_stats)
            
        return pd.Series(metrics)

    all_effect_sizes = []
    all_viz_dfs = []
    
    out_dir = os.path.dirname(args.out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    for subset_name, current_df in subsets.items():
        if current_df.empty: continue
        print(f"\n--- Processing Subset: {subset_name} ---", file=sys.stderr)
        
        grouped = current_df.groupby(['search_term', 'search_engine'])
        result_df = grouped.apply(apply_metrics).reset_index()
        result_df['subset'] = subset_name
        all_results.append(result_df)
        
        engines = result_df['search_engine'].unique()
        effect_sizes = []
        import itertools
        for e1, e2 in itertools.combinations(engines, 2):
            g1 = result_df[result_df['search_engine'] == e1]
            g2 = result_df[result_df['search_engine'] == e2]
            
            metrics_to_test = ['domain_gini'] + [f"{f}_mean" for f in analysis_feats]
            for metric in metrics_to_test:
                if metric not in result_df.columns: continue
                m1, m2 = g1[metric].dropna().mean(), g2[metric].dropna().mean()
                s1, s2 = g1[metric].std(), g2[metric].std()
                n1, n2 = len(g1), len(g2)
                
                pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2)) if n1+n2>2 else 0
                d = (m1 - m2) / pooled_std if pooled_std > 0 else 0.0
                    
                effect_sizes.append({
                    'subset': subset_name,
                    'pair': f"{e1} vs {e2}",
                    'metric': metric,
                    'diff': m1 - m2,
                    'cohens_d': d,
                    'interpretation': 'Large' if abs(d) > 0.8 else ('Medium' if abs(d) > 0.5 else 'Small')
                })
        
        all_effect_sizes.extend(effect_sizes)
        effect_df = pd.DataFrame(effect_sizes)
        
        insufficient = result_df[result_df['status'] == 'insufficient_data']
        valid_df = result_df[result_df['status'] == 'ok']
        qa_stats = {
            'total_groups': len(result_df),
            'insufficient_count': len(insufficient),
            'insufficient_pct': len(insufficient) / len(result_df) * 100 if len(result_df) > 0 else 0,
            'valid_count': len(valid_df)
        }
        
        print(f"QA Summary ({subset_name}): Valid={qa_stats['valid_count']} Insufficient={qa_stats['insufficient_count']}")
        
        report_path_sub = str(Path(args.report or args.out).with_name(f"rq1_report_{subset_name}.md"))
        generate_report(result_df, qa_stats, report_path_sub, effect_df)
        
        if 'sim_content' in current_df.columns:
            all_viz_dfs.append((subset_name, current_df[['search_term', 'search_engine', 'rank', 'sim_content']].dropna(subset=['sim_content']).copy()))

    final_result_df = pd.concat(all_results, ignore_index=True)
    final_effect_df = pd.DataFrame(all_effect_sizes)
    
    print(f"\nSaving results to {args.out}...", file=sys.stderr)
    final_result_df.to_parquet(args.out, index=False)
    
    # --- Visualization Data Export (RQ1 Extra) ---
    # Export data for src/generators (JSON format)
    # We need:
    # 1. Density/ECDF data: Raw sim_content by search_engine (downsampled if needed to keep size low, but <100k is fine)
    # 2. Trends: Aggregated mean/se by rank bin
    
    if all_viz_dfs:
        print(f"Preparing visualization data for src/generators...", file=sys.stderr)
        viz_data = {"meta": {
            "generated_at": datetime.datetime.now().isoformat(),
            "desc": "RQ1 Visualization Data: Semantic Similarity Distribution and Trends"
        }, "distribution": [], "trends": []}
        
        tiers = get_rank_tiers()
        
        for subset_name, viz_df in all_viz_dfs:
            dist_data = viz_df[['search_engine', 'sim_content']].copy()
            dist_data['subset'] = subset_name
            viz_data["distribution"].extend(dist_data.to_dict(orient='records'))
            
            viz_df['rank_bin'] = pd.cut(viz_df['rank'], bins=tiers['bins'], labels=tiers['labels'])
            agg = viz_df.groupby(['search_engine', 'rank_bin'])['sim_content'].agg(['mean', 'std', 'count']).reset_index()
            agg['se'] = agg['std'] / np.sqrt(agg['count'])
            agg['ci_lower'] = agg['mean'] - 1.96 * agg['se']
            agg['ci_upper'] = agg['mean'] + 1.96 * agg['se']
            agg['subset'] = subset_name
            
            viz_data["trends"].extend(agg.to_dict(orient='records'))
            
        viz_json_path = str(Path(args.out).parent / "rq1_viz.json")
        import json
        with open(viz_json_path, 'w') as f:
            json.dump(viz_data, f)
            
        print(f"Exported visualization data to {viz_json_path}", file=sys.stderr)
        
    # --- Output Overall 4x11x3x3x3 Target DataFrame ---
    print("Aggregating overall feature trends for all categories/features...", file=sys.stderr)
    all_feature_trends = []
    tiers = get_rank_tiers()
    cat_map = get_feature_categories()
    
    for subset_name, current_df in subsets.items():
        if current_df.empty: continue
        current_df['rank_bin'] = pd.cut(current_df['rank'], bins=tiers['bins'], labels=tiers['labels'])
        
        for feat in analysis_feats:
            if feat not in current_df.columns: continue
            
            valid_rows = current_df.dropna(subset=[feat])
            if valid_rows.empty: continue
            
            agg = valid_rows.groupby(['search_engine', 'rank_bin'], observed=False)[feat].agg(['mean', 'std', 'count']).reset_index()
            agg['se'] = agg['std'] / np.sqrt(agg['count'])
            agg['ci_lower'] = agg['mean'] - 1.96 * agg['se']
            agg['ci_upper'] = agg['mean'] + 1.96 * agg['se']
            agg['feature'] = feat
            agg['subset'] = subset_name
            
            all_feature_trends.append(agg)
            
    if all_feature_trends:
        trends_df = pd.concat(all_feature_trends, ignore_index=True)
        trends_df['category'] = trends_df['feature'].map(cat_map)
        
        trends_path = str(Path(args.out).parent / "rq1_feature_trends.csv")
        trends_df.to_csv(trends_path, index=False)
        print(f"Exported overall feature trends to {trends_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
