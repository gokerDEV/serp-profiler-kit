import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
import glob
import datetime

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# --- LATEX HELPERS ---

def tex_escape(text):
    """Escapes underscores and other tex-special chars"""
    if not isinstance(text, str): return text
    return text.replace("_", "\\_").replace("%", "\\%")

def wrap_latex_table(content, caption, label, column_specs, tablenotes=None):
    """Wraps content in LaTeX table environment"""
    # Enforce extracolsep textwidth stretch for tabular*
    if "@{\\extracolsep{\\fill}}" not in column_specs:
        column_specs = column_specs[0] + "@{\\extracolsep{\\fill}}" + column_specs[1:]
        
    notes_block = ""
    if tablenotes:
        notes_block = f"\\begin{{tablenotes}}[flushleft]\n\\scriptsize\n\\item {tablenotes}\n\\end{{tablenotes}}\n"

    template = f"""\\begin{{table*}}[htbp!]
\\centering
\\begin{{threeparttable}}
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{tabular*}}{{\\textwidth}}{{{column_specs}}}
\\toprule
{content}
\\bottomrule
\\end{{tabular*}}
{notes_block}\\end{{threeparttable}}
\\end{{table*}}"""
    return template

def get_latest_dataset():
    files = glob.glob("data/dataset-*.parquet")
    if not files:
        return None
    return max(files, key=os.path.getctime)

def format_p_val(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "<.001"
    return f"{p:.3f}"

def format_ci(l, u):
    if pd.isna(l) or pd.isna(u): return "-"
    return f"[{l:.3f}, {u:.3f}]"

# --- TABLE GENERATORS ---

def generate_dataset_distribution_table(out_dir):
    ds_path = get_latest_dataset()
    if not ds_path: return
    df = pd.read_parquet(ds_path)
    
    ok_df = df.copy() # Dataset already contains only accepted records
    from src.helpers.data_loader import get_rank_tiers
    tiers_info = get_rank_tiers()
    ok_df['rank_bin'] = pd.cut(ok_df['rank'], bins=tiers_info['bins'], labels=tiers_info['labels'])
    
    pivot_full = pd.crosstab(ok_df['search_engine'], ok_df['rank_bin'], margins=True, margins_name='Total')
    
    # Source domain subset
    top_domain = ok_df['domain'].value_counts().index[0]
    if 'is_source_domain' in ok_df.columns:
        subset_source = ok_df[ok_df['is_source_domain'] == True].copy()
    else:
        subset_source = ok_df[ok_df['domain'] == top_domain].copy()
    pivot_source = pd.crosstab(subset_source['search_engine'], subset_source['rank_bin'], margins=True, margins_name='Total')
    
    cols = list(pivot_full.columns)
    col_str = " & ".join([tex_escape(str(c)) for c in cols])
    rows = [f"\\sbf{{Search Engine}} & {col_str} \\\\", "\\dmidrule"]
    
    rows.append("\\multicolumn{5}{l}{\\sit{Subset: Full}} \\\\")
    for idx, row in pivot_full.iterrows():
        r_vals = " & ".join([f"{int(x):,}" for x in row])
        rows.append(f"{tex_escape(str(idx).capitalize())} & {r_vals} \\\\")
        
    rows.append("\\multicolumn{5}{l}{\\sit{Subset: Source}} \\\\")
    for idx, row in pivot_source.iterrows():
        r_vals = " & ".join([f"{int(x):,}" for x in row])
        rows.append(f"{tex_escape(str(idx).capitalize())} & {r_vals} \\\\")
        
    content = "\n".join(rows)
    notes = "Counts reflect the final accepted dataset after all cleaning and outlier removal steps.\n\\item Subset: Source is domain visibility for " + tex_escape(top_domain) + "."
    latex = wrap_latex_table(content, "Dataset Distribution: Accepted Results by Rank", "tab:dataset_dist", "l" + "c" * len(cols), tablenotes=notes)
    with open(f"{out_dir}/table_dataset_distribution.tex", "w") as f:
        f.write(latex)



def generate_feature_match_rate_table(out_dir):
    ds_path = get_latest_dataset()
    if not ds_path: return
    df = pd.read_parquet(ds_path)
    
    features = {
        'A (Runtime)': 'ttfb_ms',
        'B (Accessibility)': 'axe_score',
        'C (HTML Structure)': 'http_status',
        'D (Readability)': 'flesch_reading_ease',
        'E (Semantic)': 'sim_content'
    }
    
    rows = ["\\sbf{Feature Set} & \\sbf{Total OK Records} & \\sbf{Matched} & \\sbf{Missing} & \\sbf{Match Rate (\\%)} \\\\", "\\dmidrule"]
    
    ok_df = df[df['status'] == 'ok']
    total_ok = len(ok_df)
    
    if total_ok == 0: return
    
    for fname, col in features.items():
        if col in ok_df.columns:
            matched = ok_df[col].notna().sum()
            missing = total_ok - matched
            rate = (matched / total_ok) * 100
        else:
            matched, missing, rate = 0, total_ok, 0.0
            
        rows.append(f"{fname} & {total_ok:,} & {matched:,} & {missing:,} & {rate:.1f}\\% \\\\")
        
    content = "\n".join(rows)
    latex = wrap_latex_table(content, "Feature Match Rates for OK Records", "tab:feature_match", "lcccc")
    with open(f"{out_dir}/table_feature_match.tex", "w") as f:
        f.write(latex)

def generate_html_size_stats_table(out_dir):
    idx_path = "data/index.parquet"
    if not os.path.exists(idx_path): return
    df = pd.read_parquet(idx_path)
    
    if 'html_size_bytes' not in df.columns: return
    
    all_stats = df['html_size_bytes'].describe()
    ok_stats = df[df['status'] == 'ok']['html_size_bytes'].describe()
    
    rows = ["\\sbf{Subset} & \\sbf{Count} & \\sbf{Mean} & \\sbf{Std} & \\sbf{Min} & \\sbf{Median} & \\sbf{Max} \\\\", "\\dmidrule"]
    
    rows.append(f"All Records & {int(all_stats['count']):,} & {all_stats['mean']:.0f} & {all_stats['std']:.0f} & {all_stats['min']:.0f} & {all_stats['50%']:.0f} & {all_stats['max']:.0f} \\\\")
    rows.append(f"OK Only & {int(ok_stats['count']):,} & {ok_stats['mean']:.0f} & {ok_stats['std']:.0f} & {ok_stats['min']:.0f} & {ok_stats['50%']:.0f} & {ok_stats['max']:.0f} \\\\")
    
    ds_path = get_latest_dataset()
    if ds_path and os.path.exists(ds_path):
        ds_df = pd.read_parquet(ds_path)
        if 'html_size_bytes' in ds_df.columns:
            ds_stats = ds_df['html_size_bytes'].describe()
            rows.append(f"Accepted Dataset & {int(ds_stats['count']):,} & {ds_stats['mean']:.0f} & {ds_stats['std']:.0f} & {ds_stats['min']:.0f} & {ds_stats['50%']:.0f} & {ds_stats['max']:.0f} \\\\")

    content = "\n".join(rows)
    latex = wrap_latex_table(content, "Descriptive Statistics: HTML File Size (Bytes)", "tab:html_size", "lcccccc")
    with open(f"{out_dir}/table_html_size.tex", "w") as f:
        f.write(latex)

def generate_model_eligibility_table(out_dir):
    path = "data/analysis/D/model_eligibility_report.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    rows = ["\\sbf{Model ID} & \\sbf{Total Obs} & \\sbf{Used Obs} & \\sbf{Loss (\\%)} & \\sbf{Warning Flags} \\\\", "\\dmidrule"]
    for _, row in df.iterrows():
        model_id = tex_escape(str(row.get('model_id', '')))
        total = int(row.get('N_pre', 0))
        used = int(row.get('N_complete_case', 0))
        loss = float(row.get('loss_pct', 0.0))
        
        warnings = []
        if row.get('status') != 'ok': warnings.append(str(row.get('status')))
        if row.get('cluster_warning_flag'): warnings.append("ClusterWarning")
        if row.get('indicator_sensitivity_triggered'): warnings.append("IndicatorSens")
        
        flags = tex_escape(", ".join(warnings)) if warnings else "None"
        rows.append(f"{model_id} & {total:,} & {used:,} & {loss:.1f}\\% & {flags} \\\\")
        
    content = "\n".join(rows)
    latex = wrap_latex_table(content, "Model Eligibility and Data Loss Report", "tab:model_elig", "lrrrl")
    with open(f"{out_dir}/table_model_eligibility.tex", "w") as f:
        f.write(latex)

def generate_rq1_engine_concentration_table(out_dir):
    # Try to read rq1 output from parquet or compute directly
    rq1_path = "data/analysis/C/rq1_concentration.parquet"
    if os.path.exists(rq1_path):
        df = pd.read_parquet(rq1_path)
    else:
        return
        
    ok_df = df[df['status'] == 'ok'] if 'status' in df.columns else df
    if 'subset' not in ok_df.columns: ok_df['subset'] = 'Full'
    
    if 'domain_gini' not in ok_df.columns: return
    
    stats = ok_df.groupby(['subset', 'search_engine']).agg({
        'domain_gini': 'mean',
        'domain_entropy_norm': 'mean',
        'domain_is_monopoly': 'mean'
    }).reset_index()
    
    rows = ["\\sbf{Search Engine} & \\sbf{Mean Gini} & \\sbf{Mean Norm. Entropy} & \\sbf{Monopoly Rate (\\%)} \\\\", "\\dmidrule"]
    for subset, group in stats.groupby('subset'):
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\sit{{Subset: {tex_escape(subset)}}}}} \\\\")
        for _, row in group.iterrows():
            eng = str(row['search_engine']).capitalize()
            m_rate = row['domain_is_monopoly'] * 100
            rows.append(f"{eng} & {row['domain_gini']:.3f} & {row['domain_entropy_norm']:.3f} & {m_rate:.1f}\\% \\\\")
        rows.append("\\midrule")
    if rows[-1] == "\\midrule": rows.pop()
        
    content = "\n".join(rows)
    latex = wrap_latex_table(content, "RQ1: Engine Rank Concentration Summary", "tab:rq1_summary", "lccc", tablenotes="Mean Gini: domain-level Gini coefficient computed per query over share of results in Top 20 (higher = more concentrated). Monopoly rate: percentage of queries where a single domain occupies all Top 3 positions.")
    with open(f"{out_dir}/table_rq1_summary.tex", "w") as f:
        f.write(latex)

def generate_rq1_effect_sizes_table(out_dir):
    # We can compute effect sizes on the fly if file is missing
    rq1_path = "data/analysis/C/rq1_concentration.parquet"
    if not os.path.exists(rq1_path): return
    df = pd.read_parquet(rq1_path)
    ok_df = df[df['status'] == 'ok'] if 'status' in df.columns else df
    
    if 'subset' not in ok_df.columns: ok_df['subset'] = 'Full'
    engines = ok_df['search_engine'].dropna().unique()
    subsets = ok_df['subset'].dropna().unique()
    
    rows = ["\\sbf{Comparison Pair} & \\sbf{Metric} & \\sbf{Mean Diff} & \\sbf{Cohen's d} & \\sbf{Interpretation} \\\\", "\\dmidrule"]
    import itertools
    for sub in subsets:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\sit{{Subset: {tex_escape(sub)}}}}} \\\\")
        sub_df = ok_df[ok_df['subset'] == sub]
        for e1, e2 in itertools.combinations(engines, 2):
            g1 = sub_df[sub_df['search_engine'] == e1]
            g2 = sub_df[sub_df['search_engine'] == e2]
            for metric in ['domain_gini', 'sim_content']:
                if metric not in sub_df.columns: continue
                valid1 = g1[metric].dropna()
                valid2 = g2[metric].dropna()
                m1, m2 = valid1.mean(), valid2.mean()
                s1, s2 = valid1.std(), valid2.std()
                n1, n2 = len(valid1), len(valid2)
                
                if n1 < 2 or n2 < 2: continue
                
                pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
                if pooled_std == 0: continue
                d = (m1 - m2) / pooled_std
                interp = 'Large' if abs(d) > 0.8 else ('Medium' if abs(d) > 0.5 else 'Small')
                
                rows.append(f"{e1.capitalize()} vs {e2.capitalize()} & {tex_escape(metric)} & {m1-m2:.3f} & {d:.3f} & {interp} \\\\")
        rows.append("\\midrule")
    if rows[-1] == "\\midrule": rows.pop()
            
    content = "\n".join(rows)
    latex = wrap_latex_table(content, "RQ1: Engine Pairwise Comparisons (Effect Sizes)", "tab:rq1_effects", "llcll")
    with open(f"{out_dir}/table_rq1_effects.tex", "w") as f:
        f.write(latex)

def generate_cooks_d_table(out_dir):
    path = "data/analysis/D/cooks_error_distribution.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    # Assuming columns: model_id, max_cooks_d, mean_cooks_d, outliers_pct
    rows = ["\\sbf{Model ID} & \\sbf{Max Cook's D} & \\sbf{Mean Cook's D} & \\sbf{Outliers (\\%)} \\\\", "\\dmidrule"]
    for _, row in df.iterrows():
        rows.append(f"{tex_escape(row['model_id'])} & {row['max_cooks_d']:.3f} & {row['mean_cooks_d']:.3f} & {row['outliers_pct']:.2f}\\% \\\\")
    content = "\n".join(rows)
    latex = wrap_latex_table(content, "RQ2: Error Distribution and Effect Analysis (Cook's D)", "tab:cooks_d", "lccc")
    with open(f"{out_dir}/table_cooks_d.tex", "w") as f:
        f.write(latex)

def generate_confirmatory_table(out_dir):
    conf_path = "data/analysis/D/confirmatory_coeffs_r.csv"
    if not os.path.exists(conf_path):
        conf_path = "data/analysis/D/confirmatory_coeffs.csv"
        
    if os.path.exists(conf_path):
        df = pd.read_csv(conf_path)
        if 'effect_size' not in df.columns: df['effect_size'] = df.get('coef', 0)
        if 'p_raw' not in df.columns: df['p_raw'] = df.get('pval', 0)
        if 'ci_lower_95' not in df.columns:
             df['ci_lower_95'] = df.get('ci_lower', 0)
             df['ci_upper_95'] = df.get('ci_upper', 0)
             
        if 'subset' not in df.columns: df['subset'] = 'Full'
        
        df = df[~df['term'].astype(str).str.contains('search_engine')]
        df = df[df['term'] != '(Intercept)']
        
        rows = ["\\sbf{Model} & \\sbf{Predictor} & \\sbf{Estimate ($\\beta^*$)} & \\sbf{CI95} & \\sbf{p-val} & \\sbf{Flag} \\\\", "\\dmidrule"]
        
        for subset, sub_grp in df.groupby('subset'):
            rows.append(f"\\multicolumn{{6}}{{l}}{{\\sit{{Subset: {tex_escape(subset)}}}}} \\\\")
            for mid, grp in sub_grp.groupby('model_id'):
                label = str(mid).replace("_R_FE", "")
                label = tex_escape(label)
                is_first = True
                for _, row in grp.iterrows():
                    term = tex_escape(row['term'])
                    pract = "Yes" if row.get('practical_flag', False) else "No"
                    p_m = label if is_first else ""
                    rows.append(f"{p_m} & {term} & {row['effect_size']:.3f} & {format_ci(row['ci_lower_95'], row['ci_upper_95'])} & {format_p_val(row['p_raw'])} & {pract} \\\\")
                    is_first = False
                rows.append("\\thinrule")
            if rows and rows[-1] == "\\thinrule":
                rows[-1] = "\\midrule"
            
        if rows and rows[-1] == "\\midrule": rows.pop()
        content = "\n".join(rows)
        latex = wrap_latex_table(content, "Confirmatory Regression Results", "tab:confirmatory_main", "l@{\\extracolsep{\\fill}}lrlll")
        with open(f"{out_dir}/table_confirmatory_main.tex", "w") as f:
            f.write(latex)

def generate_supplementary_confirmatory_table(out_dir):
    conf_path = "data/analysis/D/supplementary_coeffs_r.csv"
    if not os.path.exists(conf_path):
        conf_path = "data/analysis/D/supplementary_coeffs.csv"
        
    if os.path.exists(conf_path):
        df = pd.read_csv(conf_path)
        if 'effect_size' not in df.columns: df['effect_size'] = df.get('coef', 0)
        if 'p_raw' not in df.columns: df['p_raw'] = df.get('pval', 0)
        if 'ci_lower_95' not in df.columns:
             df['ci_lower_95'] = df.get('ci_lower', 0)
             df['ci_upper_95'] = df.get('ci_upper', 0)
             
        if 'subset' not in df.columns: df['subset'] = 'Full'
        
        df = df[~df['term'].astype(str).str.contains('search_engine')]
        df = df[df['term'] != '(Intercept)']
        
        rows = ["\\sbf{Model} & \\sbf{Predictor} & \\sbf{Estimate ($\\beta^*$)} & \\sbf{CI95} & \\sbf{p-val} & \\sbf{Flag} \\\\", "\\dmidrule"]
        
        for subset, sub_grp in df.groupby('subset'):
            rows.append(f"\\multicolumn{{6}}{{l}}{{\\sit{{Subset: {tex_escape(subset)}}}}} \\\\")
            for mid, grp in sub_grp.groupby('model_id'):
                label = str(mid).replace("_R_FE_Full", "").replace("_R_FE_NoSource", "").replace("_R_FE_Source", "").replace("_R_FE", "")
                label = tex_escape(label)
                is_first = True
                for _, row in grp.iterrows():
                    term = tex_escape(row['term'])
                    pract = "Yes" if row.get('practical_flag', False) else "No"
                    p_m = label if is_first else ""
                    rows.append(f"{p_m} & {term} & {row['effect_size']:.3f} & {format_ci(row['ci_lower_95'], row['ci_upper_95'])} & {format_p_val(row['p_raw'])} & {pract} \\\\")
                    is_first = False
                rows.append("\\thinrule")
            if rows and rows[-1] == "\\thinrule":
                rows[-1] = "\\midrule"
            
        if rows and rows[-1] == "\\midrule": rows.pop()
        content = "\n".join(rows)
        latex = wrap_latex_table(content, "Supplementary Regression Results", "tab:confirmatory_supplementary", "l@{\\extracolsep{\\fill}}lrlll")
        with open(f"{out_dir}/table_confirmatory_supplementary.tex", "w") as f:
            f.write(latex)

def generate_nested_model_table(out_dir):
    path = "data/analysis/D/nested_model_fit.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    rows = ["\\sbf{Model} & \\sbf{Added Block} & \\sbf{$\\Delta R^2$} & \\sbf{$\\Delta$ AIC} & \\sbf{$\\Delta$ BIC} & \\sbf{LR Test $p$} & \\sbf{Practical Gain} \\\\", "\\dmidrule"]
    
    for _, row in df.iterrows():
        m_id = tex_escape(str(row['Model']))
        block = tex_escape(str(row['Added_Block']))
        dr2 = row['Delta_R2']
        daic = row['Delta_AIC']
        dbic = row['Delta_BIC']
        lr_p = row['LR_p']
        
        # Practical gain flag if delta R2 >= 0.01
        pract = "Yes" if dr2 >= 0.01 else "No"
        
        rows.append(f"{m_id} & {block} & {dr2:.4f} & {daic:.1f} & {dbic:.1f} & {format_p_val(lr_p)} & {pract} \\\\")
        
    content = "\n".join(rows)
    note = "$\\Delta R^2$, $\\Delta$ AIC, and $\\Delta$ BIC represent changes compared to the baseline (Semantics vs. Engine-only, others vs. Semantics). Practical Gain is flagged for $\\Delta R^2 \\geq 0.01$."
    latex = wrap_latex_table(content, "Nested Model Fit Comparison (RQ3--RQ5)", "tab:nested_model_fit", "l@{\\extracolsep{\\fill}}lrrrrr", tablenotes=tex_escape(note))
    with open(f"{out_dir}/table_nested_model_fit.tex", "w") as f:
        f.write(latex)

def generate_ablation_predictive_table(out_dir):
    abl_path = "data/analysis/H/ablation_predictive.csv"
    if os.path.exists(abl_path):
        df = pd.read_csv(abl_path)
        if 'subset' not in df.columns: df['subset'] = 'Full'
        
        rows = ["\\sbf{Ablated Block} & \\sbf{Mean NDCG@10} & \\sbf{$\\Delta$ NDCG [95\\% CI]} & \\sbf{Family} \\\\", "\\dmidrule"]
        
        for sub_name, sub_grp in df.groupby('subset'):
            rows.append(f"\\multicolumn{{4}}{{l}}{{\\sit{{Subset: {tex_escape(sub_name)}}}}} \\\\")
            for _, row in sub_grp.iterrows():
                if row['set_name'] == 'Full Model':
                    delta_str = "---"
                else:
                    eff = row.get('effect_size', 0.0)
                    ci_lo = row.get('ci_lower_95', eff)
                    ci_up = row.get('ci_upper_95', eff)
                    delta_str = f"{eff:.4f} [{ci_lo:.4f}, {ci_up:.4f}]"

                rows.append(f"{tex_escape(row['set_name'])} & {row['ndcg_mean']:.4f} & {delta_str} & {row['model_family']} \\\\")
            rows.append("\\midrule")
            
        if rows[-1] == "\\midrule": rows.pop()
        content = "\n".join(rows)
        note = "$\\Delta$ represents the difference in NDCG@10 when a block is ablated. 95\\% CIs are estimated via query-cluster bootstrap over 1000 resamples."
        latex = wrap_latex_table(content, "Ablation Study: Predictive Success (LTR)", "tab:ablation_pred", "lccc", tablenotes=note)
        with open(f"{out_dir}/table_ablation_predictive.tex", "w") as f:
            f.write(latex)

def generate_ablation_stability_table(out_dir):
    path = "data/analysis/H/ablation_stability_r.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        rows = ["\\sbf{Comparison} & \\sbf{Avg \\% Change} & \\sbf{Unstable Count} & \\sbf{Unstable Variables} \\\\", "\\dmidrule"]
        for _, row in df.iterrows():
            comp = tex_escape(str(row.get('comparison', '')))
            avg_change = row.get('avg_pct_change', 0)
            u_count = row.get('unstable_count', 0)
            u_vars = row.get('unstable_vars', '')
            
            if pd.isna(u_vars): u_vars = 'None'
            u_vars_str = tex_escape(str(u_vars).replace(';', ', '))
            if not u_vars_str.strip(): u_vars_str = 'None'
            
            rows.append(f"{comp} & {avg_change:.1f}\\% & {int(u_count)} & {u_vars_str} \\\\")
            
        content = "\n".join(rows)
        latex = wrap_latex_table(content, "Coefficient Stability Analysis (Block Comparisons)", "tab:ablation_stab", "llcl")
        with open(f"{out_dir}/table_ablation_stability.tex", "w") as f:
            f.write(latex)

def _build_feature_trend_table(df, caption, label, filename, out_dir):
    rank_order = ['Top 1-3', 'Rank 4-10', 'Rank 11-20']
    
    def format_mean_ci(r):
        from src.helpers.data_loader import get_feature_multipliers
        mults = get_feature_multipliers()
        feature = r['feature']
        mult = mults.get(feature, 1.0)
        
        mean = r['mean'] * mult
        lo = r['ci_lower'] * mult
        up = r['ci_upper'] * mult
        
        # If multiplier was negative, swap lower and upper bounds
        if mult < 0:
            lo, up = up, lo
            
        return f"{mean:.3f} [{lo:.3f}, {up:.3f}]"
        
    df['formatted'] = df.apply(format_mean_ci, axis=1)
    
    pivot = df.pivot_table(index=['subset', 'feature', 'search_engine'], columns='rank_bin', values='formatted', aggfunc='first').reset_index()
    
    cols = ['subset', 'feature', 'search_engine'] + [r for r in rank_order if r in pivot.columns]
    pivot = pivot[cols]
    
    num_cols = len(cols) - 1
    rows = ["\\sbf{Feature} & \\sbf{Engine} & " + " & ".join([f"\\sbf{{{tex_escape(c)}}}" for c in cols[3:]]) + " \\\\", "\\dmidrule"]
    
    for subset, sub_grp in pivot.groupby('subset'):
        rows.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\sit{{Subset: {tex_escape(subset)}}}}} \\\\")
        for feature, grp in sub_grp.groupby('feature'):
            is_first_feat = True
            for _, row in grp.iterrows():
                feat_str = tex_escape(feature) if is_first_feat else ""
                eng_str = str(row['search_engine']).capitalize()
                
                vals = []
                for c in cols[3:]:
                    vals.append(str(row[c]) if pd.notna(row[c]) else "-")
                    
                val_str = " & ".join(vals)
                rows.append(f"{feat_str} & {eng_str} & {val_str} \\\\")
                is_first_feat = False
            rows.append("\\thinrule")
        if rows and rows[-1] == "\\thinrule":
            rows[-1] = "\\midrule"
        
    if rows and rows[-1] == "\\midrule":
        rows.pop()
        
    content = "\n".join(rows)
    col_spec = "ll" + "c" * (len(cols) - 3)
    latex = wrap_latex_table(content, caption, label, col_spec)
    with open(f"{out_dir}/{filename}", "w") as f:
        f.write(latex)

def generate_rq1_feature_trends_non_semantic_table(out_dir):
    path = "data/analysis/C/rq1_feature_trends.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    df_ns = df[df['category'] != 'semantic'].copy()
    if df_ns.empty: return
    
    _build_feature_trend_table(df_ns, "RQ1: Rank-Binned Trends for Non-Semantic Features", "tab:rq1_trends_non_semantic", "table_rq1_trends_non_semantic.tex", out_dir)

def generate_rq1_feature_trends_semantic_table(out_dir):
    path = "data/analysis/C/rq1_feature_trends.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    df_s = df[df['category'] == 'semantic'].copy()
    if df_s.empty: return
    
    _build_feature_trend_table(df_s, "RQ1: Rank-Binned Trends for Semantic Similarity Features", "tab:rq1_trends_semantic", "table_rq1_trends_semantic.tex", out_dir)

def generate_rq8_robustness_table_main(out_dir):
    path = "data/analysis/G/robustness_coeffs_r.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    schema_terms = list(cat_map.keys())
    
    df = df[df['term'].isin(schema_terms)].copy()
    
    baseline_id = 'RQ8_Robustness_Baseline_R'
    win_id = 'RQ8_Robustness_Winsorized_R'
    cluster_id = 'RQ8_Robustness_2WayCluster_R'
    
    header = "\\sbf{Predictor} & \\sbf{Baseline $\\beta^*$ (CI95)} & \\sbf{Winsorized} & \\sbf{Two-way Clustered} & \\sbf{Sign Match} & \\sbf{Pract. Match} \\\\"
    
    def get_val(r_df):
        if r_df.empty: return "---", None, None
        r = r_df.iloc[0]
        return f"{r['effect_size']:.3f} [{r['ci_lower_95']:.3f}, {r['ci_upper_95']:.3f}]", r['effect_size'], r.get('practical_flag', False)

    table_rows = [header, "\\dmidrule"]
    
    for term in schema_terms:
        sub = df[df['term'] == term]
        if sub.empty: continue
        
        b_str, b_ef, b_pr = get_val(sub[sub['model_id'] == baseline_id])
        w_str, w_ef, w_pr = get_val(sub[sub['model_id'] == win_id])
        c_str, c_ef, c_pr = get_val(sub[sub['model_id'] == cluster_id])
        
        if b_str == "---": continue
        
        # Comparisons
        signs = [np.sign(e) for e in [b_ef, w_ef, c_ef] if e is not None]
        sign_match = "Yes" if len(set(signs)) == 1 else "No"
        
        practs = [p for p in [b_pr, w_pr, c_pr] if p is not None]
        pract_match = "Yes" if len(set(practs)) == 1 else "No"
        
        table_rows.append(f"{tex_escape(term)} & {b_str} & {w_str} & {c_str} & {sign_match} & {pract_match} \\\\")

    content = "\n".join(table_rows)
    caption = "Robustness checks (RQ8-A)"
    notes = "Baseline: main specification with query-cluster robust SEs. Winsorized: 1\\%--99\\% winsorization on continuous predictors. Two-way clustered: SEs clustered by query and domain.\\newline\nSign Match compares coefficient signs to Baseline. Pract. Match indicates whether the practical-importance flag ($|\\beta^*|\\ge 0.03$) matches Baseline."
    
    latex = wrap_latex_table(content, caption, "tab:rq8a_robustness", "l l l l c c", tablenotes=notes)
    with open(f"{out_dir}/table_rq8a_robustness.tex", "w") as f:
        f.write(latex)

def generate_rq8_robustness_table_nosource(out_dir):
    path = "data/analysis/G/robustness_coeffs_r.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    schema_terms = list(cat_map.keys())
    
    df = df[df['term'].isin(schema_terms)].copy()
    
    baseline_id = 'RQ8_Robustness_Baseline_R'
    nosource_id = 'RQ8_Robustness_NoSourceDomain_R'
    
    if nosource_id not in df['model_id'].values:
        return
        
    header = "\\sbf{Predictor} & \\sbf{Baseline $\\beta^*$ (CI95)} & \\sbf{NoSourceDomain $\\beta^*$ (CI95)} & \\sbf{Sign Match} & \\sbf{Pract. Match} \\\\"
    
    def get_val(r_df):
        if r_df.empty: return "---", None, None
        r = r_df.iloc[0]
        return f"{r['effect_size']:.3f} [{r['ci_lower_95']:.3f}, {r['ci_upper_95']:.3f}]", r['effect_size'], r.get('practical_flag', False)

    table_rows = [header, "\\dmidrule"]
    
    for term in schema_terms:
        sub = df[df['term'] == term]
        if sub.empty: continue
        
        b_str, b_ef, b_pr = get_val(sub[sub['model_id'] == baseline_id])
        ns_str, ns_ef, ns_pr = get_val(sub[sub['model_id'] == nosource_id])
        
        if b_str == "---" or ns_str == "---": continue
        
        signs = [np.sign(e) for e in [b_ef, ns_ef] if e is not None]
        sign_match = "Yes" if len(set(signs)) == 1 else "No"
        
        practs = [p for p in [b_pr, ns_pr] if p is not None]
        pract_match = "Yes" if len(set(practs)) == 1 else "No"
        
        table_rows.append(f"{tex_escape(term)} & {b_str} & {ns_str} & {sign_match} & {pract_match} \\\\")

    content = "\n".join(table_rows)
    caption = "Sub-dataset Sensitivity (RQ8-B)"
    notes = "Baseline (Full Data) is compared against the subset excluding source domains (NoSourceDomain).\\newline\nSign Match compares coefficient signs to Baseline. Pract. Match indicates whether the practical-importance flag ($|\\beta^*|\\ge 0.03$) matches Baseline."
    
    latex = wrap_latex_table(content, caption, "tab:rq8b_sensitivity", "l l l c c", tablenotes=notes)
    with open(f"{out_dir}/table_rq8b_sensitivity.tex", "w") as f:
        f.write(latex)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/reports/tables")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Generating LaTeX Tables...")
    generate_dataset_distribution_table(args.out_dir)
    generate_feature_match_rate_table(args.out_dir)
    generate_html_size_stats_table(args.out_dir)
    generate_model_eligibility_table(args.out_dir)
    generate_rq1_engine_concentration_table(args.out_dir)
    generate_rq1_effect_sizes_table(args.out_dir)
    generate_cooks_d_table(args.out_dir)
    generate_confirmatory_table(args.out_dir)
    generate_supplementary_confirmatory_table(args.out_dir)
    generate_nested_model_table(args.out_dir)
    generate_ablation_predictive_table(args.out_dir)
    generate_ablation_stability_table(args.out_dir)
    generate_rq1_feature_trends_non_semantic_table(args.out_dir)
    generate_rq1_feature_trends_semantic_table(args.out_dir)
    generate_rq8_robustness_table_main(args.out_dir)
    generate_rq8_robustness_table_nosource(args.out_dir)
    
    print(f"Done. Tables saved to {args.out_dir}")

if __name__ == "__main__":
    main()
