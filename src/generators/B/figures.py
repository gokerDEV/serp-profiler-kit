import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Globals
ENGINE_ORDER = ["google", "brave", "mojeek"]
SUNSET_SUNRISE_PALETTE = ['#3b5b92', '#7a5195', '#ce5a8f', '#f0ad5f', '#e27850', '#cc4040', '#a42c33']
ENGINE_COLORS = {"google": "#4c8bf5", "brave": "#ff631c", "mojeek": "#7abb3b"}

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Global font sizes
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 10

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_forest(df, x_col, y_col, xerr_lower, xerr_upper, title, output_path, color='black'):
    plt.figure(figsize=(12, 4))
    y_pos = np.arange(len(df))
    plt.errorbar(
        x=df[x_col], 
        y=y_pos, 
        xerr=[xerr_lower, xerr_upper], 
        fmt='o', color=color, ecolor='gray', capsize=4
    )
    plt.yticks(y_pos, df[y_col])
    plt.gca().invert_yaxis()
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel("Effect Size / Standardized Coefficient")
    save_plot(output_path)

def draw_category_separators(ax, features_list, cat_map):
    boundaries = []
    if not features_list: return
    current_cat = cat_map.get(features_list[0])
    for i, f in enumerate(features_list):
        cat = cat_map.get(f)
        if cat != current_cat:
            boundaries.append(i - 0.5)
            current_cat = cat
    for b in boundaries:
        ax.axhline(b, linestyle=":", linewidth=1, color='lightgray')

def generate_rq1_plots(out_dir):
    rq1_json = "data/analysis/C/rq1_viz.json"
    if not os.path.exists(rq1_json): return
    with open(rq1_json, 'r') as f:
        data = json.load(f)
        
    if 'distribution' in data:
        df_dist = pd.DataFrame(data['distribution'])
        if 'subset' in df_dist.columns:
            df_dist = df_dist[df_dist['subset'] == 'Full']
            
        if 'sim_content' in df_dist.columns:
            # Sort engines
            avail_engines = [e for e in ENGINE_ORDER if e in df_dist['search_engine'].unique()]
            if avail_engines:
                df_dist['search_engine'] = pd.Categorical(df_dist['search_engine'], categories=avail_engines, ordered=True)
                df_dist = df_dist.sort_values('search_engine')
                
            plt.figure(figsize=(12, 3))
            sns.kdeplot(data=df_dist, x='sim_content', hue='search_engine', fill=True, alpha=0.3, common_norm=False, palette=ENGINE_COLORS)
            plt.title("Semantic Similarity Density Plot by Engine")
            plt.xlabel("Semantic Similarity (sim_content)")
            plt.ylabel("Density")
            save_plot(f"{out_dir}/fig_semantic_density.png")
            
    if 'trends' in data:
        df_trends = pd.DataFrame(data['trends'])
        
        target_subsets = ['Full', 'NoSource', 'Source']
        avail_sub = [s for s in target_subsets if 'subset' in df_trends.columns and s in df_trends['subset'].unique()]
        if not avail_sub:
            avail_sub = ['Full']
            if 'subset' not in df_trends.columns:
                df_trends['subset'] = 'Full'
                
        fig, axes = plt.subplots(1, len(avail_sub), figsize=(6 * len(avail_sub), 3), sharey=True)
        # Ensure iterable for single plot case
        if len(avail_sub) == 1:
            axes = [axes]
            
        bins = df_trends['rank_bin'].drop_duplicates().tolist()
        bin_idxs = np.arange(len(bins))
        dodge = 0.1
        markers = ["o", "s", "^", "D"]
        
        for ax, sub_name in zip(axes, avail_sub):
            sub_df = df_trends[df_trends['subset'] == sub_name]
            engines = [e for e in ENGINE_ORDER if e in sub_df['search_engine'].unique()]
            
            for i, eng in enumerate(engines):
                sub = sub_df[sub_df['search_engine'] == eng]
                if sub.empty: continue
                
                x_pos = bin_idxs + (i - len(engines)/2.0 + 0.5) * dodge
                y_mean = sub['mean'].values
                ci_margin = sub['mean'].values - sub['ci_lower'].values
                
                ax.errorbar(
                    x_pos, 
                    y_mean, 
                    yerr=ci_margin,
                    marker=markers[i % len(markers)],
                    capsize=4,
                    label=eng,
                    linestyle='-',
                    color=ENGINE_COLORS.get(eng, 'black')
                )
                
            ax.set_xticks(bin_idxs)
            ax.set_xticklabels(bins)
            ax.set_title(f"Semantic Similarity by Rank Tier ({sub_name})")
            ax.set_xlabel("Rank Group")
            if ax == axes[0]:
                ax.set_ylabel("Mean Similarity (95% CI)")
            if len(engines) > 0:
                ax.legend(title="Search Engine")
                
        plt.tight_layout()
        save_plot(f"{out_dir}/fig_semantic_trends.png")

def generate_all_feature_trends(out_dir):
    path = "data/analysis/C/rq1_feature_trends.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    
    ordered_features = [f for f in cat_map.keys() if f in df['feature'].unique()]
    subsets = ['Full', 'NoSource', 'Source']
    avail_sub = [s for s in subsets if s in df['subset'].unique()]
    
    if not ordered_features or not avail_sub: return
    
    n_rows = len(ordered_features)
    n_cols = len(avail_sub)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 16), sharex=True)
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = np.array([axes])
    elif n_cols == 1: axes = axes.reshape(-1, 1)
    
    bins = ["Top 1-3", "Rank 4-10", "Rank 11-20"]
    bin_idxs = np.arange(len(bins))
    dodge = 0.1
    markers = ["o", "s", "^", "D"]
    
    lines_for_legend = []
    labels_for_legend = []
    
    for row_idx, feature in enumerate(ordered_features):
        df_feat = df[df['feature'] == feature]
        
        for col_idx, sub_name in enumerate(avail_sub):
            ax = axes[row_idx, col_idx]
            sub_df = df_feat[df_feat['subset'] == sub_name]
            engines = [e for e in ENGINE_ORDER if e in sub_df['search_engine'].unique()]
            
            for i, eng in enumerate(engines):
                sub = sub_df[sub_df['search_engine'] == eng]
                if sub.empty: continue
                
                # Make sure bins are ordered identically
                sub_indexed = sub.set_index('rank_bin')
                present_bins = [b for b in bins if b in sub_indexed.index]
                if not present_bins: continue
                
                sub_indexed = sub_indexed.loc[present_bins].reset_index()
                
                x_pos = np.array([bins.index(b) for b in present_bins]) + (i - len(engines)/2.0 + 0.5) * dodge
                y_mean = sub_indexed['mean'].values
                ci_margin = sub_indexed['mean'].values - sub_indexed['ci_lower'].values
                
                line = ax.errorbar(
                    x_pos, 
                    y_mean, 
                    yerr=ci_margin,
                    marker=markers[i % len(markers)],
                    capsize=4,
                    linestyle='-',
                    color=ENGINE_COLORS.get(eng, 'black')
                )
                if row_idx == 0 and col_idx == 0:
                    lines_for_legend.append(line)
                    labels_for_legend.append(eng.capitalize())
            
            ax.set_xticks(bin_idxs)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(bins)
            else:
                ax.set_xticklabels([])
                
            if row_idx == 0:
                ax.set_title(sub_name)
                
            if col_idx == 0:
                ax.set_ylabel(feature)
                
    plt.tight_layout(rect=[0, 0.12, 1.2, 0.98], h_pad=1.5)
    fig.subplots_adjust(bottom=0.08)
    fig.legend(lines_for_legend, labels_for_legend, loc='center', bbox_to_anchor=(0.5, 0.0), ncol=len(labels_for_legend), title="", frameon=False)
    
    save_plot(f"{out_dir}/fig_all_feature_trends.png")

def generate_semantic_decay_vis(out_dir):
    try:
        df = pd.read_parquet("data/dataset.parquet")
    except Exception:
        return
        
    if 'sim_content' not in df.columns or 'rank' not in df.columns or 'is_source_domain' not in df.columns:
        return
        
    df_plot = df[df['rank'] <= 20].copy()
    if df_plot.empty: return
    
    df_plot['Source Status'] = df_plot['is_source_domain'].map({True: 'Guardian', False: 'Non-Guardian'})
    
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_plot[df_plot['Source Status'] == 'Guardian'], x='rank', y='sim_content', 
                label='Guardian', scatter_kws={'alpha':0.1, 'color':'blue'}, line_kws={'color':'blue'})
    sns.regplot(data=df_plot[df_plot['Source Status'] == 'Non-Guardian'], x='rank', y='sim_content', 
                label='Non-Guardian', scatter_kws={'alpha':0.1, 'color':'orange'}, line_kws={'color':'orange'})
    
    plt.title("Semantic Decay Vis: sim_content vs Rank (Top 20)")
    plt.xlabel("Rank (1-20)")
    plt.ylabel("Semantic Similarity (sim_content)")
    plt.legend(title="Source Status")
    save_plot(f"{out_dir}/fig_semantic_decay_vis.png")

def generate_nosource_paired_forest(out_dir):
    path = "data/analysis/D/confirmatory_table_comparison.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    if 'Predictor' not in df.columns or 'Estimate (Full)' not in df.columns or 'Estimate (NoSource)' not in df.columns: return
    
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    ordered_terms = list(cat_map.keys())
    
    current_terms = [t for t in ordered_terms if t in df['Predictor'].values]
    df = df[df['Predictor'].isin(current_terms)].copy()
    df['Predictor'] = pd.Categorical(df['Predictor'], categories=ordered_terms, ordered=True)
    df = df.sort_values('Predictor')
    
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(12, max(5, len(current_terms)*0.4)))
    
    y_base = np.arange(len(current_terms), dtype=float)
    offsets = {'Full': -0.15, 'NoSource': 0.15}
    colors = {'Full': 'tab:blue', 'NoSource': 'tab:orange'}
    
    ax.axvline(0.0, linestyle="--", linewidth=1, color='gray')
    
    for subset in ['Full', 'NoSource']:
        col = f'Estimate ({subset})'
        if col not in df.columns: continue
        
        y = y_base + offsets[subset]
        x = df[col].values
        ax.plot(x, y, 'o', color=colors[subset], label=subset, markersize=8)
        
    draw_category_separators(ax, current_terms, cat_map)
    ax.set_yticks(y_base)
    ax.set_yticklabels(current_terms)
    ax.invert_yaxis()
    ax.set_xlabel("Estimate (Standardized Coefficient)")
    ax.set_title("Paired Forest Plot: Full vs NoSource Coefficient Estimates")
    ax.legend(title="Subset", loc="upper right")
    save_plot(f"{out_dir}/fig_paired_forest_nosource.png")


def generate_permutation_importance(out_dir):
    path = "data/analysis/H/rank_stability_importance.csv"
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    if 'feature' not in df.columns: return
    if 'importance_mean' in df.columns:
        df['importance'] = df['importance_mean']
        
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    ordered_features = list(cat_map.keys())
    
    # Filter features that exist in both
    current_features = [f for f in ordered_features if f in df['feature'].values]
    df_top = df[df['feature'].isin(current_features)].copy()
    
    # Exclude 'Full'
    df_top = df_top[~df_top['subset'].astype(str).str.contains('Full', case=False)].copy()
    
    # Sort original dataframe by schema order
    df_top['feature'] = pd.Categorical(df_top['feature'], categories=current_features, ordered=True)
    
    # Sort engines properly
    avail_engines = [e for e in ENGINE_ORDER if e in df_top['subset'].unique()]
    if avail_engines:
        df_top['subset'] = pd.Categorical(df_top['subset'], categories=avail_engines, ordered=True)
        
    df_top = df_top.sort_values(['feature', 'subset'])
    
    plt.figure(figsize=(12, 4))
    sns.barplot(data=df_top, x='importance', y='feature', hue='subset', palette=ENGINE_COLORS, order=current_features)
    draw_category_separators(plt.gca(), current_features, cat_map)
    plt.title("Permutation Importance Ranking by Engine (LTR)")
    plt.xlabel("Mean Importance (Decrease in NDCG)")
    plt.ylabel("Feature")
    plt.legend(title="Search Engine", loc='upper right')
    save_plot(f"{out_dir}/fig_feature_importance.png")

def generate_replication_grid(out_dir: str) -> None:
    candidates = [
        "/mnt/data/replication_grid.csv",
        "data/analysis/E/replication_grid.csv",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"File {path} is empty. Skipping grid generation.")
        return

    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    schema_terms = list(cat_map.keys())

#     if not current_terms: return
    
#     df_plot = df[df['term'].isin(current_terms)].copy()
    
#     # Sort terms by schema
#     df_plot['term'] = pd.Categorical(df_plot['term'], categories=current_terms, ordered=True)
#     df_plot = df_plot.sort_values('term')
    
#     # Heatmap of specific statistical agreement metrics
#     metrics = ['all_sign_agreement', 'significant_agreement', 'ci_overlap']
#     avail_metrics = [m for m in metrics if m in df_plot.columns]
    
#     if not avail_metrics: return
    
#     pivot = df_plot.set_index('term')[avail_metrics].astype(float)
    
#     plt.figure(figsize=(10, max(5, len(pivot)*0.35)))
#     sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0.5, vmin=0, vmax=1)
#     plt.title("Replication Grid: Coefficient Consistency Across Engines")
#     save_plot(f"{out_dir}/fig_replication_grid.png")

def generate_engine_stratified_forest(out_dir: str) -> None:
    candidates = [
        "/mnt/data/heterogeneity_coeffs_r.csv",
        "data/analysis/E/heterogeneity_coeffs_r.csv",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return

    df = pd.read_csv(path)

    required = {"term", "effect_size", "ci_lower_95", "ci_upper_95", "subset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_eng = df[df["subset"].astype(str).str.startswith("Engine_")].copy()
    if df_eng.empty:
        return

    df_eng["engine"] = df_eng["subset"].astype(str).str.replace("Engine_", "", regex=False)

    # Use global ENGINE_ORDER
    df_eng = df_eng[df_eng["engine"].isin(ENGINE_ORDER)].copy()

    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    schema_terms = list(cat_map.keys())

    current_terms = [t for t in schema_terms if t in set(df_eng["term"].astype(str))]
    if not current_terms:
        return

    df_eng = df_eng[df_eng["term"].isin(current_terms)].copy()
    df_eng["term"] = pd.Categorical(df_eng["term"], categories=current_terms, ordered=True)
    df_eng["engine"] = pd.Categorical(df_eng["engine"], categories=ENGINE_ORDER, ordered=True)
    df_eng = df_eng.sort_values(["term", "engine"])

    terms = current_terms
    y_base = np.arange(len(terms), dtype=float)

    offsets = {
        "brave": -0.20,
        "google": 0.00,
        "mojeek": 0.20,
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.axvline(0.0, linestyle="--", linewidth=1)

    for eng in ENGINE_ORDER:
        sub = df_eng[df_eng["engine"] == eng].copy()
        if sub.empty:
            continue
            
        color = ENGINE_COLORS.get(eng, 'black')

        term_to_row = {t: i for i, t in enumerate(terms)}
        y = np.array([term_to_row[t] for t in sub["term"].astype(str)], dtype=float) + offsets[eng]

        x = sub["effect_size"].astype(float).to_numpy()
        lo = sub["ci_lower_95"].astype(float).to_numpy()
        hi = sub["ci_upper_95"].astype(float).to_numpy()

        xerr = np.vstack([x - lo, hi - x])

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            linewidth=1,
            label=eng,
            color=color,
        )

    draw_category_separators(ax, terms, cat_map)

    ax.set_yticks(y_base)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized Coefficient (β*)")
    ax.set_title("RQ6 Engine-stratified Coefficients with 95% CI")
    ax.legend(title="Search Engine", loc="upper right")

    save_plot(f"{out_dir}/fig_rq6_engine_stratified_forest.png")

def generate_difficulty_bands_heatmap(out_dir: str) -> None:
    path = "data/analysis/F/difficulty_coeffs_r.csv"
    if not os.path.exists(path):
        print("No difficulty_coeffs_r.csv found")
        return

    df_full = pd.read_csv(path)
    
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    band_order = ["Low", "Medium", "High"]
    schema_terms = list(cat_map.keys())

    subsets = df_full['subset'].unique() if 'subset' in df_full.columns else ['Full']
    
    pivots = {}
    for subset in subsets:
        strat = df_full[df_full['subset'] == subset].copy() if 'subset' in df_full.columns else df_full.copy()
        strat = strat[strat["term"].astype(str).str.contains("difficulty_band::", case=False, na=False)].copy()
        if strat.empty: continue

        strat[["band", "feature"]] = strat["term"].astype(str).str.extract(r"difficulty_band::(.*?):(.*)")
        strat["band"] = strat["band"].astype(str).str.replace("_Difficulty", "", regex=False)
        strat = strat[strat["feature"].isin(set(schema_terms))].copy()
        if strat.empty: continue

        strat["band"] = pd.Categorical(strat["band"], categories=band_order, ordered=True)
        ordered_features = [t for t in schema_terms if t in set(strat["feature"].astype(str))]
        strat["feature"] = pd.Categorical(strat["feature"], categories=ordered_features, ordered=True)
        strat = strat.sort_values(["feature", "band"])

        pivot = strat.pivot(index="feature", columns="band", values="effect_size").reindex(index=ordered_features, columns=band_order)
        pivots[subset] = (pivot, strat, ordered_features)

    def plot_heatmap(pivot, strat, ordered_features, title, filename):
        has_fdr = "fdr_significant" in strat.columns
        has_practical = "practical_flag" in strat.columns
        has_ci = "ci_lower_95" in strat.columns and "ci_upper_95" in strat.columns
        annot = pd.DataFrame("", index=ordered_features, columns=band_order)

        for _, row in strat.iterrows():
            feat = str(row["feature"])
            band = str(row["band"])
            beta = float(row["effect_size"])

            sig = False
            if has_fdr: sig = bool(row["fdr_significant"])
            elif has_ci: sig = (float(row["ci_lower_95"]) > 0.0) or (float(row["ci_upper_95"]) < 0.0)

            practical = bool(row["practical_flag"]) if has_practical else False

            tag = "*" if sig else ""
            tag += "!" if practical else ""
            annot.loc[feat, band] = f"{beta:.3f}{tag}"

        n_rows = len(ordered_features)
        fig, ax = plt.subplots(figsize=(12, 5))
        mat = pivot.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")

        ax.set_title(title)
        ax.set_xlabel("Difficulty Band")
        ax.set_ylabel("Predictor")
        ax.set_xticks(range(len(band_order)))
        ax.set_xticklabels(band_order)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ordered_features)
        ax.grid(False)

        for i in range(n_rows):
            for j in range(len(band_order)):
                txt = annot.iloc[i, j]
                val = float(pivot.iloc[i, j]) if pd.notna(pivot.iloc[i, j]) else 0.0
                if isinstance(txt, str) and txt != "":
                    color = "white" if abs(val) >= 0.05 else "black"
                    ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Effect size (β)")
        fig.text(0.5, -0.05, "*: FDR-significant (or CI excludes 0)   !: practical_flag", ha="center", va="top", fontsize=9)
        save_plot(filename)

    for subset, (pivot, strat, ordered_features) in pivots.items():
        if subset not in ['Full', 'NoSource', 'Source']: continue
        title = f"RQ7 Difficulty Bands: Predictor × Band Effects ({subset})"
        filename = f"{out_dir}/fig_difficulty_bands_heatmap_{subset}.png"
        plot_heatmap(pivot, strat, ordered_features, title, filename)
        
    if 'Full' in pivots and 'NoSource' in pivots:
        p_full, s_full, o_feat = pivots['Full']
        p_ns, _, _ = pivots['NoSource']
        
        delta_pivot = p_ns - p_full
        
        n_rows = len(o_feat)
        fig, ax = plt.subplots(figsize=(12, 5))
        mat = delta_pivot.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap="PuOr")

        ax.set_title("RQ7 Difficulty Bands: Delta (NoSource - Full)")
        ax.set_xlabel("Difficulty Band")
        ax.set_ylabel("Predictor")
        ax.set_xticks(range(len(band_order)))
        ax.set_xticklabels(band_order)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(o_feat)
        ax.grid(False)

        for i in range(n_rows):
            for j in range(len(band_order)):
                val = float(delta_pivot.iloc[i, j]) if pd.notna(delta_pivot.iloc[i, j]) else 0.0
                txt = f"{val:+.3f}"
                color = "white" if abs(val) >= (vmax/2) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Δ Effect size")
        save_plot(f"{out_dir}/fig_difficulty_bands_heatmap_Delta.png")

# def generate_difficulty_bands(out_dir):
#     path = "data/analysis/F/difficulty_coeffs_r.csv"
#     if not os.path.exists(path): return
#     df = pd.read_csv(path)
    
#     # Filter for interaction terms
#     strat = df[df['term'].str.contains('difficulty_band::', case=False)].copy()
#     if strat.empty: return
    
#     # Parse Band and exact Feature
#     strat[['band', 'feature']] = strat['term'].str.extract(r'difficulty_band::(.*?):(.*)')
#     strat['band'] = strat['band'].str.replace('_Difficulty', '')
    
#     from src.helpers.data_loader import get_feature_categories
#     cat_map = get_feature_categories()
    
#     # Subset to the valid features from schema
#     strat = strat[strat['feature'].isin(cat_map.keys())]
#     if strat.empty: return
    
#     # For plot size, picking specific feature(s). E.g. averaging over all features or picking content.
#     # The user asked to show difficulty bands.
#     # Let's aggregate by Category and Band instead.
#     strat['category'] = strat['feature'].map(cat_map).fillna('Other').str.capitalize()
    
#     # Sort bands correctly
#     band_order = ['Low', 'Medium', 'High']
#     strat['band'] = pd.Categorical(strat['band'], categories=band_order, ordered=True)
    
#     # Sort categories by schema
#     schema_cats = list(dict.fromkeys([c.capitalize() for c in cat_map.values()]))
#     current_cats = [c for c in schema_cats if c in strat['category'].values]
#     strat['category'] = pd.Categorical(strat['category'], categories=current_cats, ordered=True)
    
#     strat = strat.sort_values(['category', 'band'])
    
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=strat, x="category", y="effect_size", hue="band", palette="colorblind")
#     plt.title("Query Difficulty Stability across Metric Categories")
#     plt.ylabel("Standardized Effect Size (Average)")
#     plt.xlabel("Category")
#     plt.legend(title="Difficulty Band", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.axhline(0, color='gray', linestyle='--')
#     save_plot(f"{out_dir}/fig_difficulty_bands.png")

def generate_confirmatory_forest(out_dir: str) -> None:
    # Prefer local sandbox CSV if present; otherwise fall back to repo path.
    conf_candidates = [
        "/mnt/data/confirmatory_coeffs.csv",
        "data/analysis/D/confirmatory_coeffs.csv",
    ]
    conf_path = next((p for p in conf_candidates if os.path.exists(p)), None)
    if conf_path is None:
        return

    df = pd.read_csv(conf_path)

    # --- Required columns sanity ---
    required = {"model_id", "term", "effect_size", "ci_lower_95", "ci_upper_95"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in confirmatory CSV: {sorted(missing)}")

    # Use CSV's CI columns explicitly
    df["ci_lower"] = df["ci_lower_95"]
    df["ci_upper"] = df["ci_upper_95"]

    # Drop engine terms & intercepts using explicit flags when available
    if "is_engine_term" in df.columns:
        df = df[df["is_engine_term"] != True]
    else:
        df = df[~df["term"].astype(str).str.contains("search_engine", na=False)]

    if "is_intercept" in df.columns:
        df = df[df["is_intercept"] != True]
    else:
        df = df[~df["term"].isin(["(Intercept)", "Intercept"])]

    # Target models (your original selection)
    target_models = [
        "RQ2_Semantics_Logit_Top20_Full",
        "RQ3_Readability_Logit_Top20_Full",
        "RQ4_Performance_Logit_Top20_Full",
        "RQ5_Accessibility_Logit_Top20_Full",
    ]

    df_target = df[df["model_id"].isin(target_models)].copy()
    if df_target.empty:
        return

    # Ensure deterministic ordering: terms kept from where they are first introduced
    df_target["model_id"] = pd.Categorical(df_target["model_id"], categories=target_models, ordered=True)
    df_target = df_target.sort_values(["model_id", "term"])

    combined_df = df_target.drop_duplicates(subset=["term"], keep="first").copy()

    # Order terms by your feature schema
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()  # expected to be ordered by schema
    ordered_terms = list(cat_map.keys())

    current_terms = [t for t in ordered_terms if t in set(combined_df["term"].astype(str))]
    combined_df = combined_df[combined_df["term"].isin(current_terms)].copy()

    combined_df["term"] = pd.Categorical(combined_df["term"], categories=current_terms, ordered=True)
    combined_df = combined_df.sort_values("term")

    # --------- FIGURE 1: Logit coefficients (β) ----------
    err_l_beta = combined_df["effect_size"] - combined_df["ci_lower"]
    err_u_beta = combined_df["ci_upper"] - combined_df["effect_size"]

    plot_forest(
        combined_df,
        x_col="effect_size",
        y_col="term",
        xerr_lower=err_l_beta,
        xerr_upper=err_u_beta,
        title="Confirmatory Logit Coefficients (β, log-odds)",
        output_path=f"{out_dir}/fig_confirmatory_beta.png",
    )

    # --------- FIGURE 2: Odds Ratios (OR) ----------
    # OR and CI must be strictly positive; exp() ensures that.
    combined_or = combined_df.copy()
    combined_or["or"] = np.exp(combined_or["effect_size"])
    combined_or["or_ci_lower"] = np.exp(combined_or["ci_lower"])
    combined_or["or_ci_upper"] = np.exp(combined_or["ci_upper"])

    err_l_or = combined_or["or"] - combined_or["or_ci_lower"]
    err_u_or = combined_or["or_ci_upper"] - combined_or["or"]

    # If your plot_forest supports a log-scale argument, it should be enabled here.
    # Otherwise, it will still work, but values may be visually compressed.
    plot_forest(
        combined_or,
        x_col="or",
        y_col="term",
        xerr_lower=err_l_or,
        xerr_upper=err_u_or,
        title="Confirmatory Odds Ratios (OR = exp(β))",
        output_path=f"{out_dir}/fig_confirmatory_or.png",
        # optional if supported by your helper:
        # xscale="log"
    )

def generate_engine_heterogeneity(out_dir):
    het_path = "data/analysis/E/heterogeneity_coeffs_r.csv"
    if not os.path.exists(het_path): return
    df = pd.read_csv(het_path)
    strat = df[df['model_id'].str.contains('Stratified', case=False)]
    if strat.empty: return
    
    feat_data = strat.copy()
    feat_data = feat_data[feat_data['subset'].astype(str).str.endswith('_Full', na=False)].copy()
    feat_data['search_engine'] = feat_data['subset'].astype(str).str.replace('Engine_', '', case=False).str.replace('_Full', '', case=False)
    
    # Use schema features for ordering
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    
    ordered_terms = list(cat_map.keys())
    current_terms = [t for t in ordered_terms if t in feat_data['term'].values]
    
    # Exclude rows where term is not in the schema mapping correctly
    feat_data = feat_data[feat_data['term'].isin(current_terms)].copy()
    
    avail_engines = [e for e in ENGINE_ORDER if e in feat_data['search_engine'].unique()]
    if avail_engines:
        feat_data['search_engine'] = pd.Categorical(feat_data['search_engine'], categories=avail_engines, ordered=True)
    feat_data = feat_data.sort_values(['term', 'search_engine'])
    
    plt.figure(figsize=(12, 4))
    
    sns.barplot(data=feat_data, x="effect_size", y="term", hue="search_engine", palette=ENGINE_COLORS, order=current_terms)
    draw_category_separators(plt.gca(), current_terms, cat_map)
    
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Engine-specific Standardized Effects by Feature")
    plt.xlabel("Standardized Effect Size (Coefficient)")
    plt.ylabel("Feature (Term)")
    plt.legend(title="Search Engine", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    save_plot(f"{out_dir}/fig_heterogeneity_all_features.png")

def generate_ablation_ndcg(out_dir):
    abl_path = "data/analysis/H/ablation_predictive.csv"
    if not os.path.exists(abl_path): return
    df = pd.read_csv(abl_path)
    
    # We want to plot NDCG@10 change under feature-set ablations across engines.
    # Exclude the "Full Model" baseline from bars as it has 0 change
    sub = df[~df['set_name'].astype(str).str.contains('Full', case=False, na=False)].copy()
    if sub.empty: return
    
    # Remove the '- ' prefix from ablations
    sub['set_name'] = sub['set_name'].astype(str).str.replace(r'^-?\s*', '', regex=True)
    
    plt.figure(figsize=(12, 3))
    if 'effect_size' not in sub.columns:
        if 'ndcg_diff' in sub.columns:
            sub['effect_size'] = sub['ndcg_diff']
        else:
            return
            
    # Apply schema ordering
    from src.helpers.data_loader import get_feature_categories
    cat_map = get_feature_categories()
    schema_cats = list(dict.fromkeys([c.capitalize() for c in cat_map.values()]))
    ordered_sets = schema_cats
    existing_sets = [s for s in ordered_sets if s in sub['set_name'].values]
    other_sets = [s for s in sub['set_name'].unique() if s not in ordered_sets]
    final_order = existing_sets + other_sets
    
    avail_engines = [e for e in ENGINE_ORDER if e in sub['subset'].unique()]
    if avail_engines:
        sub = sub[sub['subset'].isin(avail_engines)].copy()
        sub['subset'] = pd.Categorical(sub['subset'], categories=avail_engines, ordered=True)
    
    sub['set_name'] = pd.Categorical(sub['set_name'], categories=final_order, ordered=True)
    sub = sub.sort_values(['set_name', 'subset'])
            
    sns.barplot(data=sub, x="effect_size", y="set_name", hue="subset", palette=ENGINE_COLORS, order=final_order)
    plt.axvline(0, color='black', linewidth=1)
    plt.title("NDCG@10 change under feature-set ablations across engines")
    plt.xlabel("Change in NDCG@10 (Negative = Performance Loss)")
    plt.ylabel("Ablated Feature Set")
    plt.legend(title="Search Engine", bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(f"{out_dir}/fig_ablation_ndcg_change.png")

def generate_concentration_violin(out_dir):
    rq1_path = "data/analysis/C/rq1_concentration.parquet"
    if not os.path.exists(rq1_path): return
    
    df = pd.read_parquet(rq1_path)
    ok_df = df[df['status'] == 'ok'] if 'status' in df.columns else df
    
    if 'domain_entropy_norm' not in ok_df.columns or 'domain_gini' not in ok_df.columns:
        return
    
    # Define a consistent color palette for search engines
    avail_engines = [e for e in ENGINE_ORDER if e in ok_df['search_engine'].unique()]
    if avail_engines:
        ok_df['search_engine'] = pd.Categorical(ok_df['search_engine'], categories=avail_engines, ordered=True)
        ok_df = ok_df.sort_values('search_engine')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 2.5))

    
    if "domain_entropy_norm" in ok_df.columns:
        sns.violinplot(data=ok_df, x="domain_entropy_norm", y="search_engine",
                   hue="search_engine", palette=ENGINE_COLORS, inner="quartile",
                   ax=ax1, bw_adjust=0.6, density_norm="width", width=1.2, legend=False)
    ax1.set_title("Normalized Entropy (Diversity)")
    ax1.set_xlim(0, 1) # Changed from ylim to xlim for horizontal violin plot
    ax1.set_xlabel("Normalized Entropy") # Added label
    ax1.set_ylabel("Search Engine") # Added label
    
    # Right: Gini Coefficient
    if "domain_gini" in ok_df.columns:
        sns.violinplot(data=ok_df, x="domain_gini", y="search_engine",
                   hue="search_engine", palette=ENGINE_COLORS, inner="quartile",
                   ax=ax2, bw_adjust=0.6, density_norm="width", width=1.2, legend=False)
    ax2.set_title("Gini Coefficient (Inequality)")
    ax2.set_xlim(0, 1) # Changed from ylim to xlim for horizontal violin plot
    ax2.set_xlabel("Gini Coefficient") # Added label
    ax2.set_ylabel("") # Removed redundant label

    ax1.set_ylim(-1.0, 3.0)
    ax2.set_ylim(-1.0, 3.0)

    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.5) 
    save_plot(f"{out_dir}/fig_rank_concentration.png")


def generate_difficulty_bands_forest_smallmultiples(out_dir: str) -> None:
    candidates = [
        "/mnt/data/difficulty_coeffs_r.csv",
        "data/analysis/F/difficulty_coeffs_r.csv",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return

    df = pd.read_csv(path)

    subsets_available = []
    if "subset" in df.columns:
        subsets_available = [str(s) for s in df["subset"].unique() if pd.notna(s)]
    else:
        df["subset"] = "Full"
        subsets_available = ["Full"]
        
    for current_subset in subsets_available:
        df_sub = df[df["subset"] == current_subset].copy()
        
        strat = df_sub[df_sub["term"].astype(str).str.contains("difficulty_band::", case=False, na=False)].copy()
        if strat.empty:
            continue

        strat[["band", "feature"]] = strat["term"].astype(str).str.extract(r"difficulty_band::(.*?):(.*)")
        strat["band"] = strat["band"].astype(str).str.replace("_Difficulty", "", regex=False)

        from src.helpers.data_loader import get_feature_categories
        cat_map = get_feature_categories()

        strat = strat[strat["feature"].isin(set(cat_map.keys()))].copy()
        if strat.empty:
            continue

        # Category assignment
        strat["category"] = strat["feature"].map(cat_map).astype(str).str.lower()

        band_order = ["Low", "Medium", "High"]
        strat["band"] = pd.Categorical(strat["band"], categories=band_order, ordered=True)

        # Define category panel order (match your RQs narrative)
        panel_order = ["semantic", "performance", "accessibility", "readability"]
        panels = [c for c in panel_order if c in set(strat["category"])]

        # Term order by schema within each category
        schema_terms = list(cat_map.keys())

        # Build ordered features and category boundaries
        ordered_features = []
        category_boundaries = []
        current_y_idx = 0
        feature_to_y = {}
        
        for cat in panels:
            sub = strat[strat["category"] == cat]
            feats = [t for t in schema_terms if t in set(sub["feature"].astype(str))]
            for f in feats:
                if f not in feature_to_y:
                    feature_to_y[f] = current_y_idx
                    ordered_features.append(f)
                    current_y_idx += 1
            category_boundaries.append(current_y_idx - 0.5)

        if category_boundaries:
            category_boundaries.pop() # remove the last line

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.axvline(0.0, linestyle="--", linewidth=1, color='gray')
        
        for b in category_boundaries:
            ax.axhline(b, linestyle=":", linewidth=1, color='lightgray')

        offsets = {"Low": -0.22, "Medium": 0.0, "High": 0.22}
        band_colors = {"Low": SUNSET_SUNRISE_PALETTE[0], "Medium": SUNSET_SUNRISE_PALETTE[1], "High": SUNSET_SUNRISE_PALETTE[2]}

        for b in band_order:
            sb = strat[strat["band"].astype(str) == b].copy()
            if sb.empty:
                continue
            sb = sb[sb["feature"].isin(ordered_features)]
            
            y = np.array([feature_to_y[str(f)] for f in sb["feature"].astype(str)], dtype=float) + offsets[b]
            
            x = sb["effect_size"].astype(float).to_numpy()
            lo = sb["ci_lower_95"].astype(float).to_numpy()
            hi = sb["ci_upper_95"].astype(float).to_numpy()
            
            xerr = np.vstack([x - lo, hi - x])

            ax.errorbar(
                x,
                y,
                xerr=xerr,
                fmt="o",
                capsize=3,
                linewidth=1,
                label=b,
                color=band_colors.get(b, 'black')
            )

        ax.set_yticks(range(len(ordered_features)))
        ax.set_yticklabels(ordered_features)
        ax.invert_yaxis()

        ax.set_xlabel("Standardized Coefficient (β*)")
        ax.set_title(f"RQ7 Difficulty Bands: Engine-agnostic Interaction Effects ({current_subset})")
        ax.legend(title="Difficulty Band", loc="lower right")

        save_plot(f"{out_dir}/fig_difficulty_bands_smallmultiples_{current_subset}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/reports/figures")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    print("Generating Figures for Analysis Plan...")
    
    generate_rq1_plots(args.out_dir)
    generate_concentration_violin(args.out_dir)
    generate_all_feature_trends(args.out_dir)
    generate_permutation_importance(args.out_dir)
    generate_replication_grid(args.out_dir)
    generate_engine_stratified_forest(args.out_dir)
    # generate_difficulty_bands(args.out_dir)
    generate_difficulty_bands_forest_smallmultiples(args.out_dir)
    generate_difficulty_bands_heatmap(args.out_dir)
    generate_confirmatory_forest(args.out_dir)
    generate_engine_heterogeneity(args.out_dir)
    generate_ablation_ndcg(args.out_dir)
    
    print("Done. Figures saved.")

if __name__ == "__main__":
    main()
