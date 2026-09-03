import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.helpers.data_loader import get_feature_categories, get_feature_multipliers


ENGINE_ORDER = ["Google", "Brave", "Mojeek"]
ENGINE_MAP = {"google": "Google", "brave": "Brave", "mojeek": "Mojeek"}
ENGINE_COLORS = {"Google": "#4c8bf5", "Brave": "#ff631c", "Mojeek": "#7abb3b"}
SUNSET_SUNRISE_PALETTE = ["#3b5b92", "#7a5195", "#ce5a8f", "#f0ad5f", "#e27850", "#cc4040", "#a42c33"]
FIG_WIDTH = 8

TEXT_MAP = {
    "titles": {
        "semantic_density": "Arama motoruna gore anlamsal benzerlik yogunlugu",
        "all_feature_trends": "Tum ozelliklerde siralama egilimleri",
        "feature_importance": "Arama motoruna gore ozellik onem siralamasi",
        "heterogeneity_all_features": "Arama motoruna gore standartlastirilmis etki buyuklukleri",
        "replication_grid_full": "Motorlar arasi katsayi tutarliligi izgara ozeti (Full)",
        "engine_stratified_forest_full": "Motor bazli katsayilar ve %95 guven araliklari (Full)",
        "dispersion_heatmap_full": "Dispersion bantlarina gore ozellik etkileri (Full)",
        "dispersion_smallmultiples_full": "Dispersion bantlarina gore etkiler (Full)",
        "ablation_ndcg_change": "Ozellik grubu cikariminda NDCG@10 degisimi",
    },
    "axes": {
        "semantic_similarity": "Anlamsal benzerlik (sim_content)",
        "density": "Yogunluk",
        "rank_group": "Siralama grubu",
        "mean_similarity_ci": "Ortalama benzerlik (%95 GA)",
        "importance": "Ortalama onem (NDCG azalis)",
        "feature": "Ozellik",
        "feature_plural": "Ozellikler",
        "standardized_coefficient": "Standartlastirilmis katsayi",
        "dispersion_band": "Dispersion bandi",
        "predictor": "Yordayici",
        "effect_size_beta": "Etki buyuklugu (β)",
        "ndcg_change": "NDCG@10 degisimi",
        "ablated_feature_set": "Cikarilan ozellik grubu",
        "coefficient": "Katsayi",
    },
    "legend": {
        "search_engine": "Arama Motoru",
        "dispersion_band": "Dispersion Bandi",
    },
    "bands": {"Low": "Dusuk", "Medium": "Orta", "High": "Yuksek"},
    "subsets": {"Full": "Full", "NoSource": "NoSource", "Source": "Source"},
    "categories": {
        "performance": "Performans",
        "accessibility": "Erisilebilirlik",
        "readability": "Okunabilirlik",
        "semantic": "Anlamsal",
    },
    "features": {
        "lcp_ms": "LCP",
        "ttfb_ms": "TTFB",
        "cls": "CLS",
        "axe_score": "AXE Score",
        "contrast_score": "Contrast Score",
        "flesch_reading_ease": "FRE",
        "flesch_kincaid_grade": "FKG",
        "sim_title": "Sim. Title",
        "sim_description": "Sim. Desc.",
        "sim_h1": "Sim H1",
        "sim_content": "Sim. Content",
    },
}

FILENAME_MAP = {
    "fig_rq9_semantic_density.png": "fig_rq9_semantic_density.png",
    "fig_rq9_trends.png": "fig_rq9_trends.png",
    "fig_rq11_feature_importance.png": "fig_rq11_feature_importance.png",
    "fig_rq12_heterogeneity_coeffs.png": "fig_rq12_heterogeneity_coeffs.png",
    "fig_rq12_heterogeneity_coeffs_nosource.png": "fig_rq12_heterogeneity_coeffs_nosource.png",
    "fig_rq12_replication_grid.png": "fig_rq12_replication_grid.png",
    "fig_rq12_engine_stratified_effects.png": "fig_rq12_engine_stratified_effects.png",
    "fig_rq13_dispersion_heatmap.png": "fig_rq13_dispersion_heatmap.png",
    "fig_rq13_dispersion_smallmultiples.png": "fig_rq13_dispersion_smallmultiples.png",
    "fig_rq13_ablation_ndcg_change.png": "fig_rq13_ablation_ndcg_change.png",
}


sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
    }
)


def figsize(height: float) -> tuple[float, float]:
    return (FIG_WIDTH, height)


def save_plot(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / filename}")


def draw_category_separators(ax, features_list, cat_map):
    boundaries = []
    if not features_list:
        return
    current_cat = cat_map.get(features_list[0])
    for i, feature in enumerate(features_list):
        cat = cat_map.get(feature)
        if cat != current_cat:
            boundaries.append(i - 0.5)
            current_cat = cat
    for boundary in boundaries:
        ax.axhline(boundary, linestyle=":", linewidth=1, color="lightgray")


def generate_semantic_density(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/C/rq1_viz.json"
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "distribution" not in data:
        return

    df = pd.DataFrame(data["distribution"])
    if "subset" in df.columns:
        df = df[df["subset"] == "Full"]
    if "sim_content" not in df.columns:
        return
    if "search_engine" in df.columns:
        df["search_engine"] = df["search_engine"].replace(ENGINE_MAP)

    available_engines = [engine for engine in ENGINE_ORDER if engine in df["search_engine"].unique()]
    if available_engines:
        df["search_engine"] = pd.Categorical(df["search_engine"], categories=available_engines, ordered=True)
        df = df.sort_values("search_engine")

    plt.figure(figsize=figsize(3))
    sns.kdeplot(
        data=df,
        x="sim_content",
        hue="search_engine",
        fill=True,
        alpha=0.3,
        common_norm=False,
        palette=ENGINE_COLORS,
    )
    plt.title(TEXT_MAP["titles"]["semantic_density"])
    plt.xlabel(TEXT_MAP["axes"]["semantic_similarity"])
    plt.ylabel(TEXT_MAP["axes"]["density"])
    legend = plt.gca().get_legend()
    if legend is not None:
        legend.set_title(TEXT_MAP["legend"]["search_engine"])
    save_plot(output_dir, FILENAME_MAP["fig_rq9_semantic_density.png"])


def generate_all_feature_trends(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/C/rq1_feature_trends.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    if "search_engine" in df.columns:
        df["search_engine"] = df["search_engine"].replace(ENGINE_MAP)

    category_map = get_feature_categories()
    feature_multipliers = get_feature_multipliers()
    ordered_features = [feature for feature in category_map.keys() if feature in df["feature"].unique()]
    subsets = ["Full", "NoSource", "Source"]
    available_subsets = [subset for subset in subsets if subset in df["subset"].unique()]

    if not ordered_features or not available_subsets:
        return

    row_height = max(1.0, 10.0 / max(len(ordered_features), 1))
    fig, axes = plt.subplots(
        len(ordered_features),
        len(available_subsets),
        figsize=figsize(max(8, len(ordered_features) * row_height)),
        sharex=True,
    )
    if len(ordered_features) == 1 and len(available_subsets) == 1:
        axes = np.array([[axes]])
    elif len(ordered_features) == 1:
        axes = np.array([axes])
    elif len(available_subsets) == 1:
        axes = axes.reshape(-1, 1)

    bins = ["Top 1-3", "Rank 4-10", "Rank 11-20"]
    bin_labels_tr = ["Ilk 1-3", "Sira 4-10", "Sira 11-20"]
    markers = ["o", "s", "^", "D"]
    dodge = 0.1
    legend_handles = []
    legend_labels = []

    for row_idx, feature in enumerate(ordered_features):
        feature_df = df[df["feature"] == feature]
        for col_idx, subset in enumerate(available_subsets):
            ax = axes[row_idx, col_idx]
            subset_df = feature_df[feature_df["subset"] == subset]
            engines = [engine for engine in ENGINE_ORDER if engine in subset_df["search_engine"].unique()]

            for engine_idx, engine in enumerate(engines):
                engine_df = subset_df[subset_df["search_engine"] == engine]
                if engine_df.empty:
                    continue

                indexed = engine_df.set_index("rank_bin")
                present_bins = [band for band in bins if band in indexed.index]
                if not present_bins:
                    continue

                indexed = indexed.loc[present_bins].reset_index()
                multiplier = feature_multipliers.get(feature, 1.0)
                y_mean = indexed["mean"].values * multiplier
                y_ci_lower = indexed["ci_lower"].values * multiplier
                if multiplier < 0:
                    y_ci_lower = indexed["ci_upper"].values * multiplier

                x_pos = np.array([bins.index(band) for band in present_bins]) + (engine_idx - len(engines) / 2.0 + 0.5) * dodge
                error = y_mean - y_ci_lower

                line = ax.errorbar(
                    x_pos,
                    y_mean,
                    yerr=error,
                    marker=markers[engine_idx % len(markers)],
                    capsize=4,
                    linestyle="-",
                    color=ENGINE_COLORS.get(engine, "black"),
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(engine)

            ax.set_xticks(np.arange(len(bins)))
            ax.set_xticklabels(bin_labels_tr if row_idx == len(ordered_features) - 1 else [])
            if row_idx == 0:
                ax.set_title(TEXT_MAP["subsets"].get(subset, subset))
            if col_idx == 0:
                ax.set_ylabel(TEXT_MAP["features"].get(feature, feature))

    fig.suptitle(TEXT_MAP["titles"]["all_feature_trends"], y=1.01)
    plt.tight_layout(rect=[0, 0.12, 1.2, 0.98], h_pad=1.5)
    fig.subplots_adjust(bottom=0.08)
    fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.0), ncol=len(legend_labels), title="", frameon=False)
    
    save_plot(output_dir, FILENAME_MAP["fig_rq9_trends.png"])


def generate_feature_importance(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/H/rank_stability_importance.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return
    if "subset" in df.columns:
        df["subset"] = df["subset"].replace(ENGINE_MAP)
    if "importance_mean" in df.columns:
        df["importance"] = df["importance_mean"]

    category_map = get_feature_categories()
    ordered_features = list(category_map.keys())
    current_features = [feature for feature in ordered_features if feature in df["feature"].values]
    df = df[df["feature"].isin(current_features)].copy()
    df = df[~df["subset"].astype(str).str.contains("Full", case=False, na=False)].copy()
    df["feature"] = pd.Categorical(df["feature"], categories=current_features, ordered=True)

    available_engines = [engine for engine in ENGINE_ORDER if engine in df["subset"].unique()]
    if available_engines:
        df["subset"] = pd.Categorical(df["subset"], categories=available_engines, ordered=True)
    df = df.sort_values(["feature", "subset"])

    feature_labels = {key: TEXT_MAP["features"].get(key, key) for key in current_features}
    df["feature_label"] = df["feature"].astype(str).map(feature_labels)

    plt.figure(figsize=figsize(max(3.5, len(current_features) * 0.35)))
    sns.barplot(
        data=df,
        x="importance",
        y="feature_label",
        hue="subset",
        palette=ENGINE_COLORS,
        order=[feature_labels[feature] for feature in current_features],
    )
    draw_category_separators(
        plt.gca(),
        [feature_labels[feature] for feature in current_features],
        {feature_labels[key]: value for key, value in category_map.items() if key in feature_labels},
    )
    plt.title(TEXT_MAP["titles"]["feature_importance"])
    plt.xlabel(TEXT_MAP["axes"]["importance"])
    plt.ylabel(TEXT_MAP["axes"]["feature_plural"])
    plt.legend(title=TEXT_MAP["legend"]["search_engine"], loc="upper right")
    save_plot(output_dir, FILENAME_MAP["fig_rq11_feature_importance.png"])


def generate_heterogeneity_all_features(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/E/heterogeneity_coeffs_r.csv"
    if not path.exists():
        return

    df_main = pd.read_csv(path)
    stratified_main = df_main[df_main["model_id"].astype(str).str.contains("Stratified", case=False, na=False)].copy()
    if stratified_main.empty:
        return

    for subset_suffix in ["Full", "NoSource"]:
        stratified = stratified_main[stratified_main["subset"].astype(str).str.endswith(f"_{subset_suffix}", na=False)].copy()
        if stratified.empty:
            continue

        stratified["search_engine"] = (
            stratified["subset"].astype(str).str.replace("Engine_", "", regex=False).str.replace(f"_{subset_suffix}", "", regex=False)
        )
        stratified["search_engine"] = stratified["search_engine"].replace(ENGINE_MAP)

        category_map = get_feature_categories()
        ordered_terms = list(category_map.keys())
        current_terms = [term for term in ordered_terms if term in stratified["term"].values]
        stratified = stratified[stratified["term"].isin(current_terms)].copy()

        available_engines = [engine for engine in ENGINE_ORDER if engine in stratified["search_engine"].unique()]
        if available_engines:
            stratified["search_engine"] = pd.Categorical(stratified["search_engine"], categories=available_engines, ordered=True)
        stratified = stratified.sort_values(["term", "search_engine"])

        term_labels = {key: TEXT_MAP["features"].get(key, key) for key in current_terms}

        plt.figure(figsize=figsize(max(3.5, len(current_terms) * 0.35)))
        sns.barplot(
            data=stratified,
            x="effect_size",
            y="term",
            hue="search_engine",
            palette=ENGINE_COLORS,
            order=current_terms,
        )
        ax = plt.gca()
        ax.set_yticklabels([term_labels.get(term.get_text(), term.get_text()) for term in ax.get_yticklabels()])
        draw_category_separators(ax, current_terms, category_map)
        plt.axvline(0, color="gray", linestyle="--")
        
        title = TEXT_MAP["titles"]["heterogeneity_all_features"]
        if subset_suffix == "NoSource":
            title += " (NoSource)"
            
        plt.title(title)
        plt.xlabel(TEXT_MAP["axes"]["standardized_coefficient"])
        plt.ylabel(TEXT_MAP["axes"]["feature_plural"])
        plt.legend(title=TEXT_MAP["legend"]["search_engine"], loc="upper left", bbox_to_anchor=(1, 1))
        
        filename_key = "fig_rq12_heterogeneity_coeffs.png"
        if subset_suffix == "NoSource":
            filename_key = "fig_rq12_heterogeneity_coeffs_nosource.png"
            
        save_plot(output_dir, FILENAME_MAP.get(filename_key, filename_key))


def generate_replication_grid_full(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/E/replication_grid.csv"
    if not path.exists():
        return

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return

    category_map = get_feature_categories()
    schema_terms = list(category_map.keys())

    if "subset" not in df.columns:
        df["subset"] = "Full"

    df = df[df["subset"] == "Full"].copy()
    df = df[df["term"].isin(schema_terms)].copy()
    if df.empty:
        return

    df["term"] = pd.Categorical(df["term"], categories=schema_terms, ordered=True)
    df = df.sort_values("term")
    metrics = ["all_sign_agreement", "significant_agreement", "ci_overlap"]
    available_metrics = [metric for metric in metrics if metric in df.columns]
    if not available_metrics:
        return

    pivot = df.set_index("term")[available_metrics].astype(float)
    pivot.index = [TEXT_MAP["features"].get(term, term) for term in pivot.index]
    pivot.columns = ["Isaret", "Anlamlilik", "GA Ortusmesi"][: len(pivot.columns)]

    plt.figure(figsize=figsize(max(3.5, len(pivot.index) * 0.35)))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0.5, vmin=0, vmax=1)
    plt.title(TEXT_MAP["titles"]["replication_grid_full"])
    plt.xlabel("")
    plt.ylabel("")
    save_plot(output_dir, FILENAME_MAP["fig_rq12_replication_grid.png"])


def generate_engine_stratified_forest_full(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/E/heterogeneity_coeffs_r.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    required = {"term", "effect_size", "ci_lower_95", "ci_upper_95", "subset"}
    if not required.issubset(df.columns):
        return

    df = df[df["subset"].astype(str).str.contains(r"Engine_.*_Full", regex=True)].copy()
    if df.empty:
        return

    df["engine"] = df["subset"].astype(str).str.extract(r"Engine_([^_]+)")
    df["engine"] = df["engine"].replace(ENGINE_MAP)
    df = df[df["engine"].isin(ENGINE_ORDER)].copy()

    category_map = get_feature_categories()
    schema_terms = list(category_map.keys())
    df["category"] = df["term"].map(category_map).astype(str).str.lower()
    current_terms = [term for term in schema_terms if term in set(df["term"].astype(str))]
    if not current_terms:
        return

    panel_order = ["performance", "accessibility", "readability", "semantic"]
    panels = [category for category in panel_order if category in set(df["category"])]
    ordered_features = []
    category_boundaries = []
    current_y = 0
    feature_to_y = {}

    for category in panels:
        subset = df[df["category"] == category]
        features = [term for term in schema_terms if term in set(subset["term"].astype(str))]
        for feature in features:
            if feature not in feature_to_y:
                feature_to_y[feature] = current_y
                ordered_features.append(feature)
                current_y += 1
        category_boundaries.append(current_y - 0.5)

    if category_boundaries:
        category_boundaries.pop()

    df = df[df["term"].isin(ordered_features)].copy()
    df["term"] = pd.Categorical(df["term"], categories=ordered_features, ordered=True)
    df["engine"] = pd.Categorical(df["engine"], categories=ENGINE_ORDER, ordered=True)
    df = df.sort_values(["term", "engine"])

    offsets = {"Brave": -0.20, "Google": 0.00, "Mojeek": 0.20}
    fig, ax = plt.subplots(figsize=figsize(max(3.5, len(ordered_features) * 0.35)))
    ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")

    for boundary in category_boundaries:
        ax.axhline(boundary, linestyle=":", linewidth=1, color="lightgray")

    for engine in ENGINE_ORDER:
        subset = df[df["engine"] == engine].copy()
        if subset.empty:
            continue

        y = np.array([feature_to_y[str(term)] for term in subset["term"].astype(str)], dtype=float) + offsets[engine]
        x = subset["effect_size"].astype(float).to_numpy()
        lo = subset["ci_lower_95"].astype(float).to_numpy()
        hi = subset["ci_upper_95"].astype(float).to_numpy()
        xerr = np.vstack([x - lo, hi - x])

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            linewidth=1,
            label=engine,
            color=ENGINE_COLORS.get(engine, "black"),
        )

    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels([TEXT_MAP["features"].get(feature, feature) for feature in ordered_features])
    ax.invert_yaxis()
    ax.set_xlabel(TEXT_MAP["axes"]["standardized_coefficient"])
    ax.set_title(TEXT_MAP["titles"]["engine_stratified_forest_full"])
    ax.legend(title=TEXT_MAP["legend"]["search_engine"], loc="upper right")
    save_plot(output_dir, FILENAME_MAP["fig_rq12_engine_stratified_effects.png"])


def generate_dispersion_bands_heatmap_full(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/F/dispersion_coeffs_r.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    required = {"term", "effect_size", "ci_lower_95", "ci_upper_95"}
    if not required.issubset(df.columns):
        return

    category_map = get_feature_categories()
    schema_terms = list(category_map.keys())
    band_order = ["Low", "Medium", "High"]
    df = df[df["subset"] == "Full"].copy() if "subset" in df.columns else df.copy()
    df = df[df["term"].astype(str).str.contains("dispersion_band::", case=False, na=False)].copy()
    if df.empty:
        return

    df[["band", "feature"]] = df["term"].astype(str).str.extract(r"dispersion_band::(.*?):(.*)")
    df["band"] = df["band"].astype(str).str.replace("_Dispersion", "", regex=False)
    df = df[df["feature"].isin(set(schema_terms))].copy()
    if df.empty:
        return

    ordered_features = [term for term in schema_terms if term in set(df["feature"].astype(str))]
    df["band"] = pd.Categorical(df["band"], categories=band_order, ordered=True)
    df["feature"] = pd.Categorical(df["feature"], categories=ordered_features, ordered=True)
    df = df.sort_values(["feature", "band"])
    pivot = df.pivot(index="feature", columns="band", values="effect_size").reindex(index=ordered_features, columns=band_order)

    annotations = pd.DataFrame("", index=ordered_features, columns=band_order)
    has_fdr = "fdr_significant" in df.columns
    has_practical = "practical_flag" in df.columns

    for _, row in df.iterrows():
        feature = str(row["feature"])
        band = str(row["band"])
        beta = float(row["effect_size"])

        significant = bool(row["fdr_significant"]) if has_fdr else (float(row["ci_lower_95"]) > 0.0 or float(row["ci_upper_95"]) < 0.0)
        practical = bool(row["practical_flag"]) if has_practical else False

        tag = "*" if significant else ""
        tag += "!" if practical else ""
        annotations.loc[feature, band] = f"{beta:.3f}{tag}"

    fig, ax = plt.subplots(figsize=figsize(max(3.5, len(ordered_features) * 0.35)))
    matrix = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(TEXT_MAP["titles"]["dispersion_heatmap_full"])
    ax.set_xlabel(TEXT_MAP["axes"]["dispersion_band"])
    ax.set_ylabel(TEXT_MAP["axes"]["predictor"])
    ax.set_xticks(range(len(band_order)))
    ax.set_xticklabels([TEXT_MAP["bands"][band] for band in band_order])
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels([TEXT_MAP["features"].get(feature, feature) for feature in ordered_features])
    ax.grid(False)

    for i in range(len(ordered_features)):
        for j in range(len(band_order)):
            text = annotations.iloc[i, j]
            value = float(pivot.iloc[i, j]) if pd.notna(pivot.iloc[i, j]) else 0.0
            if text:
                color = "white" if abs(value) >= 0.05 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(TEXT_MAP["axes"]["effect_size_beta"])
    save_plot(output_dir, FILENAME_MAP["fig_rq13_dispersion_heatmap.png"])


def generate_dispersion_bands_smallmultiples_full(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/F/dispersion_coeffs_r.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    df = df[df["subset"] == "Full"].copy() if "subset" in df.columns else df.copy()
    stratified = df[df["term"].astype(str).str.contains("dispersion_band::", case=False, na=False)].copy()
    if stratified.empty:
        return

    stratified[["band", "feature"]] = stratified["term"].astype(str).str.extract(r"dispersion_band::(.*?):(.*)")
    stratified["band"] = stratified["band"].astype(str).str.replace("_Dispersion", "", regex=False)

    category_map = get_feature_categories()
    stratified = stratified[stratified["feature"].isin(set(category_map.keys()))].copy()
    if stratified.empty:
        return

    stratified["category"] = stratified["feature"].map(category_map).astype(str).str.lower()
    band_order = ["Low", "Medium", "High"]
    stratified["band"] = pd.Categorical(stratified["band"], categories=band_order, ordered=True)

    panel_order = ["performance", "accessibility", "readability", "semantic"]
    panels = [panel for panel in panel_order if panel in set(stratified["category"])]
    schema_terms = list(category_map.keys())
    ordered_features = []
    category_boundaries = []
    current_y = 0
    feature_to_y = {}

    for category in panels:
        subset = stratified[stratified["category"] == category]
        features = [term for term in schema_terms if term in set(subset["feature"].astype(str))]
        for feature in features:
            if feature not in feature_to_y:
                feature_to_y[feature] = current_y
                ordered_features.append(feature)
                current_y += 1
        category_boundaries.append(current_y - 0.5)

    if category_boundaries:
        category_boundaries.pop()

    fig, ax = plt.subplots(figsize=figsize(max(3.5, len(ordered_features) * 0.35)))
    ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")

    for boundary in category_boundaries:
        ax.axhline(boundary, linestyle=":", linewidth=1, color="lightgray")

    offsets = {"Low": -0.22, "Medium": 0.0, "High": 0.22}
    band_colors = {"Low": SUNSET_SUNRISE_PALETTE[0], "Medium": SUNSET_SUNRISE_PALETTE[1], "High": SUNSET_SUNRISE_PALETTE[2]}

    for band in band_order:
        band_df = stratified[stratified["band"].astype(str) == band].copy()
        if band_df.empty:
            continue
        band_df = band_df[band_df["feature"].isin(ordered_features)]

        y = np.array([feature_to_y[str(feature)] for feature in band_df["feature"].astype(str)], dtype=float) + offsets[band]
        x = band_df["effect_size"].astype(float).to_numpy()
        lo = band_df["ci_lower_95"].astype(float).to_numpy()
        hi = band_df["ci_upper_95"].astype(float).to_numpy()
        xerr = np.vstack([x - lo, hi - x])

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            linewidth=1,
            label=TEXT_MAP["bands"][band],
            color=band_colors.get(band, "black"),
        )

    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels([TEXT_MAP["features"].get(feature, feature) for feature in ordered_features])
    ax.invert_yaxis()
    ax.set_xlabel(TEXT_MAP["axes"]["standardized_coefficient"])
    ax.set_title(TEXT_MAP["titles"]["dispersion_smallmultiples_full"])
    ax.legend(title=TEXT_MAP["legend"]["dispersion_band"], loc="upper right")
    save_plot(output_dir, FILENAME_MAP["fig_rq13_dispersion_smallmultiples.png"])


def generate_ablation_ndcg_change(output_dir: Path) -> None:
    path = PROJECT_ROOT / "data/analysis/H/ablation_predictive.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    df = df[~df["set_name"].astype(str).str.contains("Full", case=False, na=False)].copy()
    if df.empty:
        return

    df["subset"] = df["subset"].replace(ENGINE_MAP)
    df["set_name"] = df["set_name"].astype(str).str.replace(r"^-?\s*", "", regex=True)
    if "effect_size" not in df.columns:
        if "ndcg_diff" in df.columns:
            df["effect_size"] = df["ndcg_diff"]
        else:
            return

    category_map = get_feature_categories()
    schema_categories = list(dict.fromkeys([category.capitalize() for category in category_map.values()]))
    ordered_sets = [item for item in schema_categories if item in df["set_name"].values]
    ordered_sets += [item for item in df["set_name"].unique() if item not in ordered_sets]

    available_engines = [engine for engine in ENGINE_ORDER if engine in df["subset"].unique()]
    if available_engines:
        df = df[df["subset"].isin(available_engines)].copy()
        df["subset"] = pd.Categorical(df["subset"], categories=available_engines, ordered=True)

    df["set_name"] = pd.Categorical(df["set_name"], categories=ordered_sets, ordered=True)
    df = df.sort_values(["set_name", "subset"])

    plt.figure(figsize=figsize(max(3.0, len(ordered_sets) * 0.45)))
    sns.barplot(
        data=df,
        x="effect_size",
        y="set_name",
        hue="subset",
        palette=ENGINE_COLORS,
        order=ordered_sets,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title(TEXT_MAP["titles"]["ablation_ndcg_change"])
    plt.xlabel(TEXT_MAP["axes"]["ndcg_change"])
    plt.ylabel(TEXT_MAP["axes"]["ablated_feature_set"])
    plt.legend(title=TEXT_MAP["legend"]["search_engine"], bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot(output_dir, FILENAME_MAP["fig_rq13_ablation_ndcg_change.png"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/reports/thesis/figures")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    generate_semantic_density(output_dir)
    generate_all_feature_trends(output_dir)
    generate_feature_importance(output_dir)
    generate_heterogeneity_all_features(output_dir)
    generate_replication_grid_full(output_dir)
    generate_engine_stratified_forest_full(output_dir)
    generate_dispersion_bands_heatmap_full(output_dir)
    generate_dispersion_bands_smallmultiples_full(output_dir)
    generate_ablation_ndcg_change(output_dir)


if __name__ == "__main__":
    main()
