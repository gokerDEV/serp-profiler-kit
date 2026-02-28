# Generators

This layer is responsible for converting the analysis results and the core datasets into consumable output formats like LaTeX tables and PNG figures for the final research paper/report.

---

## Directory Structure

```text
src/
  generators/
    A/
      tables.py
    B/
      figures.py
```

## Steps

### A Step: Tables

- **Script:** `src/generators/A/tables.py`
- **Output:** LaTeX tables saved in `data/reports/tables/`

This script parses parquet datasets and CSV analysis outputs to generate heavily formatted ready-to-use LaTeX tables. The resulting tables include:

1. **`table_dataset_distribution.tex`**: Shows how "ok" results are distributed across top rank bins (Top 1-3, 4-10, etc.) for each search engine.
2. **`table_source_domain.tex`**: Shows the top source domain's visibility breakdown by rank.
3. **`table_feature_match.tex`**: Shows the missingness and match rates for key representative features across semantic, readability, accessibility, and performance metrics.
4. **`table_html_size.tex`**: Descriptive statistics regarding HTML file sizes.
5. **`table_model_eligibility.tex`**: Reports how much data was lost during modeling and which warning flags were encountered.
6. **`table_rq1_summary.tex`**: Engine rank concentration summary including Mean Gini, Mean Normalized Entropy, and Monopoly rate.
7. **`table_rq1_effects.tex`**: Engine pairwise comparisons providing mean difference, Cohen's d effect sizes, and practical interpretations for Gini and semantic similarity metrics.
8. **`table_rq1_trends_non_semantic.tex`**: Rank-Binned trends for non-semantic features.
9. **`table_rq1_trends_semantic.tex`**: Rank-Binned trends for semantic similarity features.
10. **`table_cooks_d.tex`**: Distribution of Cook's D influence errors.
11. **`table_confirmatory_main.tex`**: The primary results table for Confirmatory Regression Models (RQ2-RQ5) demonstrating the estimates, 95% CIs, baseline p-values, and practical significance flags.
12. **`table_confirmatory_supplementary.tex`**: Supplementary results and robust outputs for the conflicting or supplementary models.
13. **`table_nested_model_fit.tex`**: Compares fit statistics for the nested regression models.
14. **`table_ablation_predictive.tex`**: Ablation tests detailing the NDCG@10 scores across the full model and various component-ablated models.
15. **`table_ablation_stability.tex`**: Overview of stability metrics and variables calculated during ablation phases.

---

### B Step: Figures

- **Script:** `src/generators/B/figures.py`
- **Output:** Matplotlib and Seaborn plots saved as PNGs in `data/reports/figures/`

This script creates complex scientific figures by strictly observing the metric categorization rules established in `src/schema_v1.yml`.

1. **`fig_semantic_density.png`**: Density KDE plots displaying semantic similarity distribution across engines.
2. **`fig_semantic_trends.png`**: Point plots tracing mean semantic similarity metrics across different rank tiers.
3. **`fig_all_feature_trends.png`**: Comprehensive trend tracking across ranks for all extracted features (beyond just semantics).
4. **`fig_semantic_decay_vis.png`**: Visualizes the decay models for semantic similarities over increasing rank positions.
5. **`fig_rank_concentration.png`**: Violin plots detailing the Normalized Entropy and Gini Coefficient for engine structural differences side-by-side. 
6. **`fig_feature_importance.png`**: Specifically visualizes the output of Learning-to-Rank permutation importance correctly mapped to metric categories.
7. **`fig_paired_forest_nosource.png`**: A specialized paired forest plot demonstrating coefficients when source features are blinded/excluded.
8. **`fig_rq6_engine_stratified_forest.png`**: Shows engine-stratified variable estimates to answer RQ6 (Heterogeneity).
9. **`fig_difficulty_bands_heatmap_{subset}.png` / `fig_difficulty_bands_heatmap_Delta.png`**: Heatmaps detailing exactly how classification query difficulty bands shift upon ablating data subsets, along with metric delta mapping.
10. **`fig_difficulty_bands_smallmultiples_{subset}.png`**: Separate small-multiple difficulty representations per subset.
11. **`fig_confirmatory_beta.png`**: Forest plot showing combined confirmatory model standardized Beta coefficients aligned in top-to-bottom sequence of the schema.
12. **`fig_confirmatory_or.png`**: Forest plot showing the Odds Ratios counterparts for relevant confirmatory models.
13. **`fig_heterogeneity_all_features.png`**: Grouped bar plot that analyzes engine-specific standardized effects systematically grouped and aggregated across metric categories.
14. **`fig_ablation_ndcg_change.png`**: Visualization showing the change in NDCG@10 metrics under feature-set ablation tests per engine, rigidly ordered according to the overall metrics sequence.
