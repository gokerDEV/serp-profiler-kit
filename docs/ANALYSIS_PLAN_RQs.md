# Analysis Plan

**What explains SERP visibility best—semantic relevance or measurable page-quality signals (runtime, accessibility, readability)—and are these relationships consistent across engines?**

## 0.2 Pipeline map (implementation)

| RQ          | Step           | Script Path                             | Description                                              |
| :---------- | :------------- | :-------------------------------------- | :------------------------------------------------------- |
| **RQ1**     | **Analysis C** | `src/analysis/C/rq1_concentration.py`   | Rank concentration (Gini/Entropy) & semantic dispersion. |
| **RQ2–RQ5** | **Analysis D** | `src/analysis/D/confirmatory_models.py` | Main regression models (Top-k, FE).                      |
| **RQ6**     | **Analysis E** | `src/analysis/E/engine_heterogeneity.py`| Engine-specific signal analysis.                         |
| **RQ7**     | **Analysis F** | `src/analysis/F/query_difficulty.py`    | Query difficulty bands & heterogeneity.                  |
| **RQ8**     | **Analysis G** | `src/analysis/G/robustness.py`          | Robustness checks & sensitivity.                         |
| **RQ9**     | **Analysis H** | `src/analysis/H/ablation.py`            | Composite models & ablation.                             |

---

## 1) Global standards (apply to all RQs unless stated)

### 1.1 Rank direction (sign standard)

All outcomes are reported in a **higher-is-better** direction. When the raw outcome is **rank** (1=best), we transform/report it so that **larger values indicate higher visibility** (e.g., reciprocal rank or within-query percentile).

### 1.2 Outcome families (pre-registered)

We use three outcome families (to avoid dependence on any single modeling assumption):

1. **Top-k visibility (binary):** k ∈ {5, 10, 20}.
2. **Continuous visibility:** reciprocal rank (1/rank) and/or within-query percentile rank (higher=better).
3. **Threshold (sequential cuts):** pre-registered cut points aligned to available rank ranges (e.g., Top20 vs rest; if extended ranks exist, Top40/60/80 vs rest).

### 1.3 Query-level weighting

Primary analyses treat **each observation equally**. Sensitivity analyses apply **equal weight per search_term** (query-weighted) to prevent any single query from dominating estimates.

### 1.4 Split strategy and uncertainty (R + Python)

* **Leakage control (predictive/corroborative steps):** grouped splits by **search_term**.
* **Uncertainty (inferential models):** **bootstrap confidence intervals clustered by search_term** (resampling queries; default B=1000 unless computationally constrained, then B≥300).

### 1.5 Multiple testing (BH-FDR) — scope

**Confirmatory family** = *main coefficients only* for **RQ2/RQ3/RQ4/RQ5/RQ6** (engine-wise).
Included in BH-FDR:

* coefficients for the pre-registered predictor blocks explicitly listed in each RQ’s **Primary model/test** section (excluding intercepts and nuisance controls).

Excluded from BH-FDR:

* model diagnostics (VIF, residual checks, pseudo-R²),
* LTR permutation/SHAP, calibration curves,
* robustness-only re-estimates, influence/outlier audits.

Results tables include: **p_raw**, **p_fdr** (where applicable), **effect_size**, **practical_flag**.

### 1.6 Missing-data decision rule (model-level)

1. Run **complete-case confirmatory** model (primary).
2. If complete-case removes **>15%** of usable observations for that model, run **indicator-augmented sensitivity** and report both.
3. If sensitivity **reverses sign** or changes the interpretation materially, the main-text claim is **downgraded to inconclusive** and both estimates are reported together.

### 1.7 Multicollinearity (VIF thresholds + mitigation order)

* **VIF > 5**: warning; report VIF table.
* **VIF > 10**: mitigation required (apply in this order):

  1. drop-one (within the collinear block; retain **sim_content** as anchor),
  2. ridge sensitivity,
  3. block-PCA sensitivity (appendix).

### 1.8 Practical significance thresholds

* **OR in [0.95, 1.05]** → negligible.
* **|β*| < 0.03** (standardized beta) → negligible.
  Tables must include **practical_flag** based on these rules.

### 1.9 Dependence / duplicates (standard errors)

Primary SE: **cluster-robust by search_term**.
If feasible: two-way clustering (search_term × domain OR search_term × url_hash).
If not feasible: two-way clustered SE on a **stratified, engine-balanced subset**, plus a directional-consistency check against the primary model.

**Logit Model Stability & Variance Checks:**
1. **Solver:** Use `bfgs` instead of `newton` for Logit optimization to prevent convergence failures.
2. **Variance Check:** Before running models, use a `check_variance` function to ensure variables are not constant (`nunique=1`) within each analysis subset (e.g., per engine).

### 1.10 Reporting effect sizes (uniform standard)

* **Binary (Top-k):** OR + AME (+ CI95).
* **Continuous (FE/OLS/GLM):** standardized beta (β*) (+ CI95).
* **LTR (corroborative only):** ΔNDCG (or ΔMAP) under ablations + permutation/SHAP **rank stability** summaries.

> LTR is corroborative; inferential claims come from fixed-effects / clustered models and threshold models.

### 1.11 Evidence tiers (interpretation discipline)

All reported outputs carry **evidence_tier ∈ {confirmatory, corroborative}**.

* Confirmatory: FE / Top-k / threshold models for RQ2–RQ6 (includes p_fdr where applicable).
* Corroborative: LTR / SHAP / permutation / predictive evaluations (p_fdr is NA).

### 1.12 Readability metric policy

Confirmatory readability set (RQ3):

* Flesch Reading Ease
* Flesch–Kincaid Grade

**Note:** `gunning_fog` (VIF=38.52) and `smog_index` (VIF=11.41) are dropped from the confirmatory set due to high multicollinearity (VIF > 10).

ARI (Automated Readability Index) is **appendix/sensitivity only** and excluded from BH-FDR.

### 1.13 Source-domain definition and seed-match handling

Queries are constructed from **Guardian headlines**. To control for potential source-seeding effects, the dataset includes:

* `source_domain` (string; empty/NA if none)
* `is_source_domain` (bool): `TRUE` iff the result URL’s canonical domain equals `source_domain`.

Analytical rule:

* Primary estimates remain on the full dataset.
* We report stratified robustness panels for (a) `is_source_domain = TRUE` vs `FALSE`, and (b) exclusion of `is_source_domain = TRUE` rows.
* If excluding `is_source_domain = TRUE` materially changes signs or conclusions for core predictors, claims are qualified as **source-sensitive**.

---

---

## 2) Research Questions

**Standard RQ template:** Question · Hypothesis (H1/H0) · Primary model/test · Robustness checks · Key outputs

---

## RQ1

### Question

How concentrated or dispersed is ranking visibility (domain diversity) across engines at the **query level**, and how do feature characteristics distribute across rank bins and data subsets?

### Hypothesis

* **H1:** Engines differ in query-level rank concentration (domain diversity) and feature dispersion across top-k strata.
* **H0:** No meaningful cross-engine differences in these distributional properties.

### Primary model/test

Descriptive + distributional tests organized across a **4x11x3x3x3 dimensional matrix** (4 categories × 11 features × 3 engines × 3 rank bins × 3 subsets):

* Query-level concentration indices (entropy, Gini-like measures, monopoly rate) over rank positions.
* Feature dispersion and rank-binned trends (Top 1-3, Rank 4-10, Rank 11-20) across all 4 categories (semantic, readability, performance, accessibility) covering 11 independent features.
* Granular evaluation across 3 engines (Google, Mojeek, Brave) and 3 subsets (Full, NoSource, Source).

### Robustness checks

* Recompute under url_normalization / dedup variants.
* Repeat with query-weighted summary statistics.

### Key outputs

* Full dimensional rank-binned feature trends table (feature × engine × rank bin × subset).
* Engine×query concentration summaries (Gini, Entropy, Monopoly rates) partitioned by subset.
* Semantic similarity distribution data (JSON for `src/generators`).
* Cross-engine dispersion comparison (effect sizes for group differences).
* **Note:** Visualization generation (plots) is handled in `src/generators` using the data exported here.

---

## RQ2

### Question

How strongly does **semantic relevance** explain visibility (higher-is-better outcomes) within queries?

### Hypothesis

* **H1:** Higher semantic similarity (especially sim_content) is associated with higher visibility.
* **H0:** Semantic similarity does not relate to visibility once query effects are controlled.

### Primary model/test

Query fixed-effects models (continuous outcomes) and Top-k logits:

* Continuous FE: visibility ~ sim_title + sim_description + sim_h1 + **sim_content** + controls
* Top-k: Pr(Top-k) ~ same semantic block + controls
  Controls: minimal, pre-registered technical controls that do not re-introduce post-treatment bias (e.g., extraction status flags if needed).

### Robustness checks

* Alternative semantic block specification (drop-one; ridge).
* Outcome-family triangulation: Top-k + continuous + threshold cuts.
* Engine-stratified fits (Google/Brave/Mojeek).
* Source-domain panels: (i) `is_source_domain=TRUE` vs `FALSE`, (ii) exclude `is_source_domain=TRUE`.

### Key outputs

* Main results table rows for semantic predictors (OR+AME; β*).
* Partial contribution: Δpseudo-R² / ΔAIC when adding semantic block.
* Practical significance flags.

---

## RQ3

### Question

Are **readability** signals associated with visibility after controlling for semantics and core technical signals (**association, non-causal**)?

### Hypothesis

* **H1:** Readability metrics show systematic association with visibility beyond semantics.
* **H0:** Readability metrics add no incremental association once semantics/technical controls are included.

### Primary model/test

Query FE + Top-k logits:

* visibility ~ semantic block (RQ2) + readability block + core technical controls
  Readability block (confirmatory): Flesch Reading Ease, Flesch–Kincaid Grade.

**Directional Note for Readability:** To maintain consistent interpretability across coefficient plots ("a higher/positive coefficient points to a better/targeted outcome"), the `flesch_kincaid_grade` metric is parsed with a directional multiplier of `-1` as defined in `schema_v1.yml` before models are fitted.

**Interpretation boundary:** Readability metrics may capture topic/audience targeting and editorial strategy as much as linguistic complexity. Effects are treated as associational (not causal) and interpreted conservatively.

### Robustness checks

* Winsorized metrics (1–99%) reported alongside raw.
* Engine-stratified.
* Query difficulty bands as interaction or stratification (see RQ6).

### Key outputs

* Readability coefficient table with p_raw/p_fdr/effect_size/practical_flag.
* Side-by-side raw vs winsorized estimates.
* Influence summary (appendix; no ad hoc exclusions).

---

## RQ4

### Question

Do **page load speed and visual stability** signals (Performance) have an independent effect on ranking after controlling for semantic relevance?

### Hypothesis

* **H1:** Lower `ttfb_ms`, `lcp_ms`, and `cls` values (indicating higher performance) represent higher visibility.
* **H0:** Performance signals do not add independent association beyond semantics.

### Primary model/test

Query fixed-effects with clustered SE (primary), plus Top-k and threshold confirmations:

**Core specification (continuous, higher-is-better oriented):**
`visibility ~ sim_content + ttfb_ms + lcp_ms + cls + controls`

**Directional Note for Performance:** To preserve the interpretation that "a higher, positive coefficient equates to a better or healthier dimension," lower latency and lower shift are desirable. Consequently, `lcp_ms`, `ttfb_ms` and `cls` are mathematically inverted using the `-1` multiplier defined in `schema_v1.yml` prior to inferential standardisation.

### Robustness checks

* Winsorization (1–99%) for runtime metrics.
* log-transform ttfb_ms and lcp_ms sensitivity.
* VIF protocol (Section 1.7).
* Cluster-robust SE by search_term; two-way clustering (Section 1.9).

### Key outputs

* Incremental contribution beyond semantics: Δpseudo-R² / ΔAIC; LR test where applicable.
* Effect sizes (β*, OR+AME) + practical_flag.

---

## RQ5 (New)

### Question

Do **Accessibility** scores (error/pass rates) describe ranking success independently of performance and semantic signals?

### Hypothesis

* **H1:** Higher `axe_score` and `contrast_score` values are positively associated with higher visibility.
* **H0:** Accessibility scores do not add independent association beyond semantics and performance.

### Primary model/test

**Metric Calculation:**
* `axe_score = axe_passes_total / (axe_passes_total + axe_violations_total)`
* `contrast_score = contrast_passes_count / (contrast_passes_count + contrast_violation_count)`

**Model:**
`visibility ~ sim_content + axe_score + contrast_score + axe_impact_critical + controls`

### Robustness checks

* Winsorization (1–99%).
* Check variance of scores (some might be constant if almost all sites pass).

### Key outputs

* Accessibility coefficient table.

---

## RQ6 (Formerly RQ5)

### Question

Are signal–visibility associations **engine-specific**, and do they replicate across Google, Brave, and Mojeek?

### Hypothesis

* **H1:** Coefficients differ by engine (interaction or stratified differences).
* **H0:** Coefficients are consistent across engines.

### Primary model/test

* Pooled model with interactions: visibility ~ (Core predictors from RQ4/RQ5) × search_engine + query FE.
* Engine-stratified models as replication.

### Robustness checks

* Directional consistency across engines and outcome families.
* Two-way clustering feasibility rule (Section 1.9).
* Replication grid includes source-domain panels (`is_source_domain=TRUE` vs `FALSE`).

### Key outputs

* Interaction table + engine-stratified coefficient table.
* Replication grid: sign agreement, practical_flag stability, CI overlap.

---

## RQ7 (Formerly RQ6)

### Question

Are associations consistent across **query difficulty bands** (e.g., low vs high semantic separability) and top-k strata?

### Hypothesis

* **H1:** Signal strengths vary by query difficulty band and/or top-k stratum.
* **H0:** No systematic heterogeneity across difficulty bands.

### Primary model/test

* Pre-registered query difficulty bands (constructed from within-query semantic dispersion/mean).
* Models: visibility ~ core predictors + (core predictors × difficulty_band) with query FE where applicable.
* Complementary stratified fits by band and by top-k.

### Robustness checks

* Alternative band definitions (quantile-based; documented).
* Engine-stratified.

### Key outputs

* Band-by-band coefficient summaries + heterogeneity tests.
* Consistency criterion (for “consistent” claims): sign agreement + practical magnitude not engine-dominated across ≥2 outcome families.

---

## RQ8 (Formerly RQ7)

### Question

Are findings robust to outliers, winsorization, and missingness handling, and do they preserve directional conclusions?

### Hypothesis

* **H1:** Core conclusions remain directionally consistent under robustness regimes.
* **H0:** Conclusions materially change under plausible robustness regimes.

### Primary model/test

Robustness umbrella over RQ2–RQ7 primary models:

* Raw vs winsorized (1–99%) side-by-side.
* Missingness rule application (Section 1.6).
* Query-weighted sensitivity (Section 1.3).
* Two-way clustering if feasible; fallback protocol otherwise (Section 1.9).
* Source-domain robustness: report (i) full, (ii) `is_source_domain=FALSE` only, plus (iii) `is_source_domain=TRUE` descriptive panel.

### Robustness checks

* log-transform sensitivity for heavy-tail runtime metrics.
* Influence checks (Cook-like diagnostics where applicable).

### Key outputs

* Robustness table: delta estimates, direction match, inference match, practical_flag stability.
* One main robustness summary table included in the paper.

---

## RQ9 (Formerly RQ8)

### Question

How much incremental explanatory value comes from **composite models and ablation** across semantic, readability, runtime, and accessibility blocks (corroborative + inferential alignment)?

### Hypothesis

* **H1:** Adding quality blocks to semantics improves fit and predictive ranking metrics; removing them degrades performance.
* **H0:** Composite/ablation changes are negligible once semantics is included.

### Primary model/test

Two-layer approach:

1. **Inferential:** block-wise nested models (semantics → +readability → +runtime+a11y) with Δfit (Δpseudo-R²/ΔAIC) and coefficient stability.
2. **Corroborative LTR:** grouped-by-query evaluation with ΔNDCG (and permutation/SHAP rank stability) under block ablations.

### Robustness checks

* Engine-stratified composite fits.
* Outcome-family triangulation (Top-k/continuous/threshold).
* Compute-feasibility rule: confirmatory core on full data; heavy sensitivity runs on engine-parallel or stratified subsets with explicit reporting.

### Key outputs

* Ablation contribution table (Δfit; ΔNDCG).
* Predictor/block importance stability summary (permutation/SHAP ranks).
* Narrative closure statement aligning with Section 0.

---

## 3) Software stack

### R packages

`data.table`, `arrow`, `fixest`, `ordinal`, `lme4`, `clubSandwich`, `emmeans`, `performance`, `DHARMa`, `ggplot2`

### Python packages

`pandas`, `pyarrow`, `numpy`, `statsmodels`, `linearmodels`, `scikit-learn`, `scipy`, `pingouin`, `matplotlib`

---

## 4) Reporting templates

### 4.1 Main results table schema

**effect_size | CI95 | p_raw | p_fdr | practical_flag | n | model_family | search_engine | evidence_tier**

Where:

* effect_size = OR (Top-k) or β* (continuous) or ΔNDCG (LTR block-level table).
* model_family ∈ {Top-k logit, FE-continuous, Threshold/sequential, LTR}.

### 4.2 Robustness table

**spec_id | delta_from_primary | direction_match | inference_match | practical_flag_stability | notes | evidence_tier**

### 4.3 Required provenance (all published tables)

All published tables must include:

* `dataset_id`
* `dataset_variant` ∈ {`all_raw`, `clean_winsorized`}
* `generated_at` (ISO-8601, UTC)

Recommended (release manifest): `seed`, `split_manifest_id`, `spec_id`, `code_version`.

---

## 5) Final narrative sentence (for Abstract + Conclusion)

**“Ranking variation is primarily semantic, but measurable runtime/accessibility and readability signals add incremental explanatory power with engine-specific heterogeneity.”**

---

# Appendix A — RQ1 concentration metrics (operational definitions)

For each `(search_term, search_engine)` group with `n` results:

1. **Visibility mass** (per rank `r = 1..n`):
   [ v_r = 1/r ]

2. **Domain aggregation** for each domain `d` in the set of unique domains `D`:
   [ V_d = \sum_{i \in \text{results}(d)} v_{r_i} ]

3. **Domain shares**:
   [ p_d = V_d / \sum_{k \in D} V_k ]

4. **Entropy** (domain-level) and normalized entropy:
   [ H = -\sum_{d \in D} p_d \log_2(p_d) ]
   [ H_{norm} = H / \log_2(|D|) ]
   If `|D| = 1`, set `H_norm = 0`.

5. **Gini** on domain shares `{p_d}`:

* If `|D| = 1`, return `G = NA` (inequality is undefined with a single unit).
* Otherwise, sort shares ascending `p_(1) ≤ ... ≤ p_(|D|)`:
  [ G = \frac{2\sum_{i=1}^{|D|} i,p_{(i)}}{|D|\sum_{i=1}^{|D|} p_{(i)}} - \frac{|D|+1}{|D|} ]

6. **Minimum observation rule**: If `n_results < 5`, set `status = "insufficient_data"` and keep rows for QA reporting.

QA monitoring (reported): correlations between concentration metrics and `n_results` / `n_domains`.

---

# Appendix B — BH-FDR whitelist (variable names)

Confirmatory BH-FDR is applied only to the following coefficient families (and their pre-registered RQ5 engine interactions):

* Semantic: `sim_title`, `sim_description`, `sim_h1`, `sim_content`
* Readability (confirmatory): `flesch_reading_ease`, `flesch_kincaid_grade`
* Performance (confirmatory): `lcp_ms`, `ttfb_ms`, `cls`
* Accessibility (confirmatory): `axe_score`, `contrast_score`, `axe_impact_critical`
---

# Appendix C — Status/reason semantics (ETL/QC)

*   If `status_* = "ok"`, `reason_*` may be empty/NA.
*   If `status_* != "ok"`, `reason_*` must be non-empty (reason code/message).

---

# Appendix D — Field naming contract

All ETL, analysis, and reporting artifacts use only the field names defined in `schema_v1`. Artifacts containing out-of-schema field names are treated as validation errors and are not exported.

---

# 6) Visualization Strategy (Generators)

The following figures are generated by `src/generators/B/figures.py` using the analysis outputs:

| Figure ID | RQ Link | Description | Source Data |
| :--- | :--- | :--- | :--- |
| **Fig 1** | RQ1 | **Rank Concentration & Dispatch:** Violin/Bar plots of Gini and Entropy by engine. | `data/analysis/C/rq1_concentration.parquet` |
| **Fig 2** | RQ1 | **Semantic Similarity Distribution:** Density/ECDF of semantic scores by engine. | `data/analysis/C/rq1_viz.json` |
| **Fig 3** | RQ2-5 | **Coefficient Forest Plot:** Standardized effect sizes for Semantics, Readability, Performance, Accessibility. | `data/analysis/D/confirmatory_coeffs.csv` |
| **Fig 4** | RQ6 | **Engine Heterogeneity:** Interaction effects (Forest plot comparing engines). | `data/analysis/E/heterogeneity_coeffs_r.csv` |
| **Fig 5** | RQ7 | **Query Difficulty:** Coefficient stability across difficulty bands. | `data/analysis/F/difficulty_coeffs_r.csv` |
| **Fig 6** | RQ9 | **Ablation Performance:** ΔNDCG impact of removing feature blocks. | `data/analysis/H/ablation_predictive.csv` |

**Format:** All figures are exported as **300 DPI PNG** files in `data/reports/figures/`.
