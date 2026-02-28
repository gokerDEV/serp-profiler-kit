# News SERP Dataset Pipeline Architecture

## 1) Scope

This document defines the end-to-end pipeline for building a research-ready dataset from:

- `data/serp.csv` (master SERP results)
- scraping artifacts under `scraping/<iteration-source>/<batch>/`
- feature extraction steps A..G
- final merge step Z

The pipeline is designed for:

- reproducibility
- clear step boundaries
- large-scale processing
- easy restart and incremental updates
- Python/R-friendly outputs (`.parquet`)

---

## 2) Directory Structure

### Project code layout

```text
src/
  collection/
    ... (see COLLECTION.md)
  extraction/
    ... (see EXTRACTION.md)
  analysis/
    ... (see ANALYSIS.md)
  generators/
    ... (see GENERATOR.md)
```

### Data layout

```text
data/
  keywords.csv
  serp.csv
  index.parquet
  features/
    A/
      runtime_metrics.parquet
    B/
      accessibility_metrics.parquet
    C/
      html_structure.parquet
    D/
      text_readability.parquet
    E/
      semantic_similarity.parquet
    F/
      vision_features.parquet      # optional
  dataset.parquet
  reports/
    tables/
    figures/
```

### Raw artifact layout

```text
data/raw/news/
data/raw/serp/
   brave/
     <section>/<query_id>.json
   mojeek/
     <section>/<query_id>.json
   google/
     <section>/<query_id>.json
data/raw/scraping/
  01-bot/
    000/
      <file_name>.html
      <file_name>.json
      <file_name>.png
    ...
  02-extension/
    000/
      ...
  03-extension/
    000/
      ...
```

Iteration increases globally even if source changes: `01-bot`, `02-extension`, `03-extension`, ...

---

## 3) Master Inputs

### `data/serp.csv`

Required columns:

- `id` (query/article identifier)
- `section`
- `search_term`
- `search_engine`
- `rank`
- `link`
- `file_name`
- `title`
- `snippet`

### `data/keywords.csv`

Seed and shareable metadata table (no copyrighted full article body).

---

## 4) Detailed Documentation

The pipeline logic is split into the following detailed documents:

- **[COLLECTION.md](COLLECTION.md)**: Details on step A (News), B (SERP), C (Scraping), and Z (Build Index).
- **[EXTRACTION.md](EXTRACTION.md)**: Details on feature extraction (steps A-F) and the final merge (step Z).
- **[ANALYSIS.md](ANALYSIS.md)**: Details on the analysis layer (descriptive stats, etc.).
- **[GENERATOR.md](GENERATOR.md)**: Details on generating report artifacts (tables, figures).

---

## 5) Operational Notes

- Intermediate and final artifacts are Parquet.
- Each step logs progress and failures.
- Each step output includes:
  - `record_id`
  - feature columns
  - `step_status`, `step_reason`
- Missing/invalid artifacts are resolved in A and propagated downstream.

---

## 6) Versioning & Reproducibility

Recommended metadata per step (logs or sidecar JSON):

- script name/version
- run timestamp
- input paths
- row counts
- null ratios
- error/anomaly counts
- model/version identifiers (for embedding-based steps)
