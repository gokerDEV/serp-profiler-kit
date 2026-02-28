# Analysis Layer

This document describes the analysis layer which consumes the extracted features and final dataset for high-level insights.

---

## Overview

- **Location:** `src/analysis/...`
- **Output:** Tables, plots, and reports under `data/reports/` and `data/analyzed/`.

---

## A: Descriptive Statistics

- **Script:** `src/analysis/A/index_stats.py`
- **Input:** `data/index.parquet` (Core Index), `data/keywords.csv`, `data/serp.csv`.
- **Output:** `stdout` (Console Report) saved to `data/reports/collection.md`.

Purpose:
- Generates a summary report detailing healthy vs failed scrapes.
- Analyzes distribution of keywords, SERP positions, and HTML sizes.
- Validates the base "Index" integrity before feature extraction.

## B: Outlier Analysis & Flagging

- **Script:** `src/analysis/B/outlier_analysis.py`
- **Primary Inputs:**
    - `data/index.parquet`
    - `data/features/*/*.parquet` (A, B, C, D, E)
- **Primary Outputs:**
    - `data/outliers.parquet` (Flags: `is_outlier_technical`, `is_outlier_readability`...)
    - `data/reports/outlier_report.md` (Detailed distribution comparison)

Purpose:
1. **Consistency Check:** Verifies that all 'ok' records in Index exist in Feature sets (Match Rate).
2. **Anomaly Detection:** Uses IQR (Inter-Quartile Range) method (default threshold 3.0) to flag extreme values.
   - **Technical:** Load times, sizes.
   - **Readability:** Word counts (e.g. dictionaries or empty pages).
   - **Similarity:** Relevance scores (e.g. cookie banners).
3. **Deep Dive:** Programmatically inspects extreme outliers (e.g., retrieving content snippets) to validate if they are garbage or valid edge cases.

## Usage in Pipeline

1. **Step B (Outlier Analysis)** is run *after* all Feature Extractions (A-E).
2. The resulting `outliers.parquet` is fed into **Step Z (Merge)**.
3. The final dataset includes these outlier flags, allowing downstream analysis to easily filter "Clean" vs "All" data.

## Additional Analysis

(Future analysis modules can be added here following the same pattern, e.g., Correlation Analysis).

