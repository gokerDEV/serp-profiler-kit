# SERP Profiler Kit: A Framework for Characterizing Ranking Environments

This repository contains the source code for the data collection, processing, feature extraction, and analysis pipeline developed for the academic study titled, "..." The primary objective of this project is to empirically characterize and compare the ranking environments of large-scale search engines, namely Google and Bing.

## Features

- **Data Collection:** Programmatically collects SERP (Search Engine Results Page) data from Google and Bing for a given set of keywords.
- **Web Page Archiving:** Crawls and saves the full HTML content and screenshots of URLs found in the SERPs.
- **Feature Extraction:** Automatically extracts over 12 technical (Lighthouse scores) and content (lexical and semantic) features for each resource.
- **Data Analysis:** Performs comprehensive data analysis, including K-Means clustering, non-parametric statistical tests, Ordinal Logistic Regression, and Random Forest models.
- **Output Generation:** Automatically generates publication-ready tables (in LaTeX format) and figures (in PNG format) from the analysis results.
- **Modular Architecture:** Designed with a modular structure that allows for easy extension and integration of new analyses, features, or data sources.

## Project Structure

The project is organized into a modular directory structure based on task responsibility:

```
.
├── data/                  # Contains raw, processed, and analysis data.
│   ├── raw/
│   ├── processed/
│   └── analysis/
├── figs/                  # Directory for storing figures generated from the analysis.
├── tables/                # Directory for storing tables generated from the analysis.
├── src/                   # Main source code directory.
│   ├── data_collection/   # Scripts for data acquisition from search engines and websites.
│   ├── data_processing/   # Scripts for cleaning, processing, and consolidating raw data.
│   ├── feature_extraction/ # Scripts for extracting features from downloaded content.
│   ├── analysis/          # Scripts for statistical analysis and modeling.
│   ├── generators/        # Scripts for generating figures and tables.
│   └── lib/               # Helper modules and utility functions (e.g., color palettes).
└── run.py                 # Master script to run the entire pipeline via an interactive menu.
```

## Methodological Workflow & Reproducibility

To ensure the study is fully reproducible, the following steps must be followed in order. The master script `run.py` is designed to manage this workflow through an interactive menu.

**Step 1: Data Collection (SERP Acquisition)**
- **Description:** Programmatically collects the top ~20 organic search results from Google and Bing for a predefined list of keywords.
- **Scripts:** `src/data_collection/search_on_google.py`, `src/data_collection/search_on_bing.py`
- **Input:** A keyword list, typically `data/raw/keywords/keywords.csv`.
- **Output:** Raw JSON files containing SERP data, stored in `data/raw/search_results/`.

**Step 2: Data Consolidation & Web Page Crawling**
- **Description:** Consolidates unique URLs from all collected search results into a master file. Subsequently, a scraper visits each unique URL to download and save the full HTML content.
- **Scripts:** `src/data_processing/dump_serps.py`, `src/data_collection/scraper.py`
- **Input:** Raw SERP JSON files.
- **Output:** A master URL list (`serps.csv`) and the corresponding raw HTML files stored in `data/raw/html/`.

**Step 3: Feature Extraction**
- **Description:** Processes the downloaded HTML files and SERP data to compute the 12+ technical and content features defined in the study (e.g., Lighthouse scores, semantic similarity, word count).
- **Scripts:** `src/feature_extraction/extract_features.py`, `src/data_collection/get_pagespeed.py`
- **Input:** Raw HTML files and `serps.csv`.
- **Output:** The final processed dataset containing all features, `dataset_processed.csv`.

**Step 4: Analysis**
- **Description:** Executes all statistical analyses (clustering, hypothesis tests, regression, etc.) on the final processed dataset to answer the research questions (RQ1-RQ4).
- **Script:** `src/analysis/analyze.py`
- **Input:** `dataset_processed.csv`.
- **Output:** A structured JSON file with analysis results (`analysis_results.json`) and summary text reports.

**Step 5: Output Generation**
- **Description:** Uses the structured analysis results to automatically generate all figures and LaTeX tables required for the manuscript.
- **Scripts:** `src/generators/plot.py`, `src/generators/table.py`
- **Input:** `analysis_results.json`.
- **Output:** PNG and `.tex` files saved to the `figs/` and `tables/` directories, respectively.

## Installation & Usage

It is recommended to use a virtual environment to manage dependencies.

**1. Install Dependencies:**
```bash
pip install pandas numpy scikit-learn statsmodels scipy matplotlib seaborn plotly sentence-transformers trafilatura beautifulsoup4 nltk
```
A `requirements.txt` file can be generated from a working environment for easier setup.

**2. Run the Project:**
Instead of running scripts manually, use the interactive master script from the project's root directory:
```bash
python run.py
```
This command will launch a menu prompting you to select which stage of the pipeline you wish to execute.

## Extending the Project

The modular design facilitates future expansion and contribution.

-   **To add a new feature:** Modify the `src/feature_extraction/extract_features.py` script by adding a new function and appending its output as a new column to the main dataframe.
-   **To add a new analysis:** Add a new function to `src/analysis/analyze.py` to address a new research question. You can create a corresponding output generator in the `src/generators/` directory.
-   **To add a new data source (e.g., another search engine):** Create a new script in the `src/data_collection/` directory, using `search_on_google.py` as a template. Integrate the new script into the `run.py` menu.