# SERP Profiler Kit: A Comprehensive Framework for Characterizing Search Engine Ranking Environments

This repository contains a complete research pipeline for empirically characterizing and comparing the ranking environments of large-scale search engines (Google and Bing). The framework enables systematic data collection, feature extraction, statistical analysis, and automated generation of publication-ready outputs for academic research on search engine ranking algorithms.

## 🎯 Research Objectives

The SERP Profiler Kit addresses four key research questions (RQ1-RQ4) through comprehensive data analysis:

- **RQ1: Clustering Analysis** - Identifies distinct ranking profiles using K-Means clustering on technical and content features
- **RQ2: Visibility Analysis** - Examines feature importance across different ranking tiers (high/medium/low)
- **RQ3: Engine Comparison** - Compares ranking characteristics between Google and Bing
- **RQ4: Priority Analysis** - Determines which features most strongly predict ranking positions using ordinal logistic regression

## 🚀 Key Features

### Data Collection & Processing
- **Multi-Engine SERP Collection**: Automated collection from Google Custom Search API and Bing Search API
- **Web Page Archiving**: Full HTML content and screenshot capture using headless browsers
- **Robust Error Handling**: Comprehensive exception tracking and retry mechanisms
- **Batch Processing**: Efficient parallel processing with configurable batch sizes

### Feature Extraction
- **Technical Features (4)**: Lighthouse performance, accessibility, best practices, and SEO scores
- **Content Features (8)**: Query presence in title/H1, semantic similarity, word count, query density
- **Semantic Analysis**: Advanced NLP-based content relevance scoring using sentence transformers

### Statistical Analysis
- **Clustering**: K-Means with optimal cluster determination (elbow method, silhouette analysis)
- **Non-Parametric Tests**: Kruskal-Wallis, Mann-Whitney U, Chi-square tests
- **Regression Models**: Ordinal logistic regression with proportional odds assumption testing
- **Feature Importance**: Random Forest analysis for feature ranking

### Output Generation
- **Publication-Ready Plots**: PNG format with consistent styling and color schemes
- **LaTeX Tables**: Academic paper-ready tables with statistical significance indicators
- **Comprehensive Reports**: JSON-structured analysis results for further processing

## 📁 Project Architecture

```
serp-profiler-kit/
├── data/                          # Data storage
│   ├── raw/                       # Raw collected data
│   │   ├── search_results/        # SERP JSON files
│   │   ├── html/                  # Downloaded web pages
│   │   ├── screenshots/           # Page screenshots
│   │   └── pagespeed/             # Lighthouse scores
│   ├── processed/                 # Intermediate processed data
│   ├── datasets/                  # Final analysis datasets
│   └── analysis/                  # Analysis outputs
│       ├── plots/                 # Generated figures
│       └── tables/                # LaTeX tables
├── src/                           # Source code
│   ├── data_collection/           # Data acquisition modules
│   ├── data_processing/           # Data cleaning and preparation
│   ├── feature_extraction/        # Feature computation
│   ├── analysis/                  # Statistical analysis
│   ├── generators/                # Output generation
│   └── lib/                       # Utility functions
├── run.py                         # Interactive pipeline controller
└── requirements.txt               # Python dependencies
```

## 🔧 Module Details

### Data Collection (`src/data_collection/`)

| Module                | Purpose                | Key Features                                         |
| --------------------- | ---------------------- | ---------------------------------------------------- |
| `search_on_google.py` | Google SERP collection | Custom Search API, pagination, rate limiting         |
| `search_on_bing.py`   | Bing SERP collection   | Bing Search API, result parsing                      |
| `scraper.py`          | Web page crawling      | Headless browser, screenshot capture, error handling |
| `get_pagespeed.py`    | Lighthouse scoring     | Performance metrics via PageSpeed Insights API       |
| `site_search_*.py`    | Site-specific search   | Domain-restricted search capabilities                |

### Data Processing (`src/data_processing/`)

| Module                  | Purpose                | Key Features                              |
| ----------------------- | ---------------------- | ----------------------------------------- |
| `dump_serps.py`         | SERP consolidation     | URL deduplication, result aggregation     |
| `acceptance.py`         | Data validation        | Quality filtering, completeness checks    |
| `feature_extraction.py` | Feature computation    | NLP processing, semantic analysis         |
| `combine.py`            | Data integration       | Feature and performance score merging     |
| `outliers.py`           | Anomaly detection      | Statistical outlier identification        |
| `dataset.py`            | Final dataset creation | Clean, analysis-ready dataset preparation |

### Analysis (`src/analysis/`)

| Module       | Purpose              | Key Features                       |
| ------------ | -------------------- | ---------------------------------- |
| `analyze.py` | Statistical analysis | RQ1-RQ4 complete analysis pipeline |

**Research Questions Addressed:**
- **RQ1**: K-Means clustering with optimal K determination
- **RQ2**: Feature importance analysis by ranking tier
- **RQ3**: Cross-engine comparison using non-parametric tests
- **RQ4**: Ordinal logistic regression for ranking prediction

### Output Generation (`src/generators/`)

| Module     | Purpose                | Key Features                                    |
| ---------- | ---------------------- | ----------------------------------------------- |
| `plot.py`  | Figure generation      | 8+ publication-ready visualizations             |
| `table.py` | LaTeX table generation | Statistical tables with significance indicators |

### Utilities (`src/lib/`)

| Module      | Purpose        | Key Features                            |
| ----------- | -------------- | --------------------------------------- |
| `utils.py`  | Core utilities | File handling, logging, data processing |
| `colors.py` | Visualization  | Consistent color palettes and themes    |

## 📊 Extracted Features

### Technical Features (Lighthouse Scores)
- **Performance Score**: Page load performance (0-100)
- **Accessibility Score**: Web accessibility compliance (0-100)
- **Best Practices Score**: Security and best practices (0-100)
- **SEO Score**: Search engine optimization (0-100)

### Content Features
- **Query Presence**: Binary indicators for query in title/H1
- **Exact Match**: Precise query matching in title/H1
- **Query Density**: Query frequency in body content
- **Semantic Similarity**: NLP-based relevance scores (title/body vs query)
- **Word Count**: Content length measurement

## 🔬 Statistical Methods

### Clustering Analysis (RQ1)
- **Algorithm**: K-Means clustering
- **Optimal K**: Elbow method + Silhouette analysis
- **Validation**: Kruskal-Wallis tests for cluster significance
- **Metrics**: WCSS, Silhouette, Calinski-Harabasz, Davies-Bouldin scores

### Feature Importance (RQ2)
- **Method**: Non-parametric statistical tests
- **Tests**: Kruskal-Wallis, Mann-Whitney U
- **Grouping**: High (1-5), Medium (6-10), Low (11-20) ranking tiers

### Engine Comparison (RQ3)
- **Method**: Cross-engine statistical comparison
- **Tests**: Chi-square for categorical, Mann-Whitney U for continuous
- **Scope**: Feature distribution differences between Google and Bing

### Ranking Prediction (RQ4)
- **Method**: Ordinal Logistic Regression
- **Validation**: Proportional odds assumption testing
- **Features**: Random Forest importance ranking
- **Output**: Feature coefficients and significance levels

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Chrome/Chromium browser (for web scraping)
- Google Custom Search API key
- Bing Search API key (optional)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd serp-profiler-kit
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key" > .env
echo "CX=your_custom_search_engine_id" >> .env
```

### Key Dependencies

**Core Data Science:**
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.20.0` - Numerical computing
- `scikit-learn>=0.24.0` - Machine learning
- `scipy>=1.7.0` - Statistical analysis
- `statsmodels` - Advanced statistics

**Web Scraping & APIs:**
- `pyppeteer` - Headless browser automation
- `google-api-python-client` - Google Custom Search API
- `requests` - HTTP requests

**Natural Language Processing:**
- `sentence-transformers` - Semantic similarity
- `nltk` - Text processing
- `spacy` - Advanced NLP

**Visualization:**
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization

## 🚀 Usage

### Interactive Pipeline (Recommended)

Run the complete pipeline through the interactive menu:

```bash
python run.py
```

This launches a menu-driven interface with all pipeline stages:

```
📊 Data Collection:
1  - Website Acquisition (Google)
2  - Website Acquisition (Bing)
3  - Dump SERPs (Google & Bing)

🔄 Data Processing:
4  - Scraper
5  - Acceptance
6  - Feature Extraction
7  - Pagespeed
8  - Dump Pagespeed
9  - Combine
10 - Outliers
11 - Create Dataset
12 - Dataset Info

📈 Analysis:
13 - Analyze

📋 Generation:
14 - Plot Generation
15 - LaTeX Table Generation
```

### Individual Script Execution

Each module can be run independently with specific parameters:

```bash
# Collect Google SERPs
python src/data_collection/search_on_google.py \
    --input data/raw/keywords/keywords.csv \
    --output data/raw/search_results/google \
    --max_page 2

# Scrape web pages
python src/data_collection/scraper.py \
    --input data/processed/serps.csv \
    --output data/raw/pages/ \
    --screenshots data/raw/screenshots/

# Run analysis
python src/analysis/analyze.py \
    --input data/datasets/dataset.csv \
    --output data/analysis/
```

## 📈 Output Files

### Analysis Results (`data/analysis/`)
- `analysis_results.json` - Complete statistical analysis results
- `dataset_with_clusters.csv` - Dataset with cluster assignments

### Visualizations (`data/analysis/plots/`)
- `rq1_optimal_clusters_combined.png` - Cluster optimization plots
- `rq1_cluster_radar_normalized.png` - Cluster radar charts
- `rq1_cluster_pca_2d.png` - PCA visualization
- `rq2_profile_dist_combined_grouped_bar.png` - Feature distribution
- `rq3_profile_comparison.png` - Engine comparison
- `rq4_correlation_heatmaps_combined.png` - Correlation analysis
- `rq4_feature_importance_combined.png` - Feature importance

### LaTeX Tables (`data/analysis/tables/`)
- `table_rq1_cluster_summary.tex` - Cluster characteristics
- `table_rq2_feature_rank_tests.tex` - Feature importance tests
- `table_rq3_feature_comparison.tex` - Engine comparison
- `table_rq4_regression_*.tex` - Regression results

## 🔧 Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your_google_custom_search_api_key
CX=your_custom_search_engine_id
BING_API_KEY=your_bing_search_api_key  # Optional
```

### Key Parameters
- **Batch Size**: Number of concurrent scraping operations (default: 5)
- **Max Pages**: Maximum SERP pages per query (default: 2 for Google, 1 for Bing)
- **Headless Mode**: Browser visibility during scraping (default: True)
- **Overwrite**: Overwrite existing files (default: False)

## 🧪 Research Methodology

### Data Collection Protocol
1. **Keyword Selection**: Curated keyword list representing diverse search intents
2. **SERP Collection**: Top 20 organic results from each engine
3. **Web Page Archiving**: Full HTML content and visual capture
4. **Performance Measurement**: Lighthouse scores via PageSpeed Insights

### Quality Assurance
- **Exception Tracking**: Comprehensive error logging and retry mechanisms
- **Data Validation**: Multi-stage quality checks and outlier detection
- **Reproducibility**: Deterministic random seeds and version-controlled dependencies

### Statistical Rigor
- **Multiple Validation Methods**: Cross-validation and assumption testing
- **Non-Parametric Tests**: Appropriate for non-normal distributions
- **Effect Size Reporting**: Beyond p-values for practical significance
- **Multiple Comparison Correction**: Bonferroni correction where applicable

## 🔄 Extending the Framework

### Adding New Features
1. Modify `src/feature_extraction/extract_features.py`
2. Add feature computation function
3. Update feature lists in analysis modules
4. Regenerate analysis and outputs

### Adding New Search Engines
1. Create new script in `src/data_collection/`
2. Follow existing API patterns
3. Add to `run.py` menu system
4. Update analysis for multi-engine comparison

### Adding New Analyses
1. Extend `src/analysis/analyze.py`
2. Add new research question method
3. Create corresponding plot/table generators
4. Update documentation

## 📚 Academic Use

This framework is designed for academic research and includes:

- **Reproducible Analysis**: Complete pipeline with deterministic outputs
- **Publication-Ready Outputs**: LaTeX tables and high-resolution figures
- **Statistical Rigor**: Appropriate methods and validation techniques
- **Comprehensive Documentation**: Detailed methodology and implementation

## 🔮 Future Work

The following improvements are planned for upcoming releases:

### Code Quality & Maintenance
- **Remove Print Statements**: Replace all `print()` statements with proper logging throughout the codebase
- **Enhanced Logging**: Implement structured logging with configurable log levels
- **Code Documentation**: Add comprehensive docstrings to all functions and classes
- **Type Hints**: Add type annotations to improve code maintainability

### Performance & Scalability
- **Parallel Processing**: Implement multiprocessing for data collection and feature extraction
- **Memory Optimization**: Optimize memory usage for large datasets
- **Caching**: Add intelligent caching for API calls and expensive computations
- **Database Integration**: Add support for database storage instead of CSV files

### Feature Enhancements
- **Additional Search Engines**: Support for DuckDuckGo, Yandex, and other search engines
- **Advanced NLP Features**: Implement more sophisticated text analysis features
- **Real-time Monitoring**: Add dashboard for monitoring data collection progress
- **API Rate Limiting**: Implement intelligent rate limiting for API calls

### Analysis & Visualization
- **Interactive Plots**: Convert static plots to interactive visualizations using Plotly
- **Advanced Clustering**: Implement additional clustering algorithms (DBSCAN, hierarchical)
- **Feature Selection**: Add automated feature selection methods
- **Model Interpretability**: Add SHAP values and other interpretability tools

### User Experience
- **Web Interface**: Develop a web-based interface for non-technical users
- **Configuration GUI**: Create a graphical configuration tool
- **Progress Tracking**: Add real-time progress indicators for long-running operations
- **Error Recovery**: Implement automatic error recovery and retry mechanisms

### Testing & Validation
- **Unit Tests**: Add comprehensive unit tests for all modules
- **Integration Tests**: Implement end-to-end testing
- **Performance Benchmarks**: Add performance testing and benchmarking
- **Data Validation**: Enhance data quality checks and validation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Review the documentation in each module
- Check the analysis results for methodology details

---

**Note**: This framework requires appropriate API keys and follows ethical web scraping practices. Ensure compliance with search engine terms of service and website robots.txt files.