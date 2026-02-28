import sys
import subprocess
import os
import glob
from pathlib import Path

# Detect Virtual Environment
IN_VENV = sys.prefix != sys.base_prefix
if not IN_VENV:
    print("WARNING: Not running inside a virtual environment (.venv).")
    print("Please activate it: 'source .venv/bin/activate'")
    # We continue, but user is warned.

def get_latest_dataset():
    """Find the latest dataset parquet file in data/"""
    files = glob.glob("data/dataset-*.parquet")
    if not files:
        return None
    return max(files, key=os.path.getctime)

SCRIPTS = [
    # (Display Name, Script Path, Arguments (list or callable returning list))
    ("Collection A: News Seeds (Guardian API)", "src/collection/A/collect_news.py", [
        "--out", "data/keywords.csv",
        "--months", "4",
        "--per-month", "100"
    ]),
    ("Collection B: SERP Mojeek", "src/collection/B/collect_serp.py", [
        "--engine", "mojeek",
        "--input", "data/keywords.csv",
        "--output-json-dir", "data/raw/serp_results"
    ]),
    ("Collection B: SERP Brave", "src/collection/B/collect_serp.py", [
        "--engine", "brave",
        "--input", "data/keywords.csv",
        "--output-json-dir", "data/raw/serp_results"
    ]),
    ("Collection B: SERP Google", "src/collection/B/collect_serp.py", [
        "--engine", "google",
        "--input", "data/keywords.csv",
        "--output-json-dir", "data/raw/serp_results"
    ]),
    ("Collection B: SERP Merge to CSV", "src/collection/B/collect_serp.py", [
        "--merge",
        "--output-json-dir", "data/raw/serp_results",
        "--output-csv", "data/serp.csv"
    ]),
    ("Collection C: Scrape Sources (Bot)", "src/collection/C/collect_sources.py", [
        "--serp", "data/serp.csv",
        "--out-root", "/Volumes/KIOXIA/scraping/",
        "--scraped-csv", "data/scraped.csv",
        "--bot-cmd", "python src/collection/C/helpers/bot.py"
    ]),
    ("Collection C: Scraping Audit/Summary", "src/collection/C/collect_sources.py", [
        "--summary",
        "--scraped-csv", "data/scraped.csv"
    ]),
    ("Collection Z: Build Index", "src/collection/Z/build.py", [
        "--serp", "data/serp.csv",
        "--scraping-root", "/Volumes/KIOXIA/scraping/",
        "--out", "data/index.parquet"
    ]),
    ("Analysis A: Index Stats", "src/analysis/A/index_stats.py", lambda: (
        ["--dataset", "data/index.parquet"]
    )),
    ("Extraction A: Runtime Metrics", "src/extraction/A/runtime_metrics.py", [
        "--index", "data/index.parquet",
        "--scraping-root", "/Volumes/KIOXIA/scraping/",
        "--out", "data/features/A/runtime_metrics.parquet",
        "--workers", "8"
    ]),
    ("Extraction B: Accessibility Metrics", "src/extraction/B/accessibility_metrics.py", [
        "--index", "data/index.parquet",
        "--scraping-root", "/Volumes/KIOXIA/scraping/",
        "--out", "data/features/B/accessibility_metrics.parquet",
        "--workers", "8"
    ]),
    ("Extraction C: HTML Structure", "src/extraction/C/html_structure.py", [
        "--mode", "all",
        "--index", "data/index.parquet",
        "--scraping-root", "/Volumes/KIOXIA/scraping/",
        "--json-root", "/Volumes/KIOXIA/extraction/",
        "--out", "data/features/C/html_structure.parquet",
        "--workers", "8"
    ]),
    ("Extraction D: Text Readability", "src/extraction/D/text_readability.py", [
        "--index", "data/index.parquet",
        "--extraction-root", "/Volumes/KIOXIA/extraction/",
        "--out", "data/features/D/text_readability.parquet",
        "--workers", "8"
    ]),
    ("Extraction E: Semantic Similarity", "src/extraction/E/semantic_similarity.py", [
        "--index", "data/index.parquet",
        "--extraction-root", "/Volumes/KIOXIA/extraction/",
        "--out", "data/features/E/semantic_similarity.parquet",
        "--workers", "8"
    ]),
    ("Analysis B: Outlier Detection", "src/analysis/B/outlier_analysis.py", [
        "--index", "data/index.parquet",
        "--out-flags", "data/outliers.parquet",
        "--out", "data/reports/outlier_report.md",
        "--iqr-threshold", "3.0"
    ]),
    ("Extraction Z: Merge Features", "src/extraction/Z/merge.py", [
        "--index", "data/index.parquet",
        "--out-dir", "data",
        "--feature-base-dir", "data/features",
        "--outliers-path", "data/outliers.parquet"
    ]),
    ("Analysis C: RQ1 Rank Concentration", "src/analysis/C/rq1_concentration.py", lambda: (
        ["--dataset", get_latest_dataset(), "--out", "data/analysis/C/rq1_concentration.parquet", "--report", "data/analysis/C/rq1_concentration.md"] if get_latest_dataset() else None
    )),
    ("Analysis D: RQ2-4 Confirmatory Models", "src/analysis/D/confirmatory_models.py", lambda: (
        ["--dataset", get_latest_dataset(), "--out-dir", "data/analysis/D"] if get_latest_dataset() else None
    )),
    ("Analysis E: RQ5 Engine Heterogeneity (R)", "src/analysis/E/engine_heterogeneity.py", lambda: (
        ["--dataset", get_latest_dataset(), "--out-dir", "data/analysis/E"] if get_latest_dataset() else None
    )),
    ("Analysis F: RQ6 Query Difficulty (R)", "src/analysis/F/query_difficulty.py", lambda: (
        ["--dataset", get_latest_dataset(), "--out-dir", "data/analysis/F"] if get_latest_dataset() else None
    )),
    ("Analysis G: RQ7 Robustness (R)", "src/analysis/G/robustness.py", lambda: (
        ["--dataset", get_latest_dataset(), "--out-dir", "data/analysis/G"] if get_latest_dataset() else None
    )),
    ("Analysis H: RQ8 Ablation (R/Python)", "src/analysis/H/ablation.py", lambda: (
        ["--dataset", get_latest_dataset(), "--out-dir", "data/analysis/H"] if get_latest_dataset() else None
    )),
    ("Generator A: LaTeX Tables", "src/generators/A/tables.py", [
        "--out-dir", "data/reports/tables"
    ]),
    ("Generator B: Figures (Plots)", "src/generators/B/figures.py", [
        "--out-dir", "data/reports/figures"
    ])
]

def check_r_availability():
    """Check if Rscript is available."""
    try:
        subprocess.run(["Rscript", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("\n--- SERP Profiler Pipeline Runner ---")
    print(f"Python: {sys.executable}")
    
    if check_r_availability():
        print("R: Available")
    else:
        print("WARNING: R (Rscript) not found. Analysis E-H will fail.")
        
    print("-------------------------------------")
    
    for i, (desc, path, _) in enumerate(SCRIPTS, 1):
        print(f"[{i}] {desc}")
        
    choice = input("\nEnter script number to run: ").strip()
    
    if not choice.isdigit():
        print("Invalid input. Please enter a number.")
        return
        
    idx = int(choice) - 1
    if 0 <= idx < len(SCRIPTS):
        desc, path, args_def = SCRIPTS[idx]
        
        # Resolve arguments
        if callable(args_def):
            args = args_def()
            if args is None:
                print(f"Error: Could not resolve arguments for {desc} (e.g. Dataset not found).")
                return
        else:
            args = args_def
            
        print(f"\nRunning: {desc}")
        print(f"Script: {path}")
        print(f"Args: {' '.join(args)}")
        
        cmd = [sys.executable, path] + args
        
        try:
            # Run and stream output
            subprocess.run(cmd, check=True)
            
            print("\n✅ Execution Successful.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Execution Failed (Exit Code: {e.returncode})")
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    main()
