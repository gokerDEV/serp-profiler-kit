import os
import sys
import contextlib

# Add helper directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from helpers.report_keywords import analyze_keywords
    from helpers.report_serp import analyze_serp
    from helpers.report_index import analyze_index
except ImportError:
    try:
        from src.analysis.A.helpers.report_keywords import analyze_keywords
        from src.analysis.A.helpers.report_serp import analyze_serp
        from src.analysis.A.helpers.report_index import analyze_index
    except ImportError:
         print("Could not import helper scripts. Ensure you are running from project root.")
         sys.exit(1)

DATA_DIR = "data"
REPORT_DIR = "data/reports"
REPORT_FILE = os.path.join(REPORT_DIR, "index_stats.md")

def main():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    print(f"Generating report at {REPORT_FILE}...")

    with open(REPORT_FILE, 'w') as f:
        with contextlib.redirect_stdout(f):
            print("# Dataset Info Report\n")
            print(f"**Report Generated:** {pd.Timestamp.now()}\n")
            
            # 1. Keywords
            print("## 1. Keywords\n")
            analyze_keywords(os.path.join(DATA_DIR, "keywords.csv"))
            print("\n")
            
            # 2. SERP
            print("## 2. SERP Analysis\n")
            analyze_serp(os.path.join(DATA_DIR, "serp.csv"))
            print("\n")
            
            # 3. Index
            print("## 3. Core Index Analysis\n")
            analyze_index(os.path.join(DATA_DIR, "index.parquet"))
            print("\n")
            
            print("---")
            print("Report End")

    print("Done.")

import pandas as pd # Import pandas here for timestamp if needed, or import at top

if __name__ == "__main__":
    main()
