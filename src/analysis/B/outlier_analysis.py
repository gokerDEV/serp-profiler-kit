import pandas as pd
import argparse
import sys
import contextlib
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from helpers
from src.analysis.B.helpers.data_loader import load_data, merge_features, FEATURE_PATHS
from src.analysis.B.helpers.stats import analyze_health, calculate_outlier_flags, report_distribution_comparison
from src.analysis.B.helpers.investigate_readability_outlier import investigate_outlier as investigate_readability
from src.analysis.B.helpers.investigate_similarity_outliers import investigate_similarity_outliers as investigate_similarity
from src.helpers.schema_validator import validate_columns

def main():
    parser = argparse.ArgumentParser(description="Analysis Step B: Outlier Analysis & Tagging")
    parser.add_argument("--index", default="data/index.parquet", help="Path to index parquet")
    parser.add_argument("--out", default=None, help="Output path for report file (optional)")
    parser.add_argument("--out-flags", default="data/outliers.parquet", help="Output path for outlier flags parquet")
    parser.add_argument("--iqr-threshold", type=float, default=3.0, help="IQR threshold for outlier detection (default: 3.0)")
    parser.add_argument("--investigate", action="store_true", help="Run deep-dive investigation on outliers")
    args = parser.parse_args()
    
    # 1. Load Data
    df = load_data(args.index)
    
    # 2. Merge Features
    full_df = merge_features(df, FEATURE_PATHS)
    
    # Validation
    print("Validating dataset schema...", file=sys.stderr)
    outlier_cols = ['is_outlier_technical', 'is_outlier_readability', 'is_outlier_similarity', 'is_outlier_accessibility']
    try:
        validate_columns(full_df, strict=False, exclude_columns=outlier_cols)
    except Exception as e:
        print(f"Schema Warning: {e}", file=sys.stderr)
    
    # 3. Define Metric Groups for Outlier Detection
    metric_groups = {
        'technical': ['html_size_bytes', 'ttfb_ms', 'dom_content_loaded_ms', 'lcp_ms', 'cls'],
        'readability': ['char_count', 'word_count', 'flesch_reading_ease', 'automated_readability_index'],
        'similarity': ['sim_title', 'sim_description', 'sim_h1', 'sim_content'],
        'accessibility': ['contrast_violation_count', 'axe_violations_total']
    }
    
    # 4. Calculate Outlier Flags
    print(f"Calculating outlier flags (Threshold={args.iqr_threshold})...", file=sys.stderr)
    flags_df = calculate_outlier_flags(full_df, metric_groups, threshold=args.iqr_threshold)

    # Add record_id from index for parquet joinability (index is record_id)
    flags_out = flags_df.reset_index()
    
    # Ensure output dir for flags
    flags_dir = os.path.dirname(args.out_flags)
    if flags_dir: os.makedirs(flags_dir, exist_ok=True)
    
    print(f"Saving outlier flags to {args.out_flags}...", file=sys.stderr)
    flags_out.to_parquet(args.out_flags, index=False)
    
    # 5. Generate Report
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        f = open(args.out, 'w', encoding='utf-8')
        cm = contextlib.redirect_stdout(f)
    else:
        f = None
        cm = contextlib.nullcontext()

    with cm:
        print(f"# Outlier & Consistency Report")
        print(f"\n**Generated:** {pd.Timestamp.now()}")
        
        analyze_health(full_df, FEATURE_PATHS.keys())
        
        # New comparative report
        report_distribution_comparison(full_df, flags_df, metric_groups)
        
        if args.investigate:
            print("\n## Deep Dive Investigation")
            print("Running detailed outlier checks...\n")
            print("### Readability Outliers")
            investigate_readability()
            print("\n### Similarity Outliers")
            investigate_similarity()
        
    if f:
        f.close()
        print(f"Report saved to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
