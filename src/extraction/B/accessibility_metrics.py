import pandas as pd
import numpy as np
import json
import os
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.helpers.progress import get_progress_bar
from src.helpers.paths import normalize_folder, resolve_path
from src.helpers.data_loader import load_valid_index

def process_record(row, scraping_root):
    """
    Process a single record to extract accessibility metrics.
    Must be top-level for multiprocessing pickling.
    """
    record_id = row["record_id"]
    folder = normalize_folder(row["folder"])
    file_name = row["file_name"]

    result = {
        "record_id": record_id,
        "contrast_violation_count": None,
        "contrast_pass_count": None,
        "min_contrast_ratio": None,
        "axe_violations_total": None,
        "axe_passes_total": None,
        "axe_incomplete_total": None,
        "axe_inapplicable_total": None,
        "axe_impact_critical": None,
        "axe_impact_serious": None,
        "axe_impact_moderate": None,
        "axe_impact_minor": None,
        "step_status": "ok",
        "step_reason": None,
    }


    json_path = resolve_path(scraping_root, folder, file_name, ".json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result["contrast_violation_count"] = data.get("contrast_violation_count")
        result["contrast_pass_count"] = data.get("contrast_pass_count")
        result["min_contrast_ratio"] = data.get("min_contrast_ratio")

        result["axe_violations_total"] = data.get("axe_violations_total")
        result["axe_passes_total"] = data.get("axe_passes_total")
        result["axe_incomplete_total"] = data.get("axe_incomplete_total")
        result["axe_inapplicable_total"] = data.get("axe_inapplicable_total")

        axe_impact = data.get("axe_impact_counts", {})
        if isinstance(axe_impact, dict):
            result["axe_impact_critical"] = axe_impact.get("critical")
            result["axe_impact_serious"] = axe_impact.get("serious")
            result["axe_impact_moderate"] = axe_impact.get("moderate")
            result["axe_impact_minor"] = axe_impact.get("minor")

    except FileNotFoundError:
        result["step_status"] = "miss"
        result["step_reason"] = "file_not_found"
    except json.JSONDecodeError:
        result["step_status"] = "fail"
        result["step_reason"] = "json_decode_error"
    except Exception as e:
        result["step_status"] = "fail"
        result["step_reason"] = str(e)[:300]

    return result


def process_record_safe(row, scraping_root):
    """
    Worker-safe wrapper: never crash pool on unexpected row-level errors.
    """
    try:
        return process_record(row, scraping_root)
    except Exception as e:
        return {
            "record_id": row.get("record_id"),
            "contrast_violation_count": None,
            "contrast_pass_count": None,
            "min_contrast_ratio": None,
            "axe_violations_total": None,
            "axe_passes_total": None,
            "axe_incomplete_total": None,
            "axe_inapplicable_total": None,
            "axe_impact_critical": None,
            "axe_impact_serious": None,
            "axe_impact_moderate": None,
            "axe_impact_minor": None,
            "step_status": "fail",
            "step_reason": f"worker_exception:{str(e)[:250]}",
        }


def run_parallel(records, scraping_root, workers):
    """
    Parallel processing with progress bar over completed futures.
    """
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_record_safe, row, scraping_root)
            for row in records
        ]

        for fut in get_progress_bar(
            as_completed(futures),
            desc=f"Extracting Accessibility Metrics (workers={workers})",
            total=len(futures),
        ):
            results.append(fut.result())
    return results


def run_sequential(records, scraping_root):
    """
    Fallback mode (workers=1), same behavior as before.
    """
    results = []
    for row in get_progress_bar(records, desc="Extracting Accessibility Metrics"):
        results.append(process_record_safe(row, scraping_root))
    return results


def generate_report(metrics_df: pd.DataFrame, output_path: str, total_input: int):
    """
    Generates a markdown report for Extraction B.
    """
    import datetime
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    status_counts = metrics_df['step_status'].value_counts()
    reason_counts = metrics_df['step_reason'].value_counts()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction B: Accessibility Metrics Report\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Summary\n")
        f.write(f"- **Total Input Records:** {total_input}\n")
        f.write(f"- **Processed Output:** {len(metrics_df)}\n")
        if len(metrics_df) > 0:
            success_rate = status_counts.get('ok', 0) / len(metrics_df) * 100
            duplicate_count = metrics_df.duplicated(subset=['record_id']).sum()
        else:
            success_rate = 0
            duplicate_count = 0
            
        f.write(f"- **Success Rate (step_status='ok'):** {success_rate:.2f}%\n")
        f.write(f"- **Duplicate Record IDs:** {duplicate_count}\n\n")
        
        f.write(f"## Status Distribution\n")
        f.write(status_counts.to_markdown(headers=["Count"]))
        f.write("\n\n")
        
        f.write(f"## Failure Reasons (Top 20)\n")
        if not reason_counts.empty:
            f.write(reason_counts.head(20).to_markdown(headers=["Count"]))
        else:
            f.write("No failures recorded.")
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Extraction Step B: Accessibility Metrics")
    parser.add_argument("--index", required=True, help="Path to input index.parquet")
    parser.add_argument("--scraping-root", required=True, help="Root directory for scraping artifacts")
    parser.add_argument("--out", required=True, help="Output path for accessibility_metrics.parquet")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (multiprocessing.cpu_count() - 1)),
        help="Number of worker processes (default: cpu_count - 1)",
    )

    args = parser.parse_args()

    # Use load_valid_index helper
    required_cols = {"record_id", "folder", "file_name", "status"}
    df = load_valid_index(args.index, required_cols)

    print(f"Loaded {len(df)} records.")
    records = df.to_dict("records")

    workers = max(1, int(args.workers))
    if workers == 1:
        results = run_sequential(records, args.scraping_root)
    else:
        results = run_parallel(records, args.scraping_root, workers)

    metrics_df = pd.DataFrame(results)

    # --- Feature Engineering (RQ5 Calculation) ---
    print("Calculating derived accessibility scores (RQ5)...", file=sys.stderr)
    
    # 1. axe_score
    if 'axe_passes_total' in metrics_df.columns and 'axe_violations_total' in metrics_df.columns:
        axe_denom = metrics_df['axe_passes_total'] + metrics_df['axe_violations_total']
        metrics_df['axe_score'] = metrics_df['axe_passes_total'] / axe_denom
        # Handle 0/0 case (if no rules checks? unlikely but safe to be NaN or 1? Standard practice: NaN if no checks)
        metrics_df.loc[axe_denom == 0, 'axe_score'] = np.nan
    else:
        print("Warning: Missing columns for axe_score calculation", file=sys.stderr)
        metrics_df['axe_score'] = np.nan

    # 2. contrast_score
    if 'contrast_pass_count' in metrics_df.columns and 'contrast_violation_count' in metrics_df.columns:
        contrast_denom = metrics_df['contrast_pass_count'] + metrics_df['contrast_violation_count']
        metrics_df['contrast_score'] = metrics_df['contrast_pass_count'] / contrast_denom
        metrics_df.loc[contrast_denom == 0, 'contrast_score'] = np.nan
    else:
        print("Warning: Missing columns for contrast_score calculation", file=sys.stderr)
        metrics_df['contrast_score'] = np.nan

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving features to {args.out}...")
    metrics_df.to_parquet(args.out, index=False)

    # Generate Report
    report_path = "data/reports/extraction_B.md"
    generate_report(metrics_df, report_path, len(df))
    print(f"Report saved to {report_path}")
    
    # quick status summary
    status_counts = metrics_df["step_status"].value_counts(dropna=False).to_dict()
    print("Step status summary:", status_counts)
    print("Done.")


if __name__ == "__main__":
    main()
