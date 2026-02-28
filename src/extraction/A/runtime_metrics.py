import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

import pandas as pd

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.helpers.progress import get_progress_bar
from src.helpers.paths import normalize_folder, resolve_path
from src.helpers.data_loader import load_valid_index


def process_record_task(task: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Worker-safe wrapper: takes (row_dict, scraping_root).
    """
    row, scraping_root = task

    record_id = str(row.get("record_id", "")).strip()
    folder = normalize_folder(row.get("folder", ""))
    file_name = str(row.get("file_name", "")).strip()
    

    result: Dict[str, Any] = {
        "record_id": record_id,
        "ttfb_ms": None,
        "dom_content_loaded_ms": None,
        "load_time_ms": None,
        "lcp_ms": None,
        "cls": None,
        "step_status": "ok",
        "step_reason": None,
    }


    if not folder or not file_name:
        result["step_status"] = "fail"
        result["step_reason"] = "missing_folder_or_file_name"
        return result

    # Robust path resolution
    json_path = resolve_path(scraping_root, folder, file_name, ".json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result["ttfb_ms"] = data.get("ttfb_ms")
        result["dom_content_loaded_ms"] = data.get("dom_content_loaded_ms")
        result["load_time_ms"] = data.get("load_time_ms")
        result["lcp_ms"] = data.get("lcp_ms")
        result["cls"] = data.get("cls")

    except FileNotFoundError:
        result["step_status"] = "miss"
        result["step_reason"] = "file_not_found"
    except json.JSONDecodeError:
        result["step_status"] = "fail"
        result["step_reason"] = "json_decode_error"
    except Exception as e:
        result["step_status"] = "fail"
        result["step_reason"] = f"exception:{type(e).__name__}"

    return result


def run_single_worker(records: List[Dict[str, Any]], scraping_root: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in get_progress_bar(records, desc="Extracting Metrics"):
        results.append(process_record_task((row, scraping_root)))
    return results


def run_multi_worker(
    records: List[Dict[str, Any]],
    scraping_root: str,
    workers: int,
    chunksize: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    tasks = [(row, scraping_root) for row in records]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_record_task, t) for t in tasks]

        for fut in get_progress_bar(
            as_completed(futures),
            desc=f"Extracting Metrics ({workers} workers)",
            total=len(futures),
        ):
            results.append(fut.result())

    return results


def generate_report(metrics_df: pd.DataFrame, output_path: str, total_input: int):
    """
    Generates a markdown report for Extraction A.
    """
    import datetime
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    status_counts = metrics_df['step_status'].value_counts()
    reason_counts = metrics_df['step_reason'].value_counts()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction A: Runtime Metrics Report\n")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction Step A: Runtime Metrics")
    parser.add_argument("--index", required=True, help="Path to input index.parquet")
    parser.add_argument("--scraping-root", required=True, help="Root directory for scraping artifacts")
    parser.add_argument("--out", required=True, help="Output path for runtime_metrics.parquet")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--chunksize", type=int, default=64, help="Reserved for future batching; keep default")
    args = parser.parse_args()



    required_cols = {"record_id", "folder", "file_name", "status"}
    df = load_valid_index(args.index, required_cols)

    records = df.to_dict("records")
    print(f"Processing {len(records)} records with workers={args.workers}...")

    if args.workers <= 1:
        results = run_single_worker(records, args.scraping_root)
    else:
        results = run_multi_worker(records, args.scraping_root, args.workers, args.chunksize)

    metrics_df = pd.DataFrame(results)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving features to {args.out}...")
    metrics_df.to_parquet(args.out, index=False)
    
    # Generate Report
    report_path = "data/reports/extraction_A.md"
    generate_report(metrics_df, report_path, len(df))
    print(f"Report saved to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
