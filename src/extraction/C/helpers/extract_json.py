import pandas as pd
import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# Add project root to path
# This is necessary because helper might be imported or run directly
project_root = str(Path(__file__).resolve().parents[4]) # Adjusted for helpers depth
if project_root not in sys.path:
    # Just in case
    pass
# Actually, imports from src.helpers require project root.
if project_root not in sys.path:
     sys.path.append(project_root)

# Standard imports for project structure
from src.helpers.progress import get_progress_bar
from src.extraction.C.helpers.parser import extract_content
from src.helpers.paths import normalize_folder
from src.helpers.data_loader import load_valid_index

def process_record_extract(row, scraping_root, out_root, max_retries=2):
    """
    Process a single record, extract HTML features, and save as JSON.
    Wrapped for multiprocessing.
    """
    record_id = row['record_id']
    folder = normalize_folder(row.get('folder'))
    file_name = row.get('file_name')
    
    # Validate types (pandas might import as NaN/float if missing)
    if not isinstance(folder, str) or not isinstance(file_name, str):
        return {'record_id': record_id, 'step_status': 'skipped', 'step_reason': f"Invalid path types: folder={type(folder)}, file={type(file_name)}"}
    
    # Mirror folder structure
    # Use explicit join to avoid any ambiguity
    out_dir = os.path.join(out_root, folder)
    out_file = os.path.join(out_dir, file_name + '.json')
    
    # Ensure dir exists
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass # Parallel race condition is fine
        
    # Extract using parser helper
    # We need to construct full path for parser. 
    html_path = os.path.join(scraping_root, folder, file_name)
    
    result = extract_content(html_path)
    
    # Add metadata for traceability
    result['record_id'] = record_id
    result['source_file'] = html_path
    result['url'] = row.get('url')
    
    # Save JSON
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Check internal status from parser to report correctly
        if result.get('status') == 'error':
            return {
                'record_id': record_id, 
                'step_status': 'fail' if result.get('error') != 'file_not_found' else 'miss',
                'step_reason': result.get('status_reason') or result.get('error')
            }
            
        return {'record_id': record_id, 'step_status': 'ok', 'step_reason': None}
    except Exception as e:
        return {'record_id': record_id, 'step_status': 'fail', 'step_reason': str(e)}

def generate_report(metrics_df: pd.DataFrame, output_path: str, total_input: int):
    """
    Generates a markdown report for Extraction C.
    """
    import datetime
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    status_counts = metrics_df['step_status'].value_counts()
    reason_counts = metrics_df['step_reason'].value_counts()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction C: HTML Structure Report\n")
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

def run_extraction_batch(index_path, scraping_root, out_root, max_workers=None):
    if not max_workers:
        max_workers = os.cpu_count() or 4
        
    df = load_valid_index(index_path)

    print(f"Processing {len(df)} records using {max_workers} workers...", file=sys.stderr)
    records = df.to_dict('records')
    
    # Create partial function for fixed args
    process_func = partial(process_record_extract, scraping_root=scraping_root, out_root=out_root)
    
    success_count = 0
    fail_count = 0
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Smaller chunk_size for smoother progress updates. 
        # Large chunks (e.g. 4000+) cause the UI to freeze for minutes.
        chunk_size = 50 
        
        # Start tasks
        # Use map for order preservation (though not critical here) and easier iteration
        futures = executor.map(process_func, records, chunksize=chunk_size)
        
        # Use progress bar wrapper
        results_iter = get_progress_bar(futures, total=len(records), desc="Extracting HTML to JSON")
        
        for res in results_iter:
            all_results.append(res)
            if res['step_status'] == 'ok':
                success_count += 1
            else:
                fail_count += 1
                
    print(f"Extraction complete. Success: {success_count}, Fail: {fail_count}", file=sys.stderr)
    
    # Generate Report
    metrics_df = pd.DataFrame(all_results)
    report_path = "data/reports/extraction_C.md"
    generate_report(metrics_df, report_path, len(df))
    print(f"Report saved to {report_path}", file=sys.stderr)
