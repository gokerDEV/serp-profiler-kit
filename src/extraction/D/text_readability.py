import pandas as pd
import json
import os
import argparse
import sys
from pathlib import Path
import textstat
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.helpers.progress import get_progress_bar
from src.helpers.paths import normalize_folder, resolve_path
from src.helpers.data_loader import load_valid_index

def process_record(payload):
    """
    Process a single record: read extracted JSON, compute readability metrics.
    Payload: (row, extraction_root)
    """
    row, extraction_root = payload
    
    record_id = row['record_id']
    folder = normalize_folder(row.get('folder')) 
    file_name = row.get('file_name')
    
    # Default result
    result = {
        'record_id': record_id,
        'char_count': None,
        'word_count': None,
        'sentence_count': None,
        'flesch_reading_ease': None,
        'flesch_kincaid_grade': None,
        'gunning_fog': None,
        'smog_index': None,
        'automated_readability_index': None,
        'step_status': 'ok',
        'step_reason': None
    }

    if not isinstance(folder, str) or not isinstance(file_name, str):
         result['step_status'] = 'skipped'
         result['step_reason'] = 'invalid_path_types'
         return result

    # Path to extracted JSON
    json_path = resolve_path(extraction_root, folder, file_name, ".json")
    
    try:
        if not os.path.exists(json_path):
             result['step_status'] = 'miss'
             result['step_reason'] = 'extraction_json_not_found'
             return result

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        content = data.get('content', '')
        
        if not content or len(content.strip()) == 0:
            result['step_status'] = 'soft_fail'
            result['step_reason'] = 'empty_content'
            return result

        # Compute basic metrics (safe)
        total_len = len(content)
        result['char_count'] = total_len
        
        # Fast noise check using regex (look for any non-whitespace chunk > 100 chars)
        # This is much faster and memory-efficient than content.split()
        import re
        is_noisy = False
        if re.search(r'[^\s]{101,}', content):
            is_noisy = True
            result['step_status'] = 'soft_fail'
            result['step_reason'] = 'technical_noise_detected'
            
        # Basic word count (fallback)
        # For word count, we still need a split, but we can do it on a truncated version if huge
        scan_content = content if total_len < 100000 else content[:100000]
        result['word_count'] = len(scan_content.split())
        
        # Only compute complex metrics if not noisy and content is within reasonable size
        if not is_noisy and total_len > 0:
            try:
                # Truncate extremely long content for textstat (beyond 100k chars is plenty for readability)
                # This prevents hangs in underlying regex engines of textstat
                calc_content = scan_content
                
                result['word_count'] = textstat.lexicon_count(calc_content, removepunct=True)
                result['sentence_count'] = textstat.sentence_count(calc_content)
                
                flesch_score = textstat.flesch_reading_ease(calc_content)
                result['flesch_reading_ease'] = flesch_score
                
                if flesch_score < -500 or flesch_score > 200:
                     result['step_status'] = 'soft_fail'
                     result['step_reason'] = f'implausible_score_{flesch_score}'
                     
                result['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(calc_content)
                result['gunning_fog'] = textstat.gunning_fog(calc_content)
                result['smog_index'] = textstat.smog_index(calc_content)
                result['automated_readability_index'] = textstat.automated_readability_index(calc_content)
            except Exception as e:
                if result['step_status'] == 'ok':
                    result['step_status'] = 'fail'
                    result['step_reason'] = f"metrics_error: {str(e)[:100]}"
        
    except FileNotFoundError:
        result['step_status'] = 'miss'
        result['step_reason'] = 'extraction_json_not_found'
    except Exception as e:
        result['step_status'] = 'fail'
        result['step_reason'] = str(e)[:300]
        
    return result

def generate_report(metrics_df: pd.DataFrame, output_path: str, total_input: int):
    """
    Generates a markdown report for Extraction D.
    """
    import datetime
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    status_counts = metrics_df['step_status'].value_counts()
    reason_counts = metrics_df['step_reason'].value_counts()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction D: Text Readability Report\n")
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
    parser = argparse.ArgumentParser(description="Extraction Step D (Stage 2): Readability Metrics")
    parser.add_argument("--index", required=True, help="Path to input index.parquet")
    parser.add_argument("--extraction-root", required=True, help="Root directory containing extracted JSONs (Stage 1 output)")
    parser.add_argument("--out", required=True, help="Output path for text_readability.parquet")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.extraction_root):
        print(f"Extraction root not found: {args.extraction_root}")
        sys.exit(1)

    df = load_valid_index(args.index)
        
    print(f"Processing {len(df)} records using {args.workers} workers...")
    
    records = df.to_dict('records')
    
    # Create payload generator
    payloads = ((r, args.extraction_root) for r in records)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        chunk_size = 50 
        futures = executor.map(process_record, payloads, chunksize=chunk_size)
        
        # Collect results with progress bar
        # Note: futures from map is an iterator, get_progress_bar consumes it
        results = list(get_progress_bar(futures, total=len(records), desc="Calculating Readability"))
        
    # Create DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
         
    print(f"Saving features to {args.out}...")
    try:
        metrics_df.to_parquet(args.out, index=False)
    except Exception as e:
         print(f"Error saving parquet: {e}")
         sys.exit(1)
         
    # Generate Report
    report_path = "data/reports/extraction_D.md"
    generate_report(metrics_df, report_path, len(df))
    print(f"Report saved to {report_path}")

    print("Done.")

if __name__ == "__main__":
    main()
