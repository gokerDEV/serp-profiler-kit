import pandas as pd
import os
import argparse
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.helpers.progress import get_progress_bar
from src.helpers.paths import normalize_folder, resolve_path
from src.helpers.data_loader import load_valid_index

MODEL_NAME = 'all-MiniLM-L6-v2'

def safe_str(val):
    """Safely convert potential NaN/None to empty string."""
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

def load_record_data(row, extraction_root):
    """
    Read JSON and extract relevant text fields.
    Executed in parallel workers.
    """
    record_id = row['record_id']
    status = row.get('status')
    
    res = {
        'record_id': record_id,
        'search_term': row.get('search_term', ''),
        'title': '',
        'description': '',
        'content': '',
        'h1_text': '',
        'step_status': 'ok',
        'step_reason': None
    }


    folder = normalize_folder(row.get('folder'))
    file_name = row.get('file_name')
    
    if not isinstance(folder, str) or not isinstance(file_name, str):
         res['step_status'] = 'skipped'
         res['step_reason'] = 'invalid_path'
         return res
         
    json_path = resolve_path(extraction_root, folder, file_name, ".json")
    
    try:
        if not os.path.exists(json_path):
             res['step_status'] = 'miss' 
             res['step_reason'] = 'file_missing'
             return res

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Try internal title from JSON, fallback to index title/snippet if empty
        res['title'] = safe_str(data.get('title')) or safe_str(row.get('title'))
        res['description'] = safe_str(data.get('description')) or safe_str(row.get('snippet'))
        res['content'] = safe_str(data.get('content'))
        
        # H1 Extraction (Handle new dict structure or old list structure)
        h_tags = data.get('h_tags', {})
        h1_text = ""
        
        if isinstance(h_tags, dict):
            h1_list = h_tags.get('h1', [])
            if isinstance(h1_list, list):
                 h1_text = " ".join(h1_list)
       
             
        res['h1_text'] = h1_text
        
        # Validation: If all text fields are empty, mark as skip/fail
        if not any([res['title'], res['description'], res['content'], res['h1_text']]):
             res['step_status'] = 'fail'
             res['step_reason'] = 'empty_extracted_text'
        
    except Exception as e:
        res['step_status'] = 'fail'
        res['step_reason'] = str(e)
        
    return res

def batch_compute_similarity(model, records_data):
    """
    Compute similarity for a batch of records using vectorized operations.
    """
    # Prepare batch lists
    queries = []
    titles = []
    descriptions = []
    contents = []
    h1s = []
    valid_indices = []

    # Filter only 'ok' records for processing
    for i, rec in enumerate(records_data):
        if rec['step_status'] == 'ok' and rec['search_term']:
            queries.append(rec['search_term'])
            
            # Truncate content safely
            t = safe_str(rec.get('title'))[:1000]
            d = safe_str(rec.get('description'))[:1000]
            c = safe_str(rec.get('content'))[:5000]
            h = safe_str(rec.get('h1_text'))[:1000]
            
            titles.append(t)
            descriptions.append(d)
            contents.append(c)
            h1s.append(h)
            valid_indices.append(i)
    
    if not queries:
        return records_data

    # Encode queries (with internal caching if possible, but batch encode is fast enough)
    q_embs = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    
    # Encode document fields
    # Note: SentenceTransformer encodes empty string "" into a vector. 
    # We will compute cosine similarity for everything to keep tensor alignment,
    # but we will assign None later if original text was empty.
    t_embs = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
    d_embs = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False)
    c_embs = model.encode(contents, convert_to_tensor=True, show_progress_bar=False)
    h_embs = model.encode(h1s, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute Cosine Similarity (Pairwise)
    # torch is imported at top level now
    
    # Calculate pairwise cosine similarity
    sim_title = torch.nn.functional.cosine_similarity(q_embs, t_embs).cpu().numpy()
    sim_desc = torch.nn.functional.cosine_similarity(q_embs, d_embs).cpu().numpy()
    sim_content = torch.nn.functional.cosine_similarity(q_embs, c_embs).cpu().numpy()
    sim_h1 = torch.nn.functional.cosine_similarity(q_embs, h_embs).cpu().numpy()
    
    # Assign back to records
    for idx, list_idx in enumerate(valid_indices):
        rec = records_data[list_idx]
        
        # Only assign score if text was present
        rec['sim_title'] = float(sim_title[idx]) if rec['title'] else None
        rec['sim_description'] = float(sim_desc[idx]) if rec['description'] else None
        rec['sim_content'] = float(sim_content[idx]) if rec['content'] else None
        rec['sim_h1'] = float(sim_h1[idx]) if rec['h1_text'] else None
        
    return records_data

def generate_report(metrics_df: pd.DataFrame, output_path: str, total_input: int):
    """
    Generates a markdown report for Extraction E.
    """
    import datetime
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    status_counts = metrics_df['step_status'].value_counts()
    reason_counts = metrics_df['step_reason'].value_counts()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction E: Semantic Similarity Report\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Summary (Current Run)\n")
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
    parser = argparse.ArgumentParser(description="Step E: Semantic Similarity Extraction (Optimized)")
    parser.add_argument("--index", default="data/index.parquet", help="Path to index parquet")
    parser.add_argument("--extraction-root", default="/Volumes/KIOXIA/extraction/", help="Root directory for extracted content")
    parser.add_argument("--out", default="data/features/E/semantic_similarity.parquet", help="Output parquet path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model inference")
    parser.add_argument("--workers", type=int, default=8, help="Workers for JSON loading")
    parser.add_argument("--device", default=None, help="Device to use (cpu, cuda, mps). If None, auto-detect.")
    
    args = parser.parse_args()
    
    # 1. Load Index
    df = load_valid_index(args.index)
        
    print(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    try:
        # Determine device
        if args.device:
            device = args.device
        else:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            
        print(f"Using device: {device}")
        
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Check for existing features (Incremental Mode)
    existing_df = None
    existing_ids = set()
    if os.path.exists(args.out):
        print(f"Features file found: {args.out}. Running in incremental mode.")
        try:
            existing_df = pd.read_parquet(args.out)
            if 'record_id' in existing_df.columns:
                existing_ids = set(existing_df['record_id'].tolist())
                print(f"Found {len(existing_ids)} existing records.")
        except Exception as e:
            print(f"Error reading existing file: {e}. Overwriting.")
            existing_df = None
            
    records = df.to_dict('records')
    
    # Filter: New records OR records that have NaN similarity scores in existing parquet
    if existing_df is not None:
        # Identify records that are "incomplete" (missing similarity scores)
        # but have step_status 'ok' (we want to re-try them with fallbacks)
        incomplete_mask = (
            (existing_df['step_status'] == 'ok') & 
            (existing_df['sim_title'].isna())
        )
        incomplete_ids = set(existing_df[incomplete_mask]['record_id'].tolist())
        
        # New records = not in existing IDs
        new_records = [r for r in records if r.get('record_id') and r.get('record_id') not in existing_ids]
        
        # Redo records = in incomplete_ids
        redo_records = [r for r in records if r.get('record_id') in incomplete_ids]
        
        to_process = new_records + redo_records
        
        # Remove redone records from existing_df so we don't have duplicates after concat
        if redo_records:
            print(f"Re-processing {len(redo_records)} incomplete records with fallback logic...")
            existing_df = existing_df[~existing_df['record_id'].isin(incomplete_ids)]
    else:
        to_process = records

    new_count = len(to_process)
    
    if new_count == 0:
        print("No new or incomplete records to process.")
        return

    print(f"Processing {new_count} records (Batch Size: {args.batch_size})...")
    
    final_results = []
    
    # Create chunks
    chunks = [to_process[i:i + args.batch_size] for i in range(0, new_count, args.batch_size)]
    
    load_func = partial(load_record_data, extraction_root=args.extraction_root)
    
    if chunks:
        # Use ProcessPoolExecutor for JSON loading
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Use get_progress_bar wrapper instead of raw tqdm
            for chunk in get_progress_bar(chunks, desc="Processing Batches", total=len(chunks)):
                # 1. Parallel Load
                loaded_data = list(executor.map(load_func, chunk))
                
                # 2. Compute Similarity (Main Thread - GPU/MPS/CPU optimized)
                processed_data = batch_compute_similarity(model, loaded_data)
                
                # 3. Collect
                final_results.extend(processed_data)
            
    # 3. Save Output
    print("Creating DataFrame...")
    metrics_df = pd.DataFrame(final_results)
    
    # Generate Report (on all processed records, including failures)
    report_path = "data/reports/extraction_E.md"
    generate_report(metrics_df, report_path, new_count)
    print(f"Report saved to {report_path}")
    
    cols_to_keep = ['record_id', 'title', 'description', 'sim_title', 'sim_description', 'sim_h1', 'sim_content', 'model_name', 'step_status', 'step_reason']
    
    # Add model name const
    metrics_df['model_name'] = MODEL_NAME
    
    # Keep all records regardless of status
    for col in cols_to_keep:
        if col not in metrics_df.columns:
             metrics_df[col] = None
    out_df = metrics_df[cols_to_keep]
        
    print(f"Processed features: {len(out_df)}")
    
    # Concatenate with existing
    if existing_df is not None:
        print("Merging with existing features...")
        final_df = pd.concat([existing_df, out_df], ignore_index=True)
    else:
        final_df = out_df
    
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Saving features to {args.out} (Total: {len(final_df)})...")
    final_df.to_parquet(args.out, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
