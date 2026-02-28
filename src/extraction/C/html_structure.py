import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import helpers from within module
from src.extraction.C.helpers.extract_json import run_extraction_batch
from src.extraction.C.helpers.merge_json import merge_json_files

def main():
    parser = argparse.ArgumentParser(description="Extraction Step C: HTML Structure (Orchestrator)")
    
    # Common Args
    parser.add_argument("--mode", choices=['extract', 'merge', 'all'], default='all', help="Operation mode")
    parser.add_argument("--index", help="Path to index parquet (Required for extract)")
    parser.add_argument("--scraping-root", help="Root directory for scraping artifacts (Required for extract)")
    parser.add_argument("--json-root", help="Root directory for intermediate JSONs (Required for extract & merge)")
    parser.add_argument("--out", help="Output path for final parquet (Required for merge)")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers (cpu count default)")
    
    args = parser.parse_args()
    
    # Validation
    if args.mode in ['extract', 'all']:
        if not args.index or not args.scraping_root or not args.json_root:
            parser.error("--mode extract requires --index, --scraping-root, and --json-root")
            
    if args.mode in ['merge', 'all']:
        if not args.json_root or not args.out:
            parser.error("--mode merge requires --json-root and --out")
            
    # Execution
    if args.mode in ['extract', 'all']:
        print("\n=== Phase 1: HTML Extraction to JSON ===", file=sys.stderr)
        run_extraction_batch(args.index, args.scraping_root, args.json_root, args.workers)
        
    if args.mode in ['merge', 'all']:
        print("\n=== Phase 2: Merging JSONs to Parquet ===", file=sys.stderr)
        merge_json_files(args.json_root, args.out, args.index, args.workers)
        
    print("\nStep C process complete.", file=sys.stderr)

if __name__ == "__main__":
    main()
