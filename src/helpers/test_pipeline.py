import pandas as pd
import os
import subprocess
import shutil
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    # Configuration
    INDEX_PATH = "data/index.parquet"
    TEST_INDEX_PATH = "data/index_test.parquet"
    SCRAPING_ROOT = "/Volumes/KIOXIA/scraping/"
    EXTRACTION_ROOT = "/Volumes/KIOXIA/extraction/test_run/" # External drive
    FEATURES_ROOT = "data/features/test/"
    
    # Clean up previous test runs if needed
    if os.path.exists(FEATURES_ROOT):
        shutil.rmtree(FEATURES_ROOT)
    os.makedirs(FEATURES_ROOT, exist_ok=True)
    # Create feature subdirs
    for subdir in ['A', 'B', 'C', 'D', 'E']:
        os.makedirs(os.path.join(FEATURES_ROOT, subdir), exist_ok=True)
        
    if os.path.exists(EXTRACTION_ROOT):
        shutil.rmtree(EXTRACTION_ROOT)
    os.makedirs(EXTRACTION_ROOT, exist_ok=True)
    
    # 1. Create Test Index (100 sample)
    print("Creating test index...")
    if not os.path.exists(INDEX_PATH):
        print(f"Index {INDEX_PATH} not found!")
        sys.exit(1)
        
    df = pd.read_parquet(INDEX_PATH)
    # Filter only valid records (status=ok)
    df = df[df['status'] == 'ok']
    
    if len(df) > 100:
        test_df = df.sample(100, random_state=42)
    else:
        test_df = df
        
    test_df.to_parquet(TEST_INDEX_PATH, index=False)
    print(f"Test index created with {len(test_df)} records.")
    
    # 2. Run Step C (HTML Structure + Content Extraction)
    print("\nRunning Step C...")
    cmd_c = f"./.venv/bin/python3 src/extraction/C/html_structure.py --mode all --index {TEST_INDEX_PATH} --scraping-root {SCRAPING_ROOT} --json-root {EXTRACTION_ROOT} --out {FEATURES_ROOT}C/html_structure.parquet"
    run_command(cmd_c)
    
    # 3. Run Step D (Readability)
    print("\nRunning Step D...")
    cmd_d = f"python3 src/extraction/D/text_readability.py --index {TEST_INDEX_PATH} --extraction-root {EXTRACTION_ROOT} --out {FEATURES_ROOT}D/text_readability.parquet"
    run_command(cmd_d)
    
    # 4. Run Step E (Semantic Similarity)
    print("\nRunning Step E...")
    cmd_e = f"python3 src/extraction/E/semantic_similarity.py --index {TEST_INDEX_PATH} --extraction-root {EXTRACTION_ROOT} --out {FEATURES_ROOT}E/semantic_similarity.parquet"
    run_command(cmd_e)
    
    # 5. Run Step Z (Merge Simulation - ACTUAL RUN)
    print("\nRunning Step Z (Merge)...")
    cmd_z = f"python3 src/extraction/Z/merge.py --index {TEST_INDEX_PATH} --out-dir {FEATURES_ROOT} --feature-base-dir {FEATURES_ROOT} --include-outliers"
    run_command(cmd_z)
    
    print("\nVerifying outputs...")
    # Find generated dataset
    dataset_file = None
    for f in os.listdir(FEATURES_ROOT):
        if f.startswith("dataset-") and f.endswith(".parquet"):
            dataset_file = os.path.join(FEATURES_ROOT, f)
            break
            
    files = [
        f"{FEATURES_ROOT}C/html_structure.parquet",
        f"{FEATURES_ROOT}D/text_readability.parquet",
        f"{FEATURES_ROOT}E/semantic_similarity.parquet",
        dataset_file
    ]
    
    for f in files:
        if f and os.path.exists(f):
            print(f"✅ Created: {f}")
            # Optional: Read and show head/stats
            try:
                out_df = pd.read_parquet(f)
                print(f"   Shape: {out_df.shape}")
                # Check for empty content/metrics
                # e.g. for E, check if sim_content is non-zero
                if 'sim_content' in out_df.columns:
                    print(f"   Sim Content Mean: {out_df['sim_content'].mean():.4f}")
                
                # Check merge result for raw_ok mapping
                if 'dataset-' in f:
                    if 'raw_ok' in out_df.columns:
                         ok_count = out_df['raw_ok'].sum()
                         print(f"   raw_ok True Count: {ok_count}")
                    else:
                        print("   ❌ raw_ok column MISSING in dataset!")
            except Exception as e:
                 print(f"   Error reading: {e}")
        else:
            print(f"❌ Missing: {f}")

    print("\nTest Run Complete. Please inspect the results.")

if __name__ == "__main__":
    main()
