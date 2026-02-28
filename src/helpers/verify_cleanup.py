import pandas as pd
import sys

def verify_parquet(path):
    print(f"Checking {path}...")
    try:
        df = pd.read_parquet(path)
        print(f"Total rows: {len(df)}")
        if 'step_status' in df.columns:
            status_dist = df['step_status'].value_counts()
            print("Status distribution:")
            print(status_dist)
            
            non_ok = df[df['step_status'] != 'ok']
            if not non_ok.empty:
                print(f"\nFound {len(non_ok)} non-ok records.")
            else:
                print("\nClean! All records are status='ok'.")
        else:
            print("Column 'step_status' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_parquet("data/features/D/text_readability.parquet")
