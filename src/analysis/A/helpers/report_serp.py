import pandas as pd

def analyze_serp(file_path):
    print(f"--- Analyzing SERP: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Total rows: {len(df)}")
    print(f"Unique queries: {df['search_term'].nunique() if 'search_term' in df else 'N/A'}")
    
    if 'search_engine' in df:
        print("\nSearch Engine Distribution:")
        print(df['search_engine'].value_counts())
        
    print(f"\nMissing snippets: {df['snippet'].isnull().sum()} / {len(df)}")
    print("-" * 30 + "\n")
