import pandas as pd

def analyze_keywords(file_path):
    print(f"--- Analyzing Keywords: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'section' in df.columns:
        print("\nDistribution by Section:")
        print(df['section'].value_counts())
        
    if 'pub_date' in df.columns:
        print("\nDate Range:")
        try:
            dates = pd.to_datetime(df['pub_date'], errors='coerce')
            print(f"Min: {dates.min()}")
            print(f"Max: {dates.max()}")
        except Exception as e:
            print(f"Could not parse dates: {e}")
            
    print("-" * 30 + "\n")
