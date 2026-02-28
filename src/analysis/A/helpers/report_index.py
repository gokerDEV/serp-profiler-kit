import pandas as pd

def analyze_index(file_path):
    print(f"--- Analyzing Index: {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Total records: {len(df)}")
    # Check for unique IDs
    if 'record_id' in df.columns:
        unique_ids = df['record_id'].nunique()
        print(f"Unique Record IDs: {unique_ids}")
        if unique_ids == len(df):
            print("record_id is unique for all records.")
        else:
            print(f"WARNING: record_id is NOT unique. Duplicates: {len(df) - unique_ids}")
    else:
        print("record_id column not found.")

    # Check for unique URLs (using 'url' column)
    if 'url' in df.columns:
        unique_urls = df['url'].nunique()
        print(f"Unique URLs: {unique_urls}")
        print(f"Duplicate URLs: {len(df) - unique_urls}")
    else:
        print("URL column (url) not found.")

        print("\nScraping Status:")
        print(df['status'].value_counts(dropna=False))
        
    if 'html_size_bytes' in df.columns:
        print("\nHTML Size Stats (bytes):")
        stats_all = df['html_size_bytes'].describe()
        # Format as standard float with 2 decimal places
        print("All records:")
        print(stats_all.apply(lambda x: "{:,.2f}".format(x)))

        # Stats for status='ok'
        if 'status' in df.columns:
            df_ok = df[df['status'] == 'ok']
            if not df_ok.empty:
                print("\nRecords with status='ok':")
                stats_ok = df_ok['html_size_bytes'].describe()
                print(stats_ok.apply(lambda x: "{:,.2f}".format(x)))

    # --- Distribution Tables ---
    if 'status' in df.columns and 'rank' in df.columns and 'search_engine' in df.columns:
        # Convert rank to numeric if needed
        df['rank_numeric'] = pd.to_numeric(df['rank'], errors='coerce')
        
        # 1. OK Count Table (Rank 1-20)
        print("\n### OK Status Distribution by Rank (1-20)")
        df_ok_ranks = df[(df['status'] == 'ok') & (df['rank_numeric'].between(1, 20))]
        if not df_ok_ranks.empty:
            pivot_ok = df_ok_ranks.pivot_table(index='search_engine', columns='rank_numeric', aggfunc='size', fill_value=0)
            pivot_ok['Total'] = pivot_ok.sum(axis=1)
            print(pivot_ok.to_markdown())
        else:
            print("No 'ok' records found in ranks 1-20.")

        # 2. Source Domain Table (Rank 1-20)
        if 'is_source_domain' in df.columns:
            print(f"\n### Source Domain Distribution (Rank 1-20)")
            df_src = df[(df['is_source_domain'] == True) & (df['rank_numeric'].between(1, 20))]
            # We don't filter by status='ok' unless requested. Usually we want to see ALL appearances.
            # But earlier user said "count(status_ok)/total". Maybe they want OK source domain?
            # User phrase: "source domain icin de bir tablo olusturalim" (general).
            # "Guardian sonuc sayisini arama motoru ve rank bazinda". Usually implies VISIBLE results.
            # I will show ALL source domain records that appeared in SERP (regardless of scrape status).
            # Because SERP appearance is what matters for "sonuc sayisi". Scrape status is secondary.
            
            if not df_src.empty:
                 pivot_src = df_src.pivot_table(index='search_engine', columns='rank_numeric', aggfunc='size', fill_value=0)
                 pivot_src['Total'] = pivot_src.sum(axis=1)
                 print(pivot_src.to_markdown())
            else:
                 print("No source domain records found in ranks 1-20.")
        
    print("-" * 30 + "\n")
