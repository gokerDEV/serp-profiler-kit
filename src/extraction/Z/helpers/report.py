
import pandas as pd
import os
import sys
import contextlib

def generate_dataset_report(df, report_path, params, acceptance_stats=None):
    """
    Generates a detailed markdown report for the merged dataset.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f):
            print(f"# Dataset Generation Report")
            print(f"**Generated:** {pd.Timestamp.now()}")
            print(f"**File:** {os.path.basename(report_path).replace('.md', '.parquet')}")
            
            print("\n## Parameters")
            for k, v in params.items():
                print(f"- **{k}:** {v}")
                
            print(f"\n## Dataset Statistics")
            print(f"- **Total Records:** {len(df)}")
            
            # Acceptance Criteria Summary
            if acceptance_stats:
                print("\n## Acceptance Criteria Summary")
                print("| Filter Description | Records Dropped |")
                print("|:---|---:|")
                for desc, count in acceptance_stats.items():
                    print(f"| {desc} | {count} |")
                total_dropped = sum(acceptance_stats.values())
                print(f"\n**Total Dropped by Acceptance Criteria:** {total_dropped}")

            # 1. Search Engine & Rank Distribution (1-20)
            if 'search_engine' in df.columns and 'rank' in df.columns:
                print("\n### Distribution by Search Engine & Rank")
                
                # Filter ranks 1-20
                valid_ranks = df[df['rank'].between(1, 20)]
                
                if not valid_ranks.empty:
                    # Pivot: Index=search_engine, Columns=rank, Values=count
                    pivot = valid_ranks.pivot_table(index='search_engine', columns='rank', aggfunc='size', fill_value=0)
                    
                    # Add Total column
                    pivot['Total'] = pivot.sum(axis=1)
                    
                    # Reset index to make search_engine a column
                    pivot = pivot.reset_index()
                    print(pivot.to_markdown(index=False))
                else:
                    print("No records found in rank range 1-20.")

            # 2. Source Domain Distribution 
            if 'is_source_domain' in df.columns:
                source_count = df['is_source_domain'].sum()
                print(f"\n### Source Domain Distribution (is_source_domain=True)")
                print(f"**Total Source Records:** {source_count}")
                
                if source_count > 0:
                    source_domain_df = df[df['is_source_domain'] == True]
                    source_domain_valid = source_domain_df[source_domain_df['rank'].between(1, 20)]
                    
                    if not source_domain_valid.empty:
                        source_domain_pivot = source_domain_valid.pivot_table(index='search_engine', columns='rank', aggfunc='size', fill_value=0)
                        source_domain_pivot['Total'] = source_domain_pivot.sum(axis=1)
                        source_domain_pivot = source_domain_pivot.reset_index()
                        print(source_domain_pivot.to_markdown(index=False))
                    else:
                        print("No source domain results found in Top 20.")
            
            # 3. Missing Values (All Columns)
            print("\n### Missing Values Analysis")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
            # Sort by Missing % desc
            missing_df = missing_df.sort_values('Missing %', ascending=False)
            print(missing_df.to_markdown())
            
            # 4. Column Details & Distributions
            print("\n## Column Details & Distributions")
            
            for col in df.columns:
                print(f"\n### Column: `{col}`")
                dtype = df[col].dtype
                print(f"- **Type:** {dtype}")
                
                # Calculate basic stats
                if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                    rows = []
                    stats_full = df[col].describe().to_dict()
                    stats_full['Subset'] = 'Full'
                    rows.append(stats_full)
                    
                    if 'is_source_domain' in df.columns:
                        stats_without = df[df['is_source_domain'] == False][col].describe().to_dict()
                        stats_without['Subset'] = 'Without source domain'
                        rows.append(stats_without)
                        
                        stats_only = df[df['is_source_domain'] == True][col].describe().to_dict()
                        stats_only['Subset'] = 'Only source domain'
                        rows.append(stats_only)
                    
                    df_stats = pd.DataFrame(rows)
                    cols_order = ['Subset'] + [c for c in df_stats.columns if c != 'Subset']
                    print(df_stats[cols_order].to_markdown(index=False))
                else:
                    # Categorical / String / Boolean
                    unique_count = df[col].nunique()
                    print(f"- **Unique Values:** {unique_count}")
                    
                    # Show top values if reasonable number
                    if unique_count <= 20:
                        print(df[col].value_counts().to_markdown())
                    else:
                        print(f"Top 5 Values:")
                        print(df[col].value_counts().head(5).to_markdown())
