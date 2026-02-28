import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import helpers
from src.analysis.D.helpers.stat_utils import calculate_vif, check_and_mitigate_collinearity, run_bootstrap_ci, run_panel_ols

def test_vif():
    print("Testing VIF Calculation...")
    # Create colinear data
    df = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
    })
    df['x3'] = df['x1'] * 0.99 + np.random.normal(0, 0.01, 100) # Highly correlated with x1
    
    predictors = ['x1', 'x2', 'x3']
    vif_df = calculate_vif(df, predictors)
    print("VIF DataFrame:\n", vif_df)
    
    if vif_df.empty:
        print("FAIL: VIF DataFrame is empty")
        return
        
    if vif_df[vif_df['variable']=='x3']['vif'].values[0] < 10:
        print(f"FAIL: VIF for x3 should be high (got {vif_df[vif_df['variable']=='x3']['vif'].values[0]})")
    else:
        print("PASS: VIF detected collinearity")

    print("\nTesting VIF Mitigation...")
    keep, drop, history = check_and_mitigate_collinearity(df, predictors, threshold=5)
    print(f"Kept: {keep}, Dropped: {drop}")
    
    if 'x1' in drop or 'x3' in drop:
        print("PASS: Collinear variable dropped")
    else:
        print("FAIL: No variable dropped")

def test_bootstrap():
    print("\nTesting Bootstrap CI...")
    df = pd.DataFrame({
        'search_term': np.random.choice(['q1', 'q2', 'q3', 'q4', 'q5'], 100),
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    })
    
    # Simple OLS function wrapper
    import statsmodels.formula.api as smf
    def simple_ols(d, f):
        return smf.ols(f, d).fit()
        
    formula = "y ~ x"
    ci = run_bootstrap_ci(simple_ols, df, formula, n_boot=20, cluster_col='search_term')
    print("Bootstrap CI:\n", ci)
    
    if ci is not None and not ci.empty:
        print("PASS: Bootstrap CI generated")
    else:
        print("FAIL: Bootstrap CI failed")

def test_r_scripts():
    print("\nChecking R Scripts existence...")
    scripts = [
        "src/analysis/E/engine_heterogeneity.R",
        "src/analysis/F/query_difficulty.R",
        "src/analysis/G/robustness.R",
        "src/analysis/H/ablation.R"
    ]
    
    all_exist = True
    for s in scripts:
        path = os.path.join(project_root, s)
        if os.path.exists(path):
            print(f"PASS: {s} exists")
        else:
            print(f"FAIL: {s} missing")
            all_exist = False
            
    # Check if Rscript is available
    import subprocess
    try:
        subprocess.run(["Rscript", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("PASS: Rscript available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Rscript not found in path. Skipping execution tests.")

if __name__ == "__main__":
    test_vif()
    test_bootstrap()
    test_r_scripts()
