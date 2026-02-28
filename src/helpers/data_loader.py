import pandas as pd
import sys
from pathlib import Path

def load_valid_index(index_path: str, required_cols: set = None) -> pd.DataFrame:
    """
    Loads index.parquet and filters for valid records (status='ok') by default.
    This ensures subsequent extraction steps only process valid data, avoiding
    misleading failure rates for known-bad records (e.g. captcha, missing artifact).
    """
    print(f"Loading index from {index_path}...", file=sys.stderr)
    try:
        df = pd.read_parquet(index_path)
    except FileNotFoundError:
        print(f"Index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    # Validate Columns
    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Missing required columns in index: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

    # Filter for Valid Records
    initial_count = len(df)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"Filtered {filtered_count} non-ok records (captcha, missing, etc.). Processing {len(df)} valid records.", file=sys.stderr)
    else:
        print("Warning: 'status' column not found in index. Processing all records.", file=sys.stderr)

    return df

import yaml
import os

def load_analysis_dataset(dataset_path: str, schema_path="src/analysis_v1.yml") -> pd.DataFrame:
    print(f"Loading dataset from {dataset_path}...", file=sys.stderr)
    try:
        df = pd.read_parquet(dataset_path)
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
        
    multipliers = get_feature_multipliers(schema_path)
    for feat, mult in multipliers.items():
        if feat in df.columns and mult != 1:
            df[feat] = pd.to_numeric(df[feat], errors='coerce') * mult
            
    return df

def get_analysis_categories(schema_path="src/analysis_v1.yml"):
    project_root = str(Path(__file__).resolve().parents[2])
    full_path = os.path.join(project_root, schema_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(os.getcwd(), schema_path)
        
    with open(full_path, "r") as f:
        schema = yaml.safe_load(f)
        
    analysis = schema.get("analysis", {})
    return list(analysis.keys())

def get_analysis_features(category=None, schema_path="src/analysis_v1.yml"):
    project_root = str(Path(__file__).resolve().parents[2])
    full_path = os.path.join(project_root, schema_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(os.getcwd(), schema_path)
        
    with open(full_path, "r") as f:
        schema = yaml.safe_load(f)
        
    analysis = schema.get("analysis", {})
    if category:
        return list(analysis.get(category, {}).keys())
    
    features = []
    for cat, feats in analysis.items():
        if feats:
            features.extend(list(feats.keys()))
    return features

def load_analysis_metrics(schema_path="src/analysis_v1.yml"):
    project_root = str(Path(__file__).resolve().parents[2])
    full_path = os.path.join(project_root, schema_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(os.getcwd(), schema_path)
        
    with open(full_path, "r") as f:
        schema = yaml.safe_load(f)
        
    analysis = schema.get("analysis", {})
    return {cat: list(features.keys()) if features else [] for cat, features in analysis.items()}

def get_feature_categories(schema_path="src/analysis_v1.yml"):
    project_root = str(Path(__file__).resolve().parents[2])
    full_path = os.path.join(project_root, schema_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(os.getcwd(), schema_path)
        
    with open(full_path, "r") as f:
        schema = yaml.safe_load(f)
        
    analysis = schema.get("analysis", {})
    mapping = {}
    for cat, features in analysis.items():
        if not features: continue
        for feat in features.keys():
            mapping[feat] = cat
    return mapping

def get_feature_multipliers(schema_path="src/analysis_v1.yml"):
    project_root = str(Path(__file__).resolve().parents[2])
    full_path = os.path.join(project_root, schema_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(os.getcwd(), schema_path)
        
    with open(full_path, "r") as f:
        schema = yaml.safe_load(f)
        
    analysis = schema.get("analysis", {})
    multipliers = {}
    for cat, features in analysis.items():
        if not features: continue
        for feat, config in features.items():
            if config and isinstance(config, dict):
                multipliers[feat] = config.get("multiplier", 1)
            else:
                multipliers[feat] = 1
    return multipliers

def get_rank_tiers(schema_path="src/analysis_v1.yml"):
    project_root = str(Path(__file__).resolve().parents[2])
    full_path = os.path.join(project_root, schema_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(os.getcwd(), schema_path)
        
    with open(full_path, "r") as f:
        schema = yaml.safe_load(f)
        
    tiers = schema.get("rank_tiers")
    if not tiers or "bins" not in tiers or "labels" not in tiers or "logit_cut_points" not in tiers:
        raise ValueError(f"'rank_tiers' configuration with 'bins', 'labels' and 'logit_cut_points' must be defined in {schema_path}.")
        
    return {
        "bins": tiers["bins"],
        "labels": tiers["labels"],
        "logit_cut_points": tiers["logit_cut_points"]
    }

