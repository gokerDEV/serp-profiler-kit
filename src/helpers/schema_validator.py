import yaml
import sys
import os
import pandas as pd
from pathlib import Path

# Add project root path logic
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "src/schema_v1.yml")

def load_schema():
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")
    
    with open(SCHEMA_PATH, 'r') as f:
        schema = yaml.safe_load(f)
    return schema

def validate_columns(df, strict=False, exclude_columns=None):
    """
    Validates DataFrame columns against schema_v1.yml.
    
    Args:
        df (pd.DataFrame): Dataset to validate.
        strict (bool): If True, raises Error on missing columns. If False, prints warning.
        exclude_columns (list): Optional list of columns to ignore (e.g. not yet generated).
    
    Returns:
        bool: True if valid (or only warnings), False if critical failure (when strict=False).
        (When strict=True, it raises Exception instead of returning False).
    """
    try:
        schema = load_schema()
    except Exception as e:
        print(f"Warning: Could not load schema: {e}", file=sys.stderr)
        return False

    canonical_cols = set(schema.get('columns', {}).keys())
    
    if exclude_columns:
        for col in exclude_columns:
            if col in canonical_cols:
                canonical_cols.remove(col)
    df_cols = set(df.columns)
    
    # Include index names if they are strings (e.g. 'record_id')
    if df.index.name:
        df_cols.add(df.index.name)
    elif df.index.names:
        for name in df.index.names:
            if name: df_cols.add(name)
    
    missing_cols = canonical_cols - df_cols
    extra_cols = df_cols - canonical_cols
    
    # Check for critical missing columns
    # We assume ALL columns in schema are "expected" for specific steps.
    # However, merge scripts might produce partial datasets if steps failed.
    # We will log missing columns.
    
    if missing_cols:
        msg = f"❌ Validation Failed: Missing {len(missing_cols)} canonical columns:\n" + \
              f"{list(missing_cols)[:10]}..." # Truncate for display
        
        if strict:
            raise ValueError(msg)
        else:
            print(msg, file=sys.stderr)
            return False
            
    # Check aliases (Optional: Suggest renaming?)
    aliases = schema.get('aliases', {})
    found_aliases = [col for col in extra_cols if col in aliases]
    
    if found_aliases:
        print(f"⚠️  Found alias columns that should be renamed: {found_aliases}", file=sys.stderr)
        
    print("✅ Schema Validation Passed.", file=sys.stderr)
    return True

def enforce_schema(df):
    """
    Wrapper that enforces schema and raises error if invalid.
    """
    validate_columns(df, strict=True)
