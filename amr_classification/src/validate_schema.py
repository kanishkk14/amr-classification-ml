"""
validate_schema.py — Input data validation for the AMR classification pipeline.
"""

import pandas as pd

REQUIRED_COLS = ["Genome_ID", "MLST_Group", "Phenotype_Resistant"]
VALID_LABELS  = {0, 1}


def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate that the input DataFrame has the required columns,
    no missing values in key fields, no duplicate genome IDs,
    and binary phenotype labels.
    """
    # Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # No nulls in required columns
    null_cols = [c for c in REQUIRED_COLS if df[c].isnull().any()]
    if null_cols:
        raise ValueError(f"Missing values in required columns: {null_cols}")

    # Unique genome IDs
    if df["Genome_ID"].duplicated().any():
        n = df["Genome_ID"].duplicated().sum()
        raise ValueError(f"Duplicate Genome_IDs found: {n} duplicates.")

    # Binary phenotype labels
    observed = set(df["Phenotype_Resistant"].dropna().unique())
    invalid  = observed - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Phenotype_Resistant must contain only 0 or 1. Found: {invalid}"
        )

    return True
