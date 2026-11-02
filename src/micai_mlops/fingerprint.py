"""Content fingerprints for datasets.

A fingerprint is a SHA-256 hash of a canonical text form of a DataFrame. Two
DataFrames with the same values, columns and row order get the same
fingerprint on any machine and pandas version, so it proves that a file
downloaded from MLflow (or rebuilt locally) is exactly the official version.
"""

import hashlib

import pandas as pd


def canonical_csv(df):
    """Deterministic CSV text: fixed float format, no index, '\\n' line endings."""
    return df.to_csv(index=False, float_format="%.6f", lineterminator="\n")


def fingerprint(df):
    return hashlib.sha256(canonical_csv(df).encode("utf-8")).hexdigest()


def short(fp):
    return fp[:12]


def read_dataset(path):
    """Read a dataset CSV written by `canonical_csv` with stable dtypes."""
    df = pd.read_csv(path)
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int64")
        else:
            df[col] = df[col].astype("float64")
    return df
