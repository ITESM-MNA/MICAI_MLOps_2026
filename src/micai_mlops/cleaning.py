"""Canonical cleaning pipeline (same logic as notebook 01).

Notebook 01 shows these functions inline so participants can read them. This
module is the reference copy used by the presenter script and by the tests;
the tests check that both produce identical datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .contract import (
    DATA_CONTRACT, EXPECTED_COLUMNS, FEATURES, PLACEHOLDER_TOKENS, ROW_ID, TARGET,
)


class CleaningLog:
    """Records what every cleaning step did (rows in/out, cells changed)."""

    def __init__(self):
        self.entries = []

    def add(self, step, detail, rows_before, rows_after, cells_changed=None):
        self.entries.append({
            "step": step, "detail": detail,
            "rows_before": rows_before, "rows_after": rows_after,
            "cells_changed": cells_changed,
        })

    def to_frame(self):
        return pd.DataFrame(self.entries)


def load_raw(path):
    """Read every value as text and add a stable row identifier."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.insert(0, ROW_ID, np.arange(len(raw), dtype="int64"))
    return raw


def enforce_schema(df, log):
    dropped = [c for c in df.columns if c not in EXPECTED_COLUMNS + [ROW_ID]]
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.add("enforce_schema", f"dropped {dropped}", len(df), len(df))
    return df[[ROW_ID] + EXPECTED_COLUMNS].copy()


def strip_whitespace(df, log):
    out = df.copy()
    out[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].apply(lambda s: s.str.strip())
    changed = int((out[EXPECTED_COLUMNS] != df[EXPECTED_COLUMNS]).sum().sum())
    log.add("strip_whitespace", "str.strip() on all columns", len(df), len(out), changed)
    return out


def placeholders_to_nan(df, log, tokens=PLACEHOLDER_TOKENS):
    out = df.copy()
    is_placeholder = df[EXPECTED_COLUMNS].apply(lambda s: s.str.lower().isin(tokens) | (s == ""))
    out[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].mask(is_placeholder)
    log.add("placeholders_to_nan", f"{len(tokens)} tokens + empty strings", len(df), len(out),
            int(is_placeholder.sum().sum()))
    return out


def cast_numeric(df, log):
    out = df.copy()
    numeric = df[EXPECTED_COLUMNS].apply(lambda s: pd.to_numeric(s, errors="coerce")).astype("float64")
    newly_nan = int((numeric.isna() & df[EXPECTED_COLUMNS].notna()).sum().sum())
    out = pd.concat([df[[ROW_ID]], numeric], axis=1)
    log.add("cast_numeric", "to float64 (unparseable -> NaN)", len(df), len(out), newly_nan)
    return out


def invalidate_out_of_range(df, log, contract=DATA_CONTRACT):
    out = df.copy()
    n = 0
    for col, (lo, hi) in contract.items():
        bad = (out[col] < lo) | (out[col] > hi)
        n += int(bad.sum())
        out.loc[bad, col] = np.nan
    log.add("invalidate_out_of_range", "values outside data-contract bounds -> NaN", len(df), len(out), n)
    return out


def invalidate_inconsistent_so2(df, log):
    out = df.copy()
    bad = out["free sulfur dioxide"] > out["total sulfur dioxide"]
    out.loc[bad, ["free sulfur dioxide", "total sulfur dioxide"]] = np.nan
    log.add("invalidate_inconsistent_so2", "free SO2 > total SO2 -> both NaN", len(df), len(out),
            int(bad.sum()) * 2)
    return out


def drop_invalid_target(df, log):
    valid = df[TARGET].notna() & (df[TARGET] % 1 == 0)
    out = df[valid].copy()
    out[TARGET] = out[TARGET].astype("int64")
    log.add("drop_invalid_target", "rows with missing/invalid quality removed", len(df), len(out))
    return out


def strategy_a_drop_rows(df, log):
    out = df.dropna(subset=FEATURES).copy()
    log.add("A: drop_incomplete_rows", "rows with any missing feature removed", len(df), len(out))
    return out


def strategy_b_impute_median(df, log):
    out = df.copy()
    medians = out[FEATURES].median()
    n = int(out[FEATURES].isna().sum().sum())
    out[FEATURES] = out[FEATURES].fillna(medians)
    log.add("B: impute_median", "missing features <- column median", len(df), len(out), n)
    consistent = out["free sulfur dioxide"] <= out["total sulfur dioxide"]
    log.add("B: recheck_so2_after_impute", "rows made inconsistent by imputation removed",
            len(out), int(consistent.sum()))
    return out[consistent].copy()


def drop_duplicates(df, log):
    # Duplicates are judged on the measured values only, never on the row identifier.
    out = df.drop_duplicates(subset=EXPECTED_COLUMNS).copy()
    log.add("drop_duplicates", "exact duplicate rows removed", len(df), len(out))
    return out


def clean(raw_df):
    """Run the full pipeline. Returns (clean_a, clean_b, cleaning_log_frame)."""
    log = CleaningLog()
    base = (
        raw_df
        .pipe(enforce_schema, log)
        .pipe(strip_whitespace, log)
        .pipe(placeholders_to_nan, log)
        .pipe(cast_numeric, log)
        .pipe(invalidate_out_of_range, log)
        .pipe(invalidate_inconsistent_so2, log)
        .pipe(drop_invalid_target, log)
    )
    clean_a = base.pipe(strategy_a_drop_rows, log).pipe(drop_duplicates, log).reset_index(drop=True)
    clean_b = base.pipe(strategy_b_impute_median, log).pipe(drop_duplicates, log).reset_index(drop=True)
    return clean_a, clean_b, log.to_frame()


def validate(df, name="dataset"):
    """Validation gate: raises AssertionError if the data contract does not hold."""
    assert list(df.columns) == [ROW_ID] + EXPECTED_COLUMNS, "column set/order"
    assert df[ROW_ID].is_unique, "row_id is unique"
    assert df.notna().all().all(), "no missing values"
    assert all(pd.api.types.is_float_dtype(df[c]) for c in FEATURES), "numeric features"
    assert pd.api.types.is_integer_dtype(df[TARGET]), "integer target"
    for col, (lo, hi) in DATA_CONTRACT.items():
        assert df[col].between(lo, hi).all(), f"{col} within contract bounds"
    assert (df["free sulfur dioxide"] <= df["total sulfur dioxide"]).all(), "SO2 consistency"
    assert not df.duplicated(subset=EXPECTED_COLUMNS).any(), "no duplicate rows"
    return True


def data_dir(repo_dir):
    return Path(repo_dir) / "data" / "wine-quality"
