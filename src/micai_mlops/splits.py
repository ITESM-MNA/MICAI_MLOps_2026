"""Fixed evaluation set (holdout) and leakage-safe training sets.

The holdout is drawn ONCE from the drop-rows version (v1), whose rows are all
real measurements, and published with the official datasets. Every model in
the workshop is evaluated on the same holdout, whichever dataset version it
was trained on, so scores are comparable.

Two kinds of leakage are prevented when a training set is built:
1. the same record (same row_id) in training and holdout;
2. near-duplicates: a training row that matches a holdout row in all but one
   column (for example an injected duplicate whose corrupted cell was later
   imputed with the median).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contract import EXPECTED_COLUMNS, GOOD_THRESHOLD, LABEL, ROW_ID, TARGET

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42


def add_label(df):
    out = df.copy()
    out[LABEL] = (out[TARGET] >= GOOD_THRESHOLD).astype("int64")
    return out


def make_holdout(clean_a):
    """Stratified 20% sample of v1 (real measurements only)."""
    labels = (clean_a[TARGET] >= GOOD_THRESHOLD).astype(int)
    _, holdout = train_test_split(
        clean_a, test_size=HOLDOUT_FRACTION, stratify=labels, random_state=HOLDOUT_SEED,
    )
    return holdout.sort_values(ROW_ID).reset_index(drop=True)


def near_duplicate_mask(candidates, reference, max_differences=1, decimals=6):
    """True for each candidate row that differs from some reference row in at most
    `max_differences` of the contract columns."""
    a = candidates[EXPECTED_COLUMNS].round(decimals).to_numpy()
    b = reference[EXPECTED_COLUMNS].round(decimals).to_numpy()
    hit = np.zeros(len(a), dtype=bool)
    for start in range(0, len(a), 512):
        block = a[start:start + 512]
        diffs = (block[:, None, :] != b[None, :, :]).sum(axis=2)
        hit[start:start + 512] = (diffs <= max_differences).any(axis=1)
    return hit


def training_set(clean_df, holdout):
    """Remove holdout records and their near-duplicates from a cleaned dataset."""
    same_record = clean_df[ROW_ID].isin(holdout[ROW_ID])
    near_dup = near_duplicate_mask(clean_df, holdout)
    keep = ~(same_record | near_dup)
    report = {
        "rows_in_version": int(len(clean_df)),
        "removed_same_record": int(same_record.sum()),
        "removed_near_duplicates": int((near_dup & ~same_record).sum()),
        "rows_for_training": int(keep.sum()),
    }
    return clean_df[keep].reset_index(drop=True), report


def xy(df, features):
    """Feature matrix and binary label."""
    labelled = add_label(df)
    return labelled[features], labelled[LABEL]


__all__ = ["add_label", "make_holdout", "near_duplicate_mask", "training_set", "xy",
           "HOLDOUT_FRACTION", "HOLDOUT_SEED"]


def summary(df):
    labels = (df[TARGET] >= GOOD_THRESHOLD)
    return pd.Series({"rows": len(df), "good_rate": round(float(labels.mean()), 3)})
