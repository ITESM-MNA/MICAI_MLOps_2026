"""Data contract for the Wine Quality (white wine) dataset.

The contract is the written agreement of what valid data looks like. It is the
single source of truth used by the cleaning pipeline, the validation gate and
the presenter script that publishes the official dataset versions.

Bounds come from oenology / lab-measurement knowledge and are deliberately
generous: they catch impossible values (a pH of 700), not unusual real wines.
"""

ROW_ID = "row_id"      # stable identifier of each raw record (data lineage)
TARGET = "quality"     # sensory score 0-10 (integer)
GOOD_THRESHOLD = 7     # binary target used for modelling: good = quality >= 7
LABEL = "good"         # name of the binary target column

DATA_CONTRACT = {
    #  column                  (min,   max)    units / rationale
    "fixed acidity":        (3.0,   16.0),  # g(tartaric acid)/L
    "volatile acidity":     (0.05,  1.6),   # g(acetic acid)/L; above ~1.2 wine is vinegar-like
    "citric acid":          (0.0,   1.7),   # g/L, cannot be negative
    "residual sugar":       (0.0,   70.0),  # g/L, even sweet table wines rarely exceed this
    "chlorides":            (0.005, 0.7),   # g(sodium chloride)/L
    "free sulfur dioxide":  (1.0,   300.0), # mg/L
    "total sulfur dioxide": (5.0,   450.0), # mg/L
    "density":              (0.985, 1.04),  # g/cm^3, wine is close to water
    "pH":                   (2.7,   4.0),   # wine is acidic
    "sulphates":            (0.2,   2.0),   # g(potassium sulphate)/L
    "alcohol":              (8.0,   15.5),  # % vol for still table wine
    "quality":              (0,     10),    # sensory score, integer 0-10 (target)
}

EXPECTED_COLUMNS = list(DATA_CONTRACT)
FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET]

# Placeholder strings that upstream systems write instead of an empty cell.
# Found by diagnostic test T4 in notebook 01; compared case-insensitively.
PLACEHOLDER_TOKENS = frozenset({"?", "error", "invalid", "n/a", "nan", "null"})


def contract_as_dict():
    """JSON-friendly version of the contract (logged to MLflow with each dataset)."""
    return {
        "row_id_column": ROW_ID,
        "target": TARGET,
        "binary_label": {"name": LABEL, "rule": f"{TARGET} >= {GOOD_THRESHOLD}"},
        "features": FEATURES,
        "bounds": {c: list(b) for c, b in DATA_CONTRACT.items()},
        "cross_field_rules": ["free sulfur dioxide <= total sulfur dioxide"],
        "placeholder_tokens": sorted(PLACEHOLDER_TOKENS),
    }
