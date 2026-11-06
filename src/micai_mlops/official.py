"""The official dataset versions used by notebooks 02 and 03.

They are published once to the experiment `00_official_datasets`: by notebook
00 (or scripts/setup_workspace.py) when you run on your own Databricks, or by
the instructor during a workshop. The notebooks download the files of that
run and check each one against the fingerprints stored in
`data/wine-quality/official_manifest.json`.

A copy of the same files ships with the repository in
`data/wine-quality/official/`. It is used when MLflow is not reachable or no
official run exists yet; as a last resort the files are rebuilt from the raw
data with the canonical cleaning code. The fingerprint proves all three
sources are identical.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from . import cleaning, splits
from .fingerprint import canonical_csv, fingerprint, read_dataset, short

OFFICIAL_VERSIONS = {
    "v1_clean_drop": "Strategy A: rows with any invalid feature dropped (real measurements only)",
    "v2_clean_impute": "Strategy B: invalid features imputed with the column median",
    "holdout": "Fixed evaluation set: 20% of v1, stratified on the label, never used for training",
}
MANIFEST_NAME = "official_manifest.json"
REPO_COPY_DIR = "official"
ARTIFACT_DIR = "official_datasets"


def build_official(data_dir):
    """Rebuild every official dataset from the raw file. Returns (datasets, cleaning_log)."""
    raw = cleaning.load_raw(Path(data_dir) / "modified.csv")
    clean_a, clean_b, log = cleaning.clean(raw)
    cleaning.validate(clean_a, "v1")
    cleaning.validate(clean_b, "v2")
    holdout = splits.make_holdout(clean_a)
    datasets = {"v1_clean_drop": clean_a, "v2_clean_impute": clean_b, "holdout": holdout}
    return datasets, log


def manifest_path(data_dir):
    return Path(data_dir) / MANIFEST_NAME


def read_manifest(data_dir):
    return json.loads(manifest_path(data_dir).read_text())


def make_manifest(datasets):
    return {
        name: {"rows": int(len(df)), "columns": list(df.columns), "sha256": fingerprint(df),
               "description": OFFICIAL_VERSIONS[name]}
        for name, df in datasets.items()
    }


def read_repo_copy(data_dir):
    """The official datasets stored in the repository, checked against the manifest."""
    folder = Path(data_dir) / REPO_COPY_DIR
    manifest = read_manifest(data_dir)
    datasets = {name: read_dataset(folder / f"{name}.csv") for name in OFFICIAL_VERSIONS}
    bad = [n for n, df in datasets.items() if fingerprint(df) != manifest[n]["sha256"]]
    if bad:
        raise ValueError(f"repository copy does not match the manifest: {bad}")
    return datasets


def _download_from_run(run_id, dst):
    import mlflow
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ARTIFACT_DIR, dst_path=dst)
    return {name: read_dataset(Path(path) / f"{name}.csv") for name in OFFICIAL_VERSIONS}


def load_official(ws, verbose=True):
    """Return the official datasets (dict name -> DataFrame) for this workshop session."""
    manifest = read_manifest(ws.data_dir)
    source = None
    datasets = None
    if ws.online and ws.official_run_id:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                datasets = _download_from_run(ws.official_run_id, tmp)
            source = f"MLflow run {ws.official_run_id[:8]}… in {ws.experiment('official')}"
        except Exception as exc:
            if verbose:
                print(f"⚠️  Could not download the official run ({type(exc).__name__}); rebuilding locally.")
    if datasets is not None:
        bad = [n for n, df in datasets.items() if fingerprint(df) != manifest[n]["sha256"]]
        if bad:
            if verbose:
                print(f"⚠️  Fingerprint mismatch for {bad}; rebuilding the official files locally.")
            datasets = None
    if datasets is None:
        try:
            datasets = read_repo_copy(ws.data_dir)
            source = "the repository copy (data/wine-quality/official/)"
        except Exception:
            rebuilt, _ = build_official(ws.data_dir)
            datasets = {n: read_dataset_from_frame(df) for n, df in rebuilt.items()}
            source = "rebuilt locally from data/wine-quality/modified.csv"
    rows = []
    for name, df in datasets.items():
        ok = fingerprint(df) == manifest[name]["sha256"]
        if not ok:
            raise RuntimeError(f"{name} does not match the official manifest")
        rows.append({"version": name, "rows": len(df), "fingerprint": short(manifest[name]["sha256"]),
                     "matches_manifest": "✅"})
    if verbose:
        print(f"Official datasets loaded from {source}")
    return datasets, pd.DataFrame(rows).set_index("version")


def read_dataset_from_frame(df):
    """Round-trip through the canonical CSV so dtypes/precision equal a downloaded file."""
    import io
    return read_dataset(io.StringIO(canonical_csv(df)))


def write_datasets(datasets, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        (folder / f"{name}.csv").write_text(canonical_csv(df))
    return folder
