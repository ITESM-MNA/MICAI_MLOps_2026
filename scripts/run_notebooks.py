"""Execute the notebooks top to bottom in local (offline) MLflow mode.

    python scripts/run_notebooks.py [--keep-outputs DIR]
Used by CI and for local rehearsal. Exits non-zero if any cell fails.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_data_quality_eda_versioning.ipynb", "02_training_tracking.ipynb", "03_monitoring_drift.ipynb"]

parser = argparse.ArgumentParser()
parser.add_argument("--keep-outputs", help="folder where executed notebooks are written")
args = parser.parse_args()

os.environ.setdefault("MICAI_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
failed = False
for name in NOTEBOOKS:
    nb = nbformat.read(REPO / "notebooks" / name, as_version=4)
    start = time.time()
    try:
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(REPO / "notebooks")}}).execute()
        print(f"OK   {name}  {time.time() - start:.1f}s")
    except Exception as exc:
        failed = True
        print(f"FAIL {name}: {str(exc)[-2000:]}")
    if args.keep_outputs:
        Path(args.keep_outputs).mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, Path(args.keep_outputs) / name)
sys.exit(1 if failed else 0)
