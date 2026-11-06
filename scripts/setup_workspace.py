"""Prepare your own MLflow workspace on Databricks (run once).

Same as notebook 00, from a terminal on your computer:

    pip install -r requirements-dev.txt
    python scripts/setup_workspace.py --config config.json

It creates the folder and the four experiments, then publishes the official
dataset versions and the reference model. Running it again is safe.
See docs/SETUP_YOUR_OWN_MLFLOW.md.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from micai_mlops.setup import setup_workshop  # noqa: E402
from micai_mlops.workspace import prepare_workspace  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to your config.json (never commit it)")
    parser.add_argument("--nickname", default="me", help="tag used for the reference model run")
    parser.add_argument("--force", action="store_true", help="publish again even if the runs exist")
    args = parser.parse_args()
    if not Path(args.config).expanduser().exists():
        sys.exit(f"❌ {args.config} not found. Copy config.template.json to config.json and fill it in.")
    ws = setup_workshop(str(Path(args.config).expanduser()), args.nickname, notebook="00", repo_dir=REPO)
    if not ws.online:
        sys.exit("❌ Could not connect to Databricks with this config (see the message above).")
    prepare_workspace(ws, force=args.force)


if __name__ == "__main__":
    main()
