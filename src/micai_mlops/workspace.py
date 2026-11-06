"""Prepare your own MLflow workspace on Databricks (run once, after the workshop).

Used by notebook 00 and scripts/setup_workspace.py. It needs a connected
session from `setup_workshop(...)` and then:

1. checks that the folder and the four experiments exist (setup creates them);
2. publishes the official dataset versions to `00_official_datasets`, copying
   the processed files stored in `data/wine-quality/official/` (checked
   against the manifest; nothing is re-cleaned);
3. publishes the reference ("production") model to `02_training`.

Running it again is safe: a step whose result already exists is skipped,
unless `force=True`.
"""

import hashlib
import tempfile

from . import contract, modeling, official, splits
from .fingerprint import fingerprint, short


def _ok(text):
    print(f"✅ {text}")


def _existing_official_run(ws, manifest):
    """The latest official run whose files match the manifest, or None."""
    import mlflow
    runs = mlflow.search_runs(experiment_names=[ws.experiment("official")],
                              filter_string="tags.role = 'official' AND attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if runs.empty:
        return None
    run = runs.iloc[0]
    tags_match = all(run.get(f"tags.fingerprint_{name}") == short(manifest[name]["sha256"])
                     for name in official.OFFICIAL_VERSIONS)
    return run.run_id if tags_match else None


def publish_official(ws, force=False):
    import mlflow
    manifest = official.read_manifest(ws.data_dir)
    datasets = official.read_repo_copy(ws.data_dir)
    _ok("Repository copy of the official datasets matches the manifest: "
        + ", ".join(f"{n} {len(d):,} rows" for n, d in datasets.items()))
    existing = None if force else _existing_official_run(ws, manifest)
    if existing:
        _ok(f"Official datasets already published (run {existing[:8]}…): skipped")
        return existing

    raw_path = ws.data_dir / "modified.csv"
    mlflow.set_experiment(ws.experiment("official"))
    with mlflow.start_run(run_name="official-datasets") as run:
        mlflow.set_tags({"role": "official", "published_by": "notebook 00 / setup_workspace.py",
                         "mlflow.note.content": "Official dataset versions used by notebooks 02 and 03 "
                                                "(copied from data/wine-quality/official/)."})
        mlflow.log_params({"raw_file": "data/wine-quality/modified.csv",
                           "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()[:16],
                           "holdout_fraction": splits.HOLDOUT_FRACTION, "holdout_seed": splits.HOLDOUT_SEED,
                           "label_rule": f"quality >= {contract.GOOD_THRESHOLD}"})
        for name, df in datasets.items():
            mlflow.log_input(mlflow.data.from_pandas(df, name=name, targets="quality"), context="official")
            mlflow.log_metric(f"rows_{name}", len(df))
            mlflow.set_tag(f"fingerprint_{name}", short(fingerprint(df)))
        with tempfile.TemporaryDirectory() as tmp:
            official.write_datasets(datasets, tmp)
            mlflow.log_artifacts(tmp, artifact_path=official.ARTIFACT_DIR)
        mlflow.log_artifact(str(raw_path), artifact_path="raw")
        mlflow.log_dict(contract.contract_as_dict(), "data_contract.json")
        mlflow.log_dict(manifest, "official_manifest.json")
    ws.official_run_id = run.info.run_id
    _ok(f"Official datasets published (run {run.info.run_id[:8]}…)")
    return run.info.run_id


def publish_reference_model(ws, force=False):
    import mlflow
    mlflow.set_experiment(ws.experiment("training"))
    if not force:
        runs = mlflow.search_runs(experiment_names=[ws.experiment("training")],
                                  filter_string="tags.role = 'reference' AND attributes.status = 'FINISHED'",
                                  max_results=1)
        if not runs.empty:
            _ok(f"Reference model already published (run {runs.iloc[0].run_id[:8]}…): skipped")
            return runs.iloc[0].run_id
    datasets = official.read_repo_copy(ws.data_dir)
    model, metrics, train, _ = modeling.train_reference(datasets)
    run_id = modeling.log_reference(model, metrics, train, alias=ws.alias)
    _ok(f"Reference model published (run {run_id[:8]}…, ROC-AUC {metrics['roc_auc']:.3f})")
    return run_id


def prepare_workspace(ws, force=False):
    """Run every step. Returns a dict with the run IDs."""
    if not ws.online:
        print("⚠️  Not connected to Databricks: preparing the LOCAL MLflow store instead. "
              "Check your config and re-run the setup cell to prepare your Databricks workspace.")
    _ok(f"Folder and experiments ready under {ws.experiment_root}")
    official_run = publish_official(ws, force)
    reference_run = publish_reference_model(ws, force)
    if ws.online:
        print("\nYour experiments:")
        for key in ("official", "cleaning", "training", "monitoring"):
            print(f"   {ws.experiment(key):45} {ws.experiment_url(key)}")
    print("\nDone. You can now run notebooks 01, 02 and 03.")
    return {"official_run_id": official_run, "reference_run_id": reference_run}
