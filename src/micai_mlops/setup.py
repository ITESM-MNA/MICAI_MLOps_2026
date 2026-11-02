"""Workshop setup: config download, MLflow connection and safe local fallback.

Participants call `setup_workshop(config_link, nickname, notebook)` from the
first cell of each notebook. The function never prints or logs the token.
If anything fails, it switches to a local MLflow store so the notebook still
runs end to end; only the shared leaderboard is missing in that case.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests

EXPERIMENT_ROOT_DEFAULT = "/Shared/MICAI_MLOps_2026"
EXPERIMENTS = {
    "official": "00_official_datasets",
    "cleaning": "01_data_cleaning",
    "training": "02_training",
    "monitoring": "03_drift_monitoring",
}
NOTEBOOK_EXPERIMENT = {"01": "cleaning", "02": "training", "03": "monitoring"}
SCHEMA_VERSION = 1

_DRIVE_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"drive\.google\.com/(?:open|uc)\?(?:.*&)?id=([A-Za-z0-9_-]+)"),
]


@dataclass
class Workshop:
    mode: str                      # "databricks" or "local"
    alias: str
    notebook: str
    repo_dir: Path
    experiment_root: str
    official_run_id: str | None = None
    host: str | None = None
    experiment_ids: dict = field(default_factory=dict)
    message: str = ""

    @property
    def online(self):
        return self.mode == "databricks"

    @property
    def data_dir(self):
        return self.repo_dir / "data" / "wine-quality"

    def experiment(self, key):
        """Full MLflow experiment name, e.g. ws.experiment('training')."""
        return f"{self.experiment_root}/{EXPERIMENTS[key]}"

    def experiment_url(self, key):
        exp_id = self.experiment_ids.get(key)
        if self.online and exp_id:
            return f"{self.host}/ml/experiments/{exp_id}"
        return None

    def run_url(self, key, run_id):
        base = self.experiment_url(key)
        return f"{base}/runs/{run_id}" if base else None


# --------------------------------------------------------------------------- helpers

def _say(ok, text):
    print(("✅ " if ok else "⚠️  ") + text)


def clean_alias(nickname):
    alias = re.sub(r"[^a-z0-9_-]+", "-", (nickname or "").strip().lower()).strip("-")[:24]
    return alias


def _ask(prompt):
    try:
        return input(prompt)
    except Exception:          # non-interactive execution (CI, nbclient)
        return ""


def direct_download_url(url):
    """Turn a Google Drive share link into a direct-download link."""
    for pattern in _DRIVE_PATTERNS:
        match = pattern.search(url)
        if match:
            return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url


def fetch_config(link, timeout=20):
    """Return the config dict from a URL, a Google Drive link, a short link or a local path."""
    link = (link or "").strip()
    if not link:
        raise ValueError("no config link was given")
    if os.path.exists(link):
        return json.loads(Path(link).read_text())
    if not link.startswith(("http://", "https://")):
        link = "https://" + link
    try:
        response = requests.get(direct_download_url(link), timeout=timeout, allow_redirects=True)
        final_direct = direct_download_url(response.url)
        if final_direct != response.url:            # short link that redirected to a Drive page
            response = requests.get(final_direct, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        raise ConnectionError(f"could not download the config link ({type(exc).__name__}); "
                              "check the link and the internet connection") from None
    if response.status_code != 200:
        raise ValueError(f"the config link answered HTTP {response.status_code}; check the link")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError("the link did not return the config file "
                         "(is it shared as 'Anyone with the link'?)") from None


def check_config(cfg):
    required = ["schema_version", "databricks_host", "databricks_token", "experiment_root"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"config is missing {missing}")
    if int(cfg["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"config schema_version {cfg['schema_version']} is not {SCHEMA_VERSION}")
    expires = cfg.get("expires")
    if expires and date.fromisoformat(expires) < date.today():
        raise ValueError(f"the workshop config expired on {expires}")


def _quiet_mlflow():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    # Fail fast instead of retrying for minutes when the network is down
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    import logging
    import warnings
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="mlflow")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def preflight(cfg, timeout=10):
    """Fast check of the server and the key with a plain HTTP call.

    MLflow's Databricks client retries unreachable hosts for minutes, so we
    first make sure the server answers and accepts the key."""
    host = cfg["databricks_host"].rstrip("/")
    name = f"{cfg['experiment_root']}/{EXPERIMENTS['training']}"
    try:
        response = requests.get(f"{host}/api/2.0/mlflow/experiments/get-by-name",
                                params={"experiment_name": name}, timeout=timeout,
                                headers={"Authorization": f"Bearer {cfg['databricks_token']}"})
    except requests.RequestException as exc:
        raise ConnectionError(f"cannot reach {host} ({type(exc).__name__})") from None
    if response.status_code in (401, 403):
        raise PermissionError("the access key was rejected (expired or revoked?)")
    if response.status_code not in (200, 404):
        raise ConnectionError(f"unexpected answer from the server (HTTP {response.status_code})")


def _with_time_limit(fn, seconds):
    """Run fn() in a background thread; raise TimeoutError if it takes too long."""
    import threading
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:   # noqa: BLE001 - re-raised below
            result["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"the MLflow server did not answer within {seconds} s")
    if "error" in result:
        raise result["error"]
    return result["value"]


def _connect_databricks(cfg, time_limit=60):
    import mlflow
    preflight(cfg)
    os.environ["DATABRICKS_HOST"] = cfg["databricks_host"].rstrip("/")
    os.environ["DATABRICKS_TOKEN"] = cfg["databricks_token"]
    os.environ["DATABRICKS_AUTH_TYPE"] = "pat"
    mlflow.set_tracking_uri("databricks")
    return _with_time_limit(lambda: _ensure_experiments(cfg), time_limit)


def _ensure_experiments(cfg):
    import mlflow
    ids = {}
    for key in EXPERIMENTS:
        name = f"{cfg['experiment_root']}/{EXPERIMENTS[key]}"
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
        else:
            exp_id = exp.experiment_id
        ids[key] = exp_id
    return ids


def _connect_local(repo_dir):
    import mlflow
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_AUTH_TYPE"):
        os.environ.pop(var, None)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    store = Path(os.environ.get("MICAI_LOCAL_MLRUNS", Path(repo_dir) / "mlruns")).resolve()
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.as_uri())
    ids = {}
    for key in EXPERIMENTS:
        name = f"{EXPERIMENT_ROOT_DEFAULT}/{EXPERIMENTS[key]}"
        exp = mlflow.get_experiment_by_name(name)
        ids[key] = exp.experiment_id if exp else mlflow.create_experiment(name)
    return ids, store


# --------------------------------------------------------------------------- main entry

def setup_workshop(config_link="", nickname="", notebook="01", repo_dir="."):
    """Connect this notebook to the shared MLflow experiments (or fall back to local)."""
    _quiet_mlflow()
    import mlflow

    repo_dir = Path(repo_dir).resolve()
    print(f"MICAI 2026 · MLOps Workshop · Notebook {notebook} setup")

    alias = clean_alias(nickname)
    if not alias:
        alias = clean_alias(_ask("Type a nickname for the leaderboard (no real names): "))
    if not alias:
        alias = "guest-" + os.urandom(2).hex()
    _say(True, f"Nickname: {alias}")

    link = (config_link or "").strip() or os.environ.get("MICAI_CONFIG", "").strip()
    if not link and not os.environ.get("MICAI_OFFLINE"):
        link = _ask("Paste the config link from the slide (or press Enter to work offline): ").strip()

    ws = None
    if link:
        try:
            cfg = fetch_config(link)
            check_config(cfg)
            _say(True, "Config downloaded and checked")
            ids = _connect_databricks(cfg)
            ws = Workshop(mode="databricks", alias=alias, notebook=notebook, repo_dir=repo_dir,
                          experiment_root=cfg["experiment_root"],
                          official_run_id=cfg.get("official_data_run_id") or None,
                          host=cfg["databricks_host"].rstrip("/"), experiment_ids=ids)
            _say(True, "Connected to the shared MLflow server on Databricks")
        except Exception as exc:     # never stop the workshop because of the network
            reason = str(exc).split("\n")[0][:160]
            for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_AUTH_TYPE"):
                os.environ.pop(var, None)
            _say(False, f"Could not use the shared MLflow server: {reason}")

    if ws is None:
        ids, store = _connect_local(repo_dir)
        ws = Workshop(mode="local", alias=alias, notebook=notebook, repo_dir=repo_dir,
                      experiment_root=EXPERIMENT_ROOT_DEFAULT, experiment_ids=ids)
        _say(True, f"Working with a LOCAL MLflow store ({store}). Everything runs; "
                   "your runs just won't appear on the shared leaderboard.")

    mlflow.set_experiment(ws.experiment(NOTEBOOK_EXPERIMENT[notebook]))
    if ws.online:
        print(f"   MLflow UI for this notebook: {ws.experiment_url(NOTEBOOK_EXPERIMENT[notebook])}")
    sys.path.insert(0, str(repo_dir / "src"))
    print("Ready. Continue with the next cell.")
    return ws
