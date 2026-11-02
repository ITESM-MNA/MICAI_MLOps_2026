"""Random forest training, evaluation and MLflow logging shared by the presenter
script and the offline fallback of notebook 03 (notebook 02 shows the same
steps inline)."""

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from . import splits
from .contract import FEATURES

# The "production" model monitored in notebook 03 (published by the presenter).
REFERENCE_VERSION = "v1_clean_drop"
REFERENCE_PARAMS = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5}
SKOPS_TRUSTED = ["sklearn.tree._tree.Tree"]
RANDOM_STATE = 42


def train_rf(params, X, y):
    model = RandomForestClassifier(**params, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    return model.fit(X, y)


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def pip_requirements():
    """Explicit requirements: faster than letting MLflow infer them."""
    import skops
    return [f"scikit-learn=={sklearn.__version__}", f"skops=={skops.__version__}"]


def train_reference(official):
    """Train the reference ('production') model on the official v1 training set."""
    train, report = splits.training_set(official[REFERENCE_VERSION], official["holdout"])
    X, y = splits.xy(train, FEATURES)
    Xh, yh = splits.xy(official["holdout"], FEATURES)
    model = train_rf(REFERENCE_PARAMS, X, y)
    return model, evaluate(model, Xh, yh), train, report


def log_reference(model, metrics, train, alias="presenter"):
    """Log the reference model as a run in the current (training) experiment."""
    import mlflow
    X, _ = splits.xy(train, FEATURES)
    with mlflow.start_run(run_name="reference-production-model") as run:
        mlflow.set_tags({"role": "reference", "alias": alias, "dataset_version": REFERENCE_VERSION})
        mlflow.log_params({**REFERENCE_PARAMS, "dataset_version": REFERENCE_VERSION,
                           "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.log_input(mlflow.data.from_pandas(train, name=f"{REFERENCE_VERSION}_train",
                                                 targets="quality"), context="training")
        mlflow.sklearn.log_model(model, name="model", input_example=X.head(3),
                                 skops_trusted_types=SKOPS_TRUSTED,
                                 pip_requirements=pip_requirements())
    return run.info.run_id
