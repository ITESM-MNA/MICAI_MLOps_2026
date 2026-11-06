# MICAI 2026 · MLOps Workshop

**Chihuahua, Mexico · November 2, 2026**

From raw data to a monitored model in Google Colab notebooks, using one dataset (*Wine Quality*, white wine) and an
MLflow server on Databricks.

> ## ⚠️ Before you start: set up your own MLflow
> The workshop's shared MLflow server is no longer available. To run the notebooks with MLflow tracking, first
> follow **[docs/SETUP_YOUR_OWN_MLFLOW.md](docs/SETUP_YOUR_OWN_MLFLOW.md)** (about 15 minutes, free): create a
> Databricks account and an access token, fill in your `config.json` from `config.template.json`, and store it as the
> Colab Secret `MICAI_CONFIG`. Then **run notebook 00 once**: it creates your experiments and publishes the official
> datasets and the reference model. Only after that, run notebooks 01–03.
>
> No account? Leave **CONFIG_LINK** empty and the notebooks run **offline** with a local MLflow store.

| # | Notebook | Question it answers | Open |
|---|---|---|---|
| 0 | **Set up your own MLflow workspace (run once, first)** | Is everything ready? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/00_setup_your_workspace.ipynb) |
| 1 | Data quality, EDA and dataset versioning | Can we trust the data? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/01_data_quality_eda_versioning.ipynb) |
| 2 | Model training and experiment tracking | Which model, trained on which data version, works best? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/02_training_tracking.ipynb) |
| 3 | Model monitoring and data drift | Is the model still valid once the data changes? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/03_monitoring_drift.ipynb) |

## How to run the notebooks

**On your own Databricks (recommended)**

1. Follow [docs/SETUP_YOUR_OWN_MLFLOW.md](docs/SETUP_YOUR_OWN_MLFLOW.md), steps 1–5: Databricks account, access token,
   `config.json`, Colab Secret `MICAI_CONFIG`.
2. Open **notebook 00** in Colab and run all cells **once**.
3. Open notebooks 01, 02 and 03 in order. In the first cell leave **CONFIG_LINK** empty, type a **nickname**, and run
   it. Wait for the ✅ messages, then run the cells from top to bottom (**Shift + Enter**).

Each notebook runs on its own Colab machine, so the first cell is repeated in every notebook. Switch on **Notebook
access** for the secret in each notebook (key icon 🔑). To keep your changes, use **File → Save a copy in Drive**.

**Offline (no account):** leave **CONFIG_LINK** empty without creating the secret, or press Enter when asked. Everything
runs with a local MLflow store; nothing is kept after the Colab session ends.

**In a workshop:** paste the config link given by the instructor in **CONFIG_LINK**; notebook 00 is not needed.

**On your computer:** `pip install -r requirements-dev.txt`, then open the notebooks in Jupyter from the `notebooks/`
folder and type the path to your `config.json` in **CONFIG_LINK**. To run all of them offline:
`python scripts/run_notebooks.py`.

## Your config file and its token

- `config.template.json` (in the repository) shows the fields. **Copy it to `config.json` and edit only the copy.**
- `config.json` is listed in `.gitignore`: Git never commits it, so your token stays on your computer. Check with
  `git check-ignore -v config.json`.
- Never put your token in the template, in a notebook, or behind a public link. Details:
  [docs/SETUP_YOUR_OWN_MLFLOW.md](docs/SETUP_YOUR_OWN_MLFLOW.md), step 4.

## What happens in each notebook

0. **Set up your workspace.** Creates the folder `/Shared/MICAI_MLOps_2026` and four experiments, publishes the
   official dataset versions (copied from `data/wine-quality/official/`) and the reference model.
1. **Data quality, EDA and versioning.** A deliberately corrupted copy of the dataset is diagnosed with nine tests,
   cleaned with a pipeline of small functions (two strategies: drop or impute), validated against a data contract,
   explored with EDA, and versioned with MLflow dataset tracking.
2. **Training and tracking.** Random forests are trained on the official dataset versions, evaluated on a fixed
   holdout without leakage, and every run is tracked in MLflow. A leaderboard compares the results.
3. **Monitoring and drift.** The production model is loaded from MLflow; a Gaussian Mixture Model simulates a
   miscalibrated lab instrument; the drift is detected with the Kolmogorov–Smirnov test and the Population Stability
   Index; the impact on the model is measured and the monitoring report is logged.

## Troubleshooting

| What you see | What to do |
|---|---|
| `Working with a LOCAL MLflow store` | The line above it names the reason; see the troubleshooting table in the [setup guide](docs/SETUP_YOUR_OWN_MLFLOW.md#troubleshooting) |
| `... has no access to it: ... 'Notebook access'` | Key icon 🔑 → switch on **Notebook access** for `MICAI_CONFIG` |
| `No official datasets found in MLflow` | Run notebook 00 once |
| `the access key was rejected` | Your token expired or was deleted: create a new one and update the secret |

## Repository layout

```
notebooks/               00 (setup, run once) and the three workshop notebooks
docs/                    SETUP_YOUR_OWN_MLFLOW.md: run the notebooks on your own Databricks
config.template.json     config fields; copy to config.json (ignored by Git) and fill in
src/micai_mlops/         shared code: setup, data contract, cleaning, official datasets, splits, models, workspace
data/wine-quality/       original.csv, modified.csv (corrupted copy), metadata.json,
                         official/ (processed official versions) and official_manifest.json (their fingerprints)
scripts/                 setup_workspace.py (notebook 00 from a terminal), run_notebooks.py (offline run)
requirements-colab.txt   installed by the setup cell in Colab
requirements-dev.txt     to run everything locally
```

## About the workshop

The materials were first presented at the MICAI 2026 MLOps Workshop in Chihuahua, Mexico, on November 2, 2026. The
exact version used that day is the Git tag `micai-2026-workshop`. Later changes let participants run the notebooks
independently on their own Databricks MLflow.

## Credits

- **Dataset:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009). *Modeling wine preferences by data
  mining from physicochemical properties.* Decision Support Systems 47(4), 547–553. Obtained from the UCI Machine
  Learning Repository. `modified.csv` is a deliberately corrupted copy prepared for this workshop.
- **Tools:** Python, pandas, scikit-learn, SciPy, MLflow, Databricks and Google Colab.

**Declaration on the use of AI.** These materials were developed with the assistance of generative AI (Anthropic's
Claude, model Claude Opus 5.5), used to draft and review code, explanations and documentation. The workshop instructors
reviewed and validated all content and are responsible for it.
