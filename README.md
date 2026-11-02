# MICAI 2026 · MLOps Workshop

**Chihuahua, Mexico · November 2, 2026**

From raw data to a monitored model in three Google Colab notebooks, using one dataset (*Wine Quality*, white wine)
and one shared MLflow server on Databricks.

| # | Notebook | Question it answers | Open |
|---|---|---|---|
| 1 | Data quality, EDA and dataset versioning | Can we trust the data? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/01_data_quality_eda_versioning.ipynb) |
| 2 | Model training and experiment tracking | Which model, trained on which data version, works best? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/02_training_tracking.ipynb) |
| 3 | Model monitoring and data drift | Is the model still valid once the data changes? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ITESM-MNA/MICAI_MLOps_2026/blob/main/notebooks/03_monitoring_drift.ipynb) |

## How to start (about 1 minute)

1. Click **Open in Colab** for notebook 1. You need a Google account; nothing else to install.
2. Run the first code cell (▶). When asked, paste the **config link** shown on the slide and type a **nickname**
   (no real names).
3. Wait for the ✅ messages, then run the cells from top to bottom (**Shift + Enter**).
4. For notebooks 2 and 3, repeat steps 1–3 with the same link and nickname.

If the setup cannot reach the shared server, it says so and continues with a local MLflow store: the notebook still
works; only the shared leaderboard is missing. To keep your changes, use **File → Save a copy in Drive**.

## What happens in each notebook

1. **Data quality, EDA and versioning.** A deliberately corrupted copy of the dataset is diagnosed with nine tests,
   cleaned with a pipeline of small functions (two strategies: drop or impute), validated against a data contract,
   explored with EDA, and versioned with MLflow dataset tracking.
2. **Training and tracking.** Random forests are trained on the official dataset versions, evaluated on a fixed
   holdout without leakage, and every run is tracked in MLflow. A class leaderboard compares the results.
3. **Monitoring and drift.** The production model is loaded from MLflow; a Gaussian Mixture Model simulates a
   miscalibrated lab instrument; the drift is detected with the Kolmogorov–Smirnov test and the Population Stability
   Index; the impact on the model is measured and the monitoring report is logged.

## Repository layout

```
notebooks/            the three workshop notebooks
src/micai_mlops/      shared code: setup, data contract, cleaning, official datasets, splits, model helpers
data/wine-quality/    original.csv, modified.csv (corrupted copy), metadata.json, official_manifest.json
requirements-colab.txt   installed by the setup cell in Colab
requirements-dev.txt     to run everything locally
scripts/run_notebooks.py runs the three notebooks offline (local MLflow)
```

**Run locally:** `pip install -r requirements-dev.txt`, then `python scripts/run_notebooks.py`, or open the notebooks
in Jupyter from the `notebooks/` folder.

## Credits

- **Dataset:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009). *Modeling wine preferences by data
  mining from physicochemical properties.* Decision Support Systems 47(4), 547–553. Obtained from the UCI Machine
  Learning Repository. `modified.csv` is a deliberately corrupted copy prepared for this workshop.
- **Tools:** Python, pandas, scikit-learn, SciPy, MLflow, Databricks and Google Colab.

**Declaration on the use of AI.** These materials were developed with the assistance of generative AI (Anthropic's
Claude, model Claude Opus 5.5), used to draft and review code, explanations and documentation. The workshop instructors
reviewed and validated all content and are responsible for it.
