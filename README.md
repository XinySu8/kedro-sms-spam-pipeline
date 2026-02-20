# kedro-sms-spam-pipeline

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

A lightweight ML-systems practice project: build a reproducible, configuration-driven end-to-end ML pipeline for the SMS Spam dataset using Kedro (ingest → clean → split → TF-IDF → train → evaluate).

## What this repo demonstrates
- Configuration-driven datasets (Kedro Data Catalog) instead of hard-coded file paths
- A reproducible end-to-end pipeline with stable schema and fixed random seeds
- Pipeline modularization (run a single node or the whole pipeline)
- Clear local artifacts: model + vectorizer + metrics report (not committed)

## Project structure
- Raw data (local only): `data/01_raw/spam.csv` (NOT committed)
- Cleaned data artifact (local only): `data/02_intermediate/sms_spam_clean.parquet` (NOT committed)
- Train/val/test splits (local only): `data/03_primary/` (NOT committed)
- Model artifacts (local only): `data/06_models/` (NOT committed)
- Metrics report (local only): `data/08_reporting/metrics.json` (NOT committed)
- Pipeline code: `src/kedro_sms_spam_pipeline/pipelines/sms_spam`
- Configs: `conf/base/catalog.yml`, `conf/base/parameters.yml`

## Overview

This Kedro project was generated using `kedro 1.2.0` and extended with a minimal, strong baseline text classification workflow:
- Split dataset with a fixed random seed (reproducible)
- TF-IDF text featurization
- Logistic Regression classifier
- Evaluation report including accuracy, precision, recall, F1, and confusion matrix

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data or artifacts under `data/` to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```powershell
pip install -r requirements.txt
pip install -e.
```

(Optional) Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

## Data setup (local only)

Place the dataset here:

- `data/01_raw/spam.csv`

Notes:
- The dataset is **NOT** committed to GitHub.
- The Data Catalog is configured to read the CSV with `latin-1` encoding (see `conf/base/catalog.yml`).

## How to run your Kedro pipeline

Run the full pipeline:

```powershell
kedro run
```

Run only a specific node (example):
```powershell
kedro run --nodes=split_sms_spam_node
```

Run only the training/evaluation nodes (if `sms_spam_clean` already exists):
```powershell
kedro run --nodes=split_sms_spam_node,tfidf_featurize_node,train_baseline_model_node,evaluate_model_node
```

## Outputs (generated locally; NOT committed)

After `kedro run`, you should see:

Cleaned dataset:
- `data/02_intermediate/sms_spam_clean.parquet`

Train/val/test splits:
- `data/03_primary/X_train.parquet`
- `data/03_primary/X_val.parquet`
- `data/03_primary/X_test.parquet`
- `data/03_primary/y_train.parquet`
- `data/03_primary/y_val.parquet`
- `data/03_primary/y_test.parquet`

Model artifacts:
- `data/06_models/tfidf_vectorizer.pkl`
- `data/06_models/spam_classifier_model.pkl`

Evaluation report:
- `data/08_reporting/metrics.json`

## Results (example run)

Example test-set metrics from a successful run:
- Accuracy: 0.98296
- Precision (spam): 0.95139
- Recall (spam): 0.91946
- F1 (spam): 0.93515

Confusion matrix (rows=true label, cols=pred label):
```text
[[TN=959, FP=7],
 [FN=12, TP=137]]
```

## How to test your Kedro project
> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.

### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```powershell
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:
```powershell
kedro jupyter notebook
```

### JupyterLab

To use JupyterLab, you need to install it:
```powershell
pip install jupyterlab
```

You can also start JupyterLab:
```powershell
kedro jupyter lab
```
### IPython
And if you want to run an IPython session:
```powershell
kedro ipython
```

### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can use tools like `nbstripout`. For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

## Package your Kedro project

Further information about building project documentation and packaging your project:  
https://docs.kedro.org/en/stable/tutorial/package_a_project.html

## Next steps
- Add an inference entrypoint (CLI) to classify new messages using the saved model/vectorizer.
- Add a small toy dataset under `tests/` (or synthetic fixtures) so CI can run without the raw dataset.
- (Optional) Add GitHub Actions to run `pytest` on every push.
