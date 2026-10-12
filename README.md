# ProFast Delivery Delay Prediction

A reproducible Python research prototype that ranks orders by delivery-delay risk
**at booking time**, after a promised delivery date has been assigned. Inspired by
ProFast, an undergraduate MERN package-management system; integration or live
operational use is not claimed.

The real-data experiment uses the **Brazilian E-Commerce Public Dataset by Olist**.
It compares a majority-class baseline, logistic regression, and a random forest.
See the [generated results](reports/results_summary.md) for measured performance
and the [technical report](reports/technical_report.md) for interpretation.

## What the experiment establishes

- One modelling row per order, with audited booking-time feature assumptions.
- Calendar-date lateness as the primary label; delivery on the promised day is on time.
- Training-fitted preprocessing inside sklearn pipelines.
- Chronological outer splits and inner cross-validation with actual delivery-time
  label-maturity checks. Identical booking timestamps stay together.
- Model selection by validation average precision and a validation-selected F1 threshold.
- Test diagnostics only after automated model/threshold selection is frozen.
- Error analysis, subgroup metrics, calibration diagnostics and bootstrap intervals.

This is a **corrected retrospective rerun**: historical test outcomes were already
inspected during project development. The code excludes them from selection, but
they are not a new confirmation set. Further rolling-origin and prospective
validation are needed. The model's scores are **uncalibrated**, and the demo displays
risk scores without probability claims or customer-contact recommendations.

## Setup — Windows PowerShell

Python 3.12 is the tested interpreter. Run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS users can use `python3.12 -m venv .venv` and `.venv/bin/python`.
The exact pinned versions are in `requirements.txt`. Installation requires network
access; tests themselves require no external dataset or network.

## Reproduce the real Olist experiment

```powershell
.\.venv\Scripts\python.exe -m src.download
.\.venv\Scripts\python.exe -m src.prepare
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
.\.venv\Scripts\python.exe -m pytest
```

The default configuration requires real data and fails clearly if it is absent.
If the public download is unavailable, download from
[the publisher page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and extract CSVs to `data/raw/olist/`. Do not add credentials to this repository.
The data is published under **CC BY-NC-SA 4.0**; verify the publisher's terms before
reuse. Raw data and fitted model binaries are excluded from git.

## Explicit synthetic software demonstration

This uses separate `demo/` output directories and cannot overwrite the published
real-data results. Synthetic scores demonstrate software behavior only.

```powershell
.\.venv\Scripts\python.exe -m src.prepare --config config/synthetic.yaml
.\.venv\Scripts\python.exe -m src.train --config config/synthetic.yaml
.\.venv\Scripts\python.exe -m src.evaluate --config config/synthetic.yaml
$env:PROFAST_CONFIG = "config/synthetic.yaml"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
Remove-Item Env:PROFAST_CONFIG
```

`PROFAST_CONFIG` selects the app's configuration; CLI tools use `--config`.
The app's defaults and choices come from training inputs saved with the model,
not from test-set distributions. A sampled test row is available for demonstration;
its predictions are not a new evaluation experiment.

## Exploratory calibration and rule comparisons

`src.evaluate` also runs `src.experiments`, which can be rerun explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.experiments
```

Isotonic calibration is fitted on saved **validation predictions only**. AP and
ROC-AUC are recomputed after calibration because isotonic mappings can introduce
ties. A fixed seven-day promise rule and fixed model/random top-up budgets are
compared retrospectively. Random top-ups use their exact sampling expectation.

Outputs: `reports/exploratory_summary.json`, `exploratory_policies.csv`, and
`exploratory_calibrated_predictions.csv`. These diagnostics do not replace the
serving bundle and do not establish business profit. The costs are hypothetical
classification penalties, not measured intervention effects. Any proposed rule
requires further backtesting and a controlled pilot.

## Project layout and outputs

| Location | Purpose |
| --- | --- |
| `src/data_prep.py`, `features.py` | Joins, label, audit, preprocessing and models |
| `src/splitting.py`, `training.py` | Temporal evaluation, mature CV, selection and saved bundle |
| `src/evaluation.py`, `evaluate.py`, `plots.py` | Metrics, error analysis and figures |
| `src/experiments.py` | Exploratory calibration and policy comparisons |
| `src/reporting.py` | Data/model cards and technical report generated from artifacts |
| `models/best_model.joblib`, `training_inputs.csv` | Trusted local serving artifacts; excluded from git |
| `reports/run_manifest.json`, `hyperparameter_search.json` | Provenance, configuration and per-CV-fold maturity audit |
| `reports/validation_predictions.csv`, `test_predictions.csv` | Saved scores for reproducible diagnostics |
| `tests/` | Leakage, boundary, maturity, experiment and isolated Streamlit runtime checks |
| `docs/` | Interview preparation, learning guide and factual CV wording |

Generated report values should be changed by rerunning the pipeline, not by editing
them manually. Model artifacts are pickle-based: load only trusted local artifacts.

## Limitations

The dataset describes 2016–2018 Brazilian marketplace orders, not Bangladesh
courier operations. Completed-order filtering and label-maturity filtering introduce
selection bias. Checkout availability of historical catalog/item fields is assumed,
not proven by a live snapshot. Missing physical measurements need further treatment.
Row-bootstrap intervals do not capture all temporal/group dependence. A single
historical split, hypothetical costs, and uncalibrated scores are insufficient for
production use.
