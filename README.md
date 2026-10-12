# ProFast Delivery Delay Prediction

A Python prototype that ranks orders by delivery-delay risk **at booking time**,
after a delivery date has been promised. Inspired by the ProFast undergraduate
package-management project, it includes a local Streamlit demo.

## Dataset and approach

The experiment uses public **Olist Brazilian e-commerce data (2016–2018)**,
with one row per order. An order is late if it arrives after the promised calendar
date; delivery on that date is on time. Models use booking-time information such
as promised lead time, order value, item details and customer/seller location.
Historical item and catalog fields are assumed to have been available at checkout.

A majority-class baseline, logistic regression and random forest are compared.
Delivery outcomes define labels and label-maturity checks, never model inputs.
Preprocessing is fitted on training data inside scikit-learn pipelines.

Evaluation uses chronological train/validation/test periods and chronological
cross-validation. Equal booking timestamps stay together. Training labels must
already be observable before the next evaluation period starts, including inside
cross-validation. Validation average precision selects the model; validation F1
selects its decision threshold. Both are frozen before test diagnostics.

## Actual results

The selected logistic regression was evaluated on **14,471 test orders**, including
620 late orders (4.28%), at a frozen threshold of **0.63**.

| Test metric | Result |
| --- | --- |
| Average precision | 0.0711 (baseline: 0.0428) |
| ROC-AUC | 0.6475 |
| Precision / recall | 0.0625 / 0.3500 |
| F1 | 0.1061 |

Ranking performance is modest. With hypothetical penalties of 25 per missed delay
and 4 per false alarm, the threshold costs **1,595.95 per 1,000 orders**, versus
**1,071.11** for flagging nothing. These penalties do not measure business profit
or intervention effects. Scores are uncalibrated risk scores, not probabilities.

See the [generated results](reports/results_summary.md) for model comparisons and
uncertainty intervals, and the [technical report](reports/technical_report.md) for
error analysis, calibration and exploratory rule comparisons.

## Run locally

Python 3.12 is tested. From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Linux/macOS: use `python3.12 -m venv .venv` and `.venv/bin/python`.
Tests need no external dataset or network.

To reproduce the real-data experiment (these commands regenerate reports):

```powershell
.\.venv\Scripts\python.exe -m src.download
.\.venv\Scripts\python.exe -m src.prepare
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Skip the download if CSVs already exist. If it fails, extract the Olist CSVs into
`data/raw/olist/` manually. See [dataset details](reports/data_card.md) for source,
checksums and the CC BY-NC-SA 4.0 license. The default configuration requires real
data; raw CSVs and fitted models are excluded from Git.

For a synthetic software demonstration, use the separate `demo/` outputs:

```powershell
.\.venv\Scripts\python.exe -m src.prepare --config config/synthetic.yaml
.\.venv\Scripts\python.exe -m src.train --config config/synthetic.yaml
.\.venv\Scripts\python.exe -m src.evaluate --config config/synthetic.yaml
$env:PROFAST_CONFIG = "config/synthetic.yaml"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
Remove-Item Env:PROFAST_CONFIG
```

Synthetic metrics are not evidence of real-world performance. CLI tools use
`--config`; the app uses `PROFAST_CONFIG`. The app's defaults come from training
inputs; a sampled test row is only a demo input. Evaluation also runs exploratory
calibration and rule comparisons; rerun those with `python -m src.experiments`.
Calibration uses validation predictions only and does not replace the serving model.

## Limitations and project files

This is a corrected retrospective rerun: historical test outcomes were inspected
during development, so the test period is not a fresh confirmation set. Completed
order and label-maturity filters introduce selection bias. Missing physical
measurements need more work; row-bootstrap intervals miss temporal/group dependence.
Brazilian marketplace results do not establish performance for Bangladesh couriers.
Rolling-origin evaluation and prospective validation are needed before operational use.

- `src/`: preparation, leakage guards, temporal splits, training and evaluation.
- `tests/`: leakage, boundary, maturity, diagnostics and app checks.
- `reports/`: generated results, figures, data/model cards and run manifests.
- `docs/`: learning guide, interview notes and [portfolio wording](docs/cv_project_entry.md).
- `models/`: local fitted bundle and training inputs; load only trusted artifacts.

Regenerate reports through the pipeline rather than editing measured values by hand.
