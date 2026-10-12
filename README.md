# ProFast - Delivery Delay Prediction at Booking Time

Predict, **at the moment an order is booked**, whether it will be delivered *after* the
delivery date the customer was promised - and turn that probability into a decision the
business can defend.

A complete, reproducible, laptop-sized machine-learning project built on the public
**Brazilian E-Commerce Public Dataset by Olist** (99,441 real orders), with a
deterministic synthetic fallback for machines that cannot download it.

---

## The question, stated precisely

> Given only the information that exists when an order is placed, what is the
> probability that the parcel arrives **later than the promised delivery date**?

Three consequences follow from the words "when an order is placed", and they shape the
entire project:

1. **No delivery timestamps may be used as inputs.** The carrier hand-over date, the
   delivery date and the order status are future events. They define the *label* and the
   *label-maturity filter* and nothing else. `src/data_prep.py:PROHIBITED_MODEL_COLUMNS`
   is the testable list that enforces this, and `tests/test_data_prep.py` fails if a
   single one of them ever appears in the feature set.
2. **The promise itself is an input.** `lead_time_days` (promised date minus booking date)
   is known at checkout and is the most informative booking-time feature. The model
   therefore learns "orders *like this one* were late", not "this order will be late".
3. **A probability is not a decision.** The output is used at a threshold chosen on
   validation data, frozen, and then applied once to the test period. The cost of a miss
   and the cost of a false alarm are business assumptions, held in the config, never in
   the model.

## What the pipeline produces

| artefact | what it is |
| --- | --- |
| `data/processed/orders_prepared.csv` | one row per order: 18 booking-time features + label |
| `models/best_model.joblib` | the frozen pipeline **and** its frozen threshold, as one bundle |
| `reports/results_summary.md` | the results, generated from the run's own artefacts |
| `reports/evaluation_summary.json` | every number the app and the report quote |
| `reports/figures/*.png` | 13 figures (curves, calibration, subgroups, cost, intervals) |
| `reports/run_manifest.json` | environment, git commit, resolved config - the reproducibility record |
| `reports/data_card.md`, `reports/model_card.md` | the dataset and the model, documented |
| `reports/technical_report.md` | the full write-up, including what does *not* work |

## Results at a glance (real Olist data)

Produced by the frozen run in this repository: 96,470 orders, 6.77% late, chronological
70/15/15 split, threshold chosen on validation and applied once to the test period
(14,471 orders, Jun-Aug 2018, 4.28% late).

| | value |
| --- | --- |
| average precision (test) | **0.0711** [0.0619, 0.0829] - a 1.66x lift over the 0.0428 base rate |
| ROC-AUC (test, flagged optimistic) | 0.6475 [0.6276, 0.6652] |
| precision / recall at the frozen threshold | 0.0625 / 0.3500 (24% of orders flagged) |
| calibration error (ECE) | 0.4736 - the scores are inflated by the class weighting; isotonic calibration fitted on validation cuts this to 0.0203 with identical ranking |
| selected model | logistic regression (a 400-tree forest had a better CV score and lost on validation, then scored ROC-AUC 0.4955 on test - chance) |

**The honest headline: the ranking skill is real but, at the assumed costs, the decision is
not worth making at a global threshold.** A missed delay is assumed to cost 25 and a false
alarm 4, which puts the break-even precision at 13.79%. The model delivers 6.25%, so flagging
at its frozen threshold costs 1,595.95 per 1,000 orders against 1,071.11 for doing nothing.

A one-line rule beats it: flagging orders whose **promised lead time is 7 days or fewer**
covers 8.7% of orders at 16.9% precision and **saves 78.57 per 1,000**. Used as a *top-up*
inside that risky population, the model does add something - 14.9% precision against 13.2%
for random picks at the same budget - so it is kept as a monitored challenger, not a product.

`reports/technical_report.md` has the full analysis, including the cost tables, the subgroup
failures and the calibrated sensitivity. `reports/model_card.md` and `reports/data_card.md`
carry the model and data documentation.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python -m src.prepare               # raw CSVs -> modelling table + audit
python -m src.train                 # tune, select, freeze the threshold, score test once
python -m src.evaluate              # uncertainty, subgroups, calibration, 13 figures
streamlit run app/streamlit_app.py  # interactive demo
python -m pytest                    # the whole test suite, no network needed
```

Runtime on a 4-core laptop: `prepare` about 40 s, `train` 6-9 min (a 400-tree forest is
cross-validated over four configurations and spends 5.5 min of that losing), `evaluate` 3-5 min,
`pytest` about 40 s.

## Getting the real data

The public dataset is used as-is; nothing is scraped and nothing is re-hosted.

```python
import urllib.request
urllib.request.urlretrieve(
    "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce",
    "olist.zip",
)
```

Unzip the nine CSVs into `data/raw/olist/`. The pipeline reads
`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_products_dataset.csv`,
`olist_customers_dataset.csv` and `olist_sellers_dataset.csv`, plus the optional payments
and category-translation tables. `data_source.mode: auto` then uses the real files and says
so in every artefact; `mode: synthetic` (or an absent `data/raw/olist/`) falls back to the
generator in `src/synthetic.py`.

Licence: the dataset is published by Olist on Kaggle under **CC BY-NC-SA 4.0**
(non-commercial, share-alike). Check the licence before any commercial use - this
repository provides only the code. `reports/data_card.md` records the exact file checksums
of the shipped run.

## How the honesty is enforced (not just claimed)

* **Leakage guard** - a prohibited-column list plus a test that fails if any outcome
  column reaches the model.
* **Time, never shuffling** - the split is a cut on `order_purchase_timestamp`: every
  training order was booked before every validation order, which was booked before every
  test order. `tests/test_splitting.py` asserts it.
* **Label maturity** - a training row is kept only if its delivery was already known when
  the next period began, which removes the look-ahead bias of a naive "all rows are
  labelled today" split.
* **Selection discipline** - hyper-parameters and the decision threshold are chosen on the
  training/validation periods only; the test period is scored **once**.
* **Uncertainty is reported** - bootstrap intervals accompany every headline metric, and a
  resample that cannot support a metric is counted rather than hidden.
* **Minority-class metric** - average precision (PR-AUC) drives selection; ROC-AUC is
  printed but flagged as optimistic at a ~7% base rate, and accuracy is shown only to
  demonstrate how uninformative it is.
* **The cost model is an assumption** - the app prints the break-even probability implied
  by the cost inputs, so a reader can disagree with the assumption without
  reverse-engineering it.
* **Synthetic data is labelled** - if the fallback is used, the app shows a warning banner
  *before* any number, and the generator's fingerprint is stored so a changed generator
  cannot silently reuse stale CSVs.

## Repository layout

```
config/config.yaml        every tunable decision, in one reviewable file
src/config.py             config loading and path resolution
src/utils.py              seeding, logging, JSON IO, fingerprints, provenance
src/synthetic.py          deterministic Olist-shaped generator (the fallback)
src/data_prep.py          cleaning, joins, the label, the leakage list
src/features.py           feature audit, preprocessor, model factory
src/splitting.py          chronological split and label-maturity filter
src/evaluation.py         metrics, threshold selection, bootstrap, subgroups, cost
src/training.py           candidate tuning, selection, the frozen bundle
src/plots.py              13 figures
src/prepare.py|train.py|evaluate.py   the three CLI entry points
app/streamlit_app.py      the demo (three tabs, artefact-driven)
tests/                    leakage, dtypes, determinism, pipeline and app-runtime tests
docs/                     interview preparation, CV entry, learning guide
reports/                  generated artefacts + data card, model card, technical report
```

## Documentation

* `reports/technical_report.md` - the full study: method, results, limitations, what was
  tried and rejected.
* `reports/data_card.md` - the dataset, the label, the excluded rows and why.
* `reports/model_card.md` - the model, its metrics with uncertainty, its intended use.
* `docs/interview_preparation.md` - the questions this project invites, with answers.
* `docs/cv_project_entry.md` - a two-line and a one-paragraph CV formulation.
* `docs/learning_guide.md` - the concepts, in the order they are needed to rebuild this.
