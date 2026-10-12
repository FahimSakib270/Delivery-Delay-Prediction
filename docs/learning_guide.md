# Learning guide

## Day 1 — understand the prediction and the data

1. Read the README and data card. Explain what is predicted, when it is predicted,
   how late deliveries are labelled, and which orders are excluded.
2. Follow one order through `src/data_prep.py`. Identify the order/item join keys,
   aggregation rules, missing measurements and prohibited future features.
3. Inspect `reports/feature_audit.csv`. Explain the assumed checkout availability
   of each retained feature and why retrospective snapshots remain a limitation.
4. Read `src/features.py`. Explain imputation, numeric scaling, categorical encoding,
   logistic regression and random forest in your own words.
5. Read `src/splitting.py` and its tests. Draw a boundary at noon and show why an
   order delivered later that day cannot contribute a known training outcome.
6. Run the isolated synthetic workflow in the README. Synthetic data tests software;
   it does not establish performance on real deliveries.

## Day 2 — understand evaluation and reproduce it

1. Read `src/training.py` and the per-fold audit in `hyperparameter_search.json`.
   Trace CV tuning, validation model selection and threshold freezing.
2. Reproduce the real-data workflow. Read generated metrics instead of manually
   replacing a number that looks disappointing.
3. Explain precision and recall using the confusion counts in `results_summary.md`.
   Compare AP to the delayed-order prevalence and accuracy to the dummy baseline.
4. Inspect subgroup errors and permutation importance. Check small sample sizes,
   temporal differences, and the distinction between association and causation.
5. Read `src/experiments.py`. Locate exactly which rows fit isotonic regression.
   Explain why ties can change AP/AUC and why the analysis remains exploratory.
6. Run pytest. The Streamlit runtime tests build their own temporary artifacts and
   verify the score display without relying on real data or pre-existing models.
7. Practice the interview questions. Use the current manifests for parameters and
   results, and describe what you actually understand and verified.

## Concepts to be able to defend

- Prediction time versus label availability time.
- Booking-time feature availability and future-feature leakage.
- Chronological splitting, grouped timestamp boundaries and mature CV folds.
- Validation selection versus retrospective test diagnostics.
- AP versus trapezoidal PR-AUC; ROC-AUC versus operational precision.
- Ranking scores versus calibrated probabilities versus business decisions.
- Row-bootstrap assumptions, selection bias and temporal distribution shift.
- Hypothetical error penalties versus measured intervention benefit.

This is a corrected retrospective study. Do not call the previously inspected test
set a new untouched experiment. A new prospective holdout and rolling-origin
backtesting are necessary next steps for stronger evidence.
