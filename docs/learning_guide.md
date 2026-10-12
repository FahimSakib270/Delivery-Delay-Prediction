# Learning guide - how to rebuild this project, in the order it makes sense

The reading order that builds the project from nothing, with the *reason* each step exists.
Each section names the file that implements it and the check that proves it works.

---

## 0. The one decision everything else follows from

**When does the prediction happen?**

"Will this delivery be late?" has three possible answers depending on the moment you ask:

| moment | available information | who uses it |
| --- | --- | --- |
| at booking | basket, promise, route, season | **this project** (a decision before the customer is disappointed) |
| at dispatch | plus carrier hand-over | a logistics control tower |
| after delivery | everything | a post-mortem |

Choosing *booking time* removes every delivery timestamp from the feature set. That single
sentence is the difference between a model that supports a decision and a model that leaks
its own label. Write it down before writing code, because every later temptation ("this
column is so predictive!") is answered by it.

→ `config/config.yaml` header, `reports/data_card.md` section 5.

## 1. Get data you can defend

* Prefer real, public, licensed data over a synthetic toy: the messiness *is* the lesson.
* Record the licence, the source and the checksums of every file you read.
* Build a **fallback generator** so the project runs anywhere, but make it declare itself
  (provenance marker) and keep its numbers out of any claim about reality.
* A generator is a hypothesis about the world. The moment you tune it so your metrics look
  good, you have started modelling your own assumptions. The safe discipline is to use it for
  *software* correctness only, which is why the shipped artefacts here come from the real CSVs.

→ `src/synthetic.py`, `reports/data_card.md` sections 1 and 8.

## 2. Clean with a counted funnel, not a vibe

Every exclusion must be a row in a table: duplicates, missing mandatory dates, canceled
orders, unresolved deliveries, impossible timestamps. Two reasons: a reader can audit the
exclusions, and *you* notice when a "harmless" filter silently removes 30% of the data or,
worse, only removes rows from one time period.

→ `src/data_prep.py:clean_orders`, `reports/data_card.md` section 3.

## 3. Define the label in the customer's terms

"Late" here means *delivered after the promised calendar date*, because that is what the
customer was told; the time-of-day component of the promise is a system artefact. Keep the
alternative definition (raw timestamps) as a sensitivity check: if the two disagree about your
conclusions, that is a finding, not a nuisance.

→ `src/data_prep.py:add_target`, `reports/data_card.md` section 4.

## 4. Aggregate one-to-many tables without duplicating rows

Orders have many items and many payments, and their items many sellers. Joining naively
multiplies rows, which silently changes the class balance and the metric weights. Aggregate to
**one row per analysis unit** first, assert the index is unique, and keep the *primary* item's
attributes for categorical features.

→ `src/data_prep.py:aggregate_items`, `tests/test_data_prep.py`.

## 5. Write the feature audit *before* the feature list

For each candidate: what it means, where it comes from, whether it exists at the decision
moment, whether it is used, and why. The row "not knowable at booking" is the one that saves
you. Then enforce the conclusion in code as a `PROHIBITED_MODEL_COLUMNS` tuple and a test, so a
future refactor cannot quietly reintroduce leakage.

→ `src/features.py:FEATURE_AUDIT`, `src/data_prep.py:PROHIBITED_MODEL_COLUMNS`,
`tests/test_data_prep.py`.

## 6. Split by time, and be honest about maturity

A random split lets the model learn from the future. Sort by booking time and cut. Then ask the
harder question: *was the label even observable* when the next period started? If not, drop the
row from training (but never from test - the test period is the future by construction). This
is the difference between "not shuffling" and "not cheating".

→ `src/splitting.py`, `tests/test_splitting.py`.

## 7. Build the pipeline so it survives production

* Impute numeric values with the **training** median, not the global one.
* One-hot encode categories with `min_frequency` merging and `handle_unknown`, so a level
  invented after deployment does not crash the service.
* Scale only for models that need it (linear), not for trees.
* Wrap preprocessing and model in one object, so the artefact *is* the scoring path.

→ `src/features.py:build_preprocessor`, `build_model_pipeline`,
`tests/test_features.py::test_preprocessor_survives_categories_never_seen_during_training`.

## 8. Choose the metric that matches the rarity

At a ~7% base rate, accuracy is a trap: predicting "on time" for everything scores 93%. Use
**average precision** (area under the precision-recall curve) for ranking quality and
class-imbalance honesty, report **ROC-AUC** but flag that it is optimistic under imbalance, and
report **Brier** score because a probability that is not calibrated is not usable at a
threshold.

→ `src/evaluation.py:compute_metrics`, `reports/model_card.md`.

## 9. Tune with time-series cross-validation

`GridSearchCV` with `TimeSeriesSplit`: each fold trains on the past and validates on the
immediately following stretch. Keep the search space small and the *scoring* metric identical
to the one you report. Always include a **dummy** baseline fitted the same way - without it you
cannot tell 0.13 average precision from luck. Report the baseline's cross-validated score, not
a `nan`: a blank in a comparison table reads as "not evaluated", not "no skill".

→ `src/training.py:tune_model`, `src/training.py:fit_candidates`.

## 10. Freeze the threshold, then score once

The threshold is a *business* decision rule, so: choose it on validation data only (maximise
F1, or take the highest recall whose precision clears a floor), store it inside the model
bundle, and score the test period exactly once. Re-tuning on test data converts a metric into a
guess. Quantify what remains with a bootstrap over test rows, and report the interval next to
every headline number - including how many resamples were usable.

→ `src/evaluation.py:select_threshold`, `bootstrap_ci`, `src/training.py:train_and_evaluate`.

## 11. Judge the model, not just the average

A single number hides segment-wide failures. Slice the errors by destination state, seller
state, product category, lead-time bucket, basket size and booking month; flag the segments too
small to conclude from; and look for the worst *large* segment. Then check calibration: is a
0.30 score really followed by delays about 30% of the time?

→ `src/evaluation.py:subgroup_report`, `calibration_table`, `reports/technical_report.md`.

## 12. Turn the probability into a decision with explicit costs

A missed delay and an unnecessary proactive contact cost different amounts, so the optimal
threshold is neither 0.5 nor the F1-maximising one. Write the arithmetic down
(`cost = FN x c_fn + FP x c_fp`), plot cost against threshold, and state the break-even
probability `p* = c_fp / (c_fp + c_fn)`. Then label the costs for what they are: **assumptions
the business owns**, not measurements from the data.

→ `src/evaluation.py:expected_cost`, `cost_curve`, `src/plots.py:plot_cost_curve`.

## 13. Make the demo refuse to oversell

The app reads the artefacts, never re-trains, never touches the test set to choose anything,
and prints the provenance banner *before* any number. It also prints what it does *not* claim:
no per-order reason codes from a tree ensemble, no causal claim from a slider, no certainty
from a probability. The decision threshold comes from the saved bundle, not a UI control, so
what the demo shows is exactly what was evaluated.

→ `app/streamlit_app.py`, `tests/test_app_runtime.py`.

## 14. Test the things that break quietly

Unit tests for the helpers are the easy half. The valuable tests are the invariants:

* a prohibited column never reaches the model;
* the split never mixes past and future, and maturity actually removes rows;
* the generator is reproducible, and its seasonal term is not absurd;
* the frame the tests fit models on has the same dtypes production loads (Arrow-backed string
  columns containing `pd.NA` are a genuine scikit-learn crash source);
* the app renders all three tabs against the artefacts on disk.

→ `tests/`.

## 15. Write down what does not work

A project is judged more by its honesty than by its AUC. Record the rejected feature, the model
that lost, the subgroup that fails, the metric that is optimistic, and the assumption that would
have to be wrong for the recommendation to be wrong.

→ `reports/technical_report.md`, sections "What did not work" and "Limitations".

---

## Where the numbers come from

| question | artefact |
| --- | --- |
| dataset, label, exclusions | `reports/data_card.md`, `data/processed/dataset_metadata.json` |
| what the model achieved, with intervals | `reports/results_summary.md`, `reports/evaluation_summary.json` |
| per-model comparison across splits | `reports/model_comparison.csv` |
| why the threshold is what it is | `reports/validation_threshold_curve.csv` |
| what the model relies on | `reports/permutation_importance.csv` |
| where it fails | `reports/test_subgroups.csv`, `reports/error_examples_*.csv` |
| how to reproduce it | `reports/run_manifest.json` |
