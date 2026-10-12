# Model card - delivery-delay risk at booking time

## 1. Model details

| field | value |
| --- | --- |
| model | `logistic_regression` inside a scikit-learn `Pipeline` (median imputation + standardisation for numerics, most-frequent imputation + one-hot encoding for categoricals) |
| hyper-parameters | `C = 1.0`, `Solver` liblinear, `class_weight = "balanced"`, `max_iter = 1000` |
| artefact | `models/best_model.joblib` - a bundle holding the fitted pipeline, the frozen threshold, the training metadata and the dataset checksum |
| frozen threshold | **0.6300** (chosen on validation data by maximising F1 for the delayed class) |
| label | `is_late` = delivered calendar date strictly after the promised calendar date |
| training data | 64,430 real Olist orders booked 2016-09-15 to 2018-04-15 (`kind: olist`) |
| validation | 12,847 orders, 2018-04-15 to 2018-06-21 (used for tuning and for the threshold) |
| test | 14,471 orders, 2018-06-21 to 2018-08-29 (scored once, never used for a choice) |
| licence | code: this repository. Data: CC BY-NC-SA 4.0 (Olist, Kaggle) - check before commercial use |
| version | 1.0.0, seed 42, deterministic |

The bundle is self-describing: it stores the threshold, so the decision rule moves with the model
rather than living in the UI or in a notebook.

## 2. Intended use

**Intended.** Rank orders *at the moment of booking* by the risk that they arrive after the
promised date, so an operations team can decide which risky orders deserve a proactive contact.
Secondary use: quantify how much of the promise is slack (median promised lead time 24 days
against a median actual journey of 10), which is a promise-policy discussion the business should
be having anyway.

**Not intended.**

* **not a customer-facing promise.** The output is a risk score, not a delivery estimate, and must
  never be shown as "your parcel will be late";
* **not a pricing or eligibility input.** It must not change what a customer is charged, and it
  must not be used to deprioritise some customers' orders;
* **not a post-booking control tower.** The features exist at booking only; if the model is fed
  information from later in the order lifecycle, the serving inputs no longer match the training
  inputs and the metrics above no longer apply;
* **not valid outside this marketplace and period.** The regime shift documented below means the
  measured skill is period-specific.

## 3. Factors and features

18 features, all available at booking, enumerated with reasons in `reports/feature_audit.csv`.
The dominant one by a wide margin is `lead_time_days` (promised date minus booking date):
permutation importance 0.0315 against 0.0031 for the next feature. Geography (seller state,
customer state, same-state) and parcel weight follow.

Relevant segments, with their test-period late rates: destination state (SP 5.76%, PR 2.15%, MG
1.27%), promise length (`<=7` days 16.90%, 8-15 days 4.98%), route (same-state 6.51%), basket size
(2+ items 2.71%) and booking month (June 0.97%, July 3.38%, August 6.19%).

## 4. Metrics

Test period, frozen threshold, with 500-resample percentile bootstrap intervals:

| metric | value | 95% interval | interpretation |
| --- | ---: | --- | --- |
| average precision (primary) | **0.0711** | [0.0619, 0.0829] | base rate is 0.0428, so a **1.66x lift**; the interval excludes chance |
| ROC-AUC (reported, optimistic) | 0.6475 | [0.6276, 0.6652] | modest separation; inflated by the many easy on-time orders |
| precision | 0.0625 | - | **half of the 13.79% break-even** implied by the cost assumptions |
| recall | 0.3500 | - | 217 of 620 late orders flagged |
| F1 | 0.1061 | - | |
| Brier | 0.2847 | - | poor, because the scores are inflated (section 5) |
| accuracy | 0.7472 | - | reported only to show that 75% "accuracy" is worse than doing nothing |

Confusion at the frozen threshold: TP 217, FP 3,255, FN 403, TN 10,596.

**Baselines, same test period.** Doing nothing costs 1,071.11 per 1,000 orders; the model at the
frozen threshold costs 1,595.95 (-524.84). A one-line business rule - flag orders whose promised
lead time is 7 days or fewer - costs 992.54 (**+78.57**). The random forest candidate, which had
the best cross-validated score of the three, achieved a test ROC-AUC of 0.4955 (chance). See
`reports/technical_report.md` section 7.

## 5. Calibration - the model's main defect

Expected calibration error on the test period: **0.4736**. Bins tell the story: orders scored
0.5-0.6 were late 5.7% of the time; orders scored 0.8-0.9 were late 8.9% of the time.

The cause is `class_weight="balanced"`, which scales the minority class by roughly
89,936/6,534 = 13.8 and therefore shifts every predicted probability upward by about that odds
factor. This is a deliberate trade (it is what keeps recall usable at a rare base rate) but it must
be stated, because it means:

* the score is a **ranking device, not a probability** - do not read "0.43" as 43%;
* any decision rule written in probability units is wrong by roughly an order of magnitude in
  odds. Applying the cost break-even probability of 0.1379 to the raw scores flags 98.6% of orders
  and loses 2,704.79 per 1,000;
* the remedy is measured and cheap: isotonic regression fitted on validation scores only moves test
  ECE from 0.4736 to **0.0203** with identical ranking metrics. It is not in the shipped bundle
  because adding it after the single-shot test evaluation would invalidate the reported score; it
  is the first recommended change.

## 6. Decision rule and business case

The stored threshold (0.63) maximises validation F1. Under the assumed costs (a missed delay 25, a
false alarm 4) that rule **does not pay for itself**: -524.84 per 1,000 orders against doing
nothing, because precision (6.25%) is below the 13.79% break-even. The model becomes worth
deploying if the flag budget is cut to the riskiest ~9% of orders (the `<=7`-day segment clears
the bar on its own), or if the true cost ratio is above roughly 15:1 rather than the assumed
6.25:1.

This is the honest headline: **the ranking has real but small skill, and at these costs the
decision it supports is not worth making at a global threshold.**

## 7. Ethical and operational considerations

* **Fairness.** The two strongest non-promise features are geography (customer/seller state) and
  parcel weight. A rules-based intervention driven by them would systematically contact customers
  in slower regions more often. That may be justified operationally, but it should be a conscious
  policy choice, monitored per segment, not a side effect of a model. Per-segment recall is
  published in `reports/test_subgroups.csv` for exactly this reason.
* **No personal data.** Only an anonymised customer id, a city and a state are present; names,
  addresses, emails and payment details are absent from the feature set.
* **Automation bias.** A probability displayed next to an order invites over-trust. The demo app
  therefore prints the break-even probability, the cost assumption it implies, and an explicit list
  of what the page does not claim.
* **Feedback loops.** If interventions change delivery outcomes, future training data becomes a
  biased sample of the un-intervened world. Any deployment needs a holdout group precisely so the
  effect of the intervention remains measurable.
* **Cost of false alarms.** Contacting customers apologetically for an on-time parcel is not free
  even when the arithmetic says it is cheap: it can itself create a perception of unreliability.
  The 4-assumption is a placeholder that should be validated with the support and marketing teams.

## 8. Caveats for anyone reusing this

1. Trained and tested on 2016-2018 Brazilian marketplace data with a documented regime shift
   (Nov 2017-Mar 2018 at 12-15% late against 1.8-5% in mid-2018). Do not transfer the numbers.
2. The label means "later than the promise", not "slow". A long-promise order delivered late is
   still late; a short-promise order delivered quickly can be late.
3. Canceled and unresolved orders (3.0% of raw rows) are excluded, so the model does not cover the
   population a live system would also have to handle.
4. Metrics come from **one** chronological split. Rolling-origin evaluation is required before any
   production decision.
5. The probabilities are not calibrated (section 5).
6. The cost assumptions drive the entire business conclusion and are config parameters, not
   measurements.

## 9. Provenance and reproduction

* dataset checksums and environment: `reports/run_manifest.json`,
  `data/processed/dataset_metadata.json`;
* label and exclusion accounting: `reports/data_card.md`;
* full evaluation, including the negative results: `reports/technical_report.md`;
* rebuild:

```bash
python -m src.prepare && python -m src.train && python -m src.evaluate
```

* the app (`streamlit run app/streamlit_app.py`) reads only these artefacts, never re-trains, and
  shows a provenance banner before any number.
