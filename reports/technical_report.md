# Technical report - delivery-delay prediction at booking time

**ProFast / Olist case study.** Everything below is generated from the artefacts of the run
that produced this repository: `reports/evaluation_summary.json`,
`reports/model_comparison.csv`, `reports/validation_threshold_curve.csv`,
`reports/permutation_importance.csv`, `reports/test_subgroups.csv` and the 13 figures in
`reports/figures/`. Nothing is quoted from memory.

---

## 1. Executive summary

**The model works; the business case does not - and both parts matter.**

* **Skill is real but modest.** On the untouched test period (14,471 orders, Jun-Aug 2018,
  4.28% late), the frozen model reaches average precision **0.0711 [0.0619, 0.0829]** against
  a base rate of 0.0428 - a 1.66x lift whose bootstrap interval excludes chance - and ROC-AUC
  **0.6475 [0.6276, 0.6652]**. At the validation-frozen threshold of 0.63 it flags 24% of
  orders with precision 6.25% and recall 35%.
* **The probabilities are not usable as probabilities.** ECE on the test period is **0.474**,
  because `class_weight="balanced"` inflates every score by roughly the class ratio. Ranking
  survives; "0.43 means 43% likely" does not. Isotonic calibration fitted on validation data
  alone brings test ECE to **0.020** without changing the ranking at all.
* **At the assumed costs, intervening loses money.** With a missed delay at 25 and a false
  alarm at 4, flagging at the frozen threshold costs **1,595.95 per 1,000 orders against
  1,071.11 for doing nothing** (-524.84). The break-even precision implied by those costs is
  **13.79%**, and the model delivers 6.25% - roughly half of what it would need.
* **A one-line rule beats the model at every comparable budget.** Flagging every order whose
  *promised* lead time is 7 days or fewer means flagging 8.7% of orders, of which **16.9%**
  are genuinely late: cost **992.54 per 1,000, a saving of 78.57** with no model at all. Given
  the same 8.7% budget, the model achieves 7.5% precision and loses 157.90.
* **Where the model does add something:** used as a *top-up* inside that risky population
  (rule plus the model's next-most-risky orders), precision at a 12% budget is 14.9% versus
  13.2% for random top-ups. So the ranking carries usable information; it is simply not strong
  enough, on this period, to justify a large flag budget at these costs.
* **The discipline still paid off.** The validation-frozen threshold lost 525 per 1,000; the
  same model at 0.5 would have lost 1,247, and applying the break-even probability naively to
  the uncalibrated scores would have flagged 98.6% of orders and lost 2,705.

The honest recommendation is therefore: **deploy the segment rule now, keep the model as a
monitored challenger with a capacity-based objective, and revisit it when (a) the business
supplies evidence for the cost ratio and (b) the probabilities are calibrated.**

## 2. The decision, and why the framing is the whole project

A late delivery is expensive in a way that is easy to underestimate: the customer has already
been told a date. Either the customer is surprised (a support ticket, a review, a
cancellation), or the business contacts them in advance with an apology and a voucher. Both
cost money; only the second is under the business's control, and it is only worth doing for
orders that are actually at risk.

So the decision is not "will this be late?" but:

> **Which orders, at booking time, are risky enough that a proactive contact is cheaper than
> silence?**

That reframing has three consequences a pure modelling exercise would miss:

1. the output must be a **probability**, not a label, because the decision compares expected
   costs (`p x c_miss` against `(1 - p) x c_false_alarm`);
2. the **threshold is a business parameter**, so it is chosen from costs, not from a metric;
3. the metric that matters is the model's ability to **rank the rare bad orders**, not its
   accuracy - at a 6.77% base rate, doing nothing is "accurate" 93% of the time.

Everything else in this report follows from taking that reframing seriously.

## 3. Data

The public **Brazilian E-Commerce Public Dataset by Olist** (99,441 orders, 2016-09 to 2018-08,
CC BY-NC-SA 4.0) is used as published, plus a deterministic synthetic fallback so the pipeline
runs offline. After cleaning, **96,470 orders** remain and **6.77%** were delivered after the
promised calendar date (8.11% under the stricter timestamp definition). Full row accounting,
checksums, the label definitions, the feature audit and the known biases are in
`reports/data_card.md`; that document is the reference for anything dataset-shaped here.

Three properties of the data drive the modelling choices:

* the base rate is low (6.77%), so the positive class is the tail of the process;
* booking volume and lateness are non-stationary - Nov 2017 to Mar 2018 runs at 12-15% late
  against 1.8-5% in mid-2018 - so a chronological split always mixes regimes;
* the promise is generous (median promised lead time 24 days against a median actual journey
  of 10 days), so this models *deviation from a promise*, not slowness.

## 4. Method

### 4.1 Leakage discipline

The prediction moment is booking, so the only admissible inputs are ones that exist then. The
project enforces this rather than documenting it:

* every candidate feature is classified in `src/features.py:FEATURE_AUDIT` with an
  "available at booking?" column and a reason;
* every rejected column is listed in `src/data_prep.py:PROHIBITED_MODEL_COLUMNS`;
* `tests/test_data_prep.py` fails if a prohibited column ever reaches the feature set, and if
  any *used* feature is marked as not available at booking.

Delivery timestamps therefore appear in exactly two places: the label and the label-maturity
filter. `order_status` - a tantalisingly predictive column - is prohibited outright because it
*is* the outcome.

### 4.2 Split and label maturity

The split is a cut on `order_purchase_timestamp` (70/15/15), never a shuffle:

| fold | booking window | rows |
| --- | --- | ---: |
| train | 2016-09-15 to 2018-04-15 | 64,430 |
| validation | 2018-04-15 to 2018-06-21 | 12,847 |
| test | 2018-06-21 to 2018-08-29 | 14,471 |

A training row must additionally have been *delivered* before the validation period began,
otherwise its label was not observable when we claim to be training; the same rule applies
between validation and test. This is what "no look-ahead" means in practice - not shuffling is
necessary but not sufficient. Test rows are never filtered: the test period is the future, and
filtering it would be circular.

### 4.3 Features and preprocessing

18 booking-time features: three calendar fields, the promised lead time, seven basket and
freight values, three physical attributes, three geography fields and the primary product
category. `lead_time_days` is the headline input - it is the promise itself - and the
non-monotonic relationship between promised lead time and lateness (16.9% late for `<=7` days,
4.8% for 8-15, 7.7% for 16-25, 6.0% above 25) is a reminder that this is not a simple "more
slack is safer" story.

Preprocessing is one scikit-learn pipeline: median imputation plus optional standardisation for
the linear model, most-frequent imputation and one-hot encoding with rare-level merging and
`handle_unknown="infrequent_if_exist"` for the categoricals. Unseen-category handling is
tested, because a level invented after deployment must not crash scoring.

### 4.4 Candidates, tuning and selection

Three candidates, all wrapped in the same preprocessor, all tuned with `TimeSeriesSplit(3)` and
all scored with average precision:

| candidate | search space |
| --- | --- |
| `dummy` (most frequent) | none - the floor that proves any skill |
| `logistic_regression` | `C` in {0.1, 1.0}, `class_weight="balanced"` |
| `random_forest` | 400 trees, `max_depth` in {12, 20}, `min_samples_leaf` in {1, 5} |

Average precision is the search metric because delays are rare: it depends only on how well the
model ranks the delayed orders, whereas ROC-AUC would be inflated by the many easy on-time
orders. The dummy baseline is cross-validated on exactly the same folds - without that, its
`nan` score would have made the comparison table look like the baseline had never been
evaluated rather than that it has no ranking skill.

### 4.5 Threshold and evaluation protocol

The decision threshold is chosen **on the validation split** by maximising F1 for the delayed
class (configurable to "highest recall whose precision clears a floor"), stored inside the saved
bundle, and then applied unchanged to the test period, which is scored **once**. Everything
reported below uses that frozen threshold. Uncertainty is quantified with 500 bootstrap
resamples over test rows; a resample containing a single class cannot support a ranking metric,
and such draws are skipped and counted rather than silently dropped.

## 5. Results

### 5.1 Candidate comparison

Average precision by split (candidates scored at 0.5 here; the frozen decision threshold is
applied in 5.3):

| model | train AP | validation AP | test AP | test ROC-AUC | test Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| dummy (most frequent) | 0.0681 | 0.0423 | 0.0428 | 0.5000 | 0.0428 |
| **logistic_regression** | 0.1502 | **0.1126** | **0.0711** | **0.6475** | 0.2847 |
| random_forest | 0.3411 | 0.1025 | 0.0430 | 0.4955 | 0.1604 |

Cross-validated inside the training period (`TimeSeriesSplit(3)`): dummy **0.0785**, logistic
**0.1355**, random forest **0.1481**.

Three things are worth reading off that table.

1. **The random forest overfits catastrophically.** Training AP 0.3411, validation 0.1025, test
   0.0430 - the base rate exactly - with a test ROC-AUC of **0.4955**, i.e. chance. On 64,430
   training rows it memorised the training regime and transferred nothing.
2. **The selection margin was inside the noise.** Validation AP 0.1126 versus 0.1025 is a gap of
   0.010, while the bootstrap interval on the test metric is about 0.021 wide. Model selection
   therefore did not *prove* the linear model better; it chose it, and the test period happened
   to vindicate the choice. The stronger signal was the generalisation gap (forest -0.239 from
   train to validation, linear -0.038), and a selection rule that prices that gap in would have
   been more defensible.
3. **The test period is harder than validation for both models** (0.1126 to 0.0711; 0.1025 to
   0.0430) although the class balance barely moves (4.23% validation, 4.28% test). This is
   regime shift, not prevalence shift: the training window contains the Nov 2017-Mar 2018
   disruption (6.81% late), while the test months run 0.97% (June), 3.38% (July) and 6.19%
   (August). The relationship between features and lateness changed, and inside the test period
   it kept changing.

### 5.2 Calibration

| bin | n | mean predicted | observed late rate | gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.4-0.5 | 3,110 | 0.452 | 0.029 | 0.424 |
| 0.5-0.6 | 3,661 | 0.549 | 0.057 | 0.492 |
| 0.6-0.7 | 2,402 | 0.644 | 0.062 | 0.582 |
| 0.8-0.9 | 473 | 0.837 | 0.089 | 0.748 |

Test ECE **0.4736**. A model that says 55% when the truth is 5.7% is not a probability model.
The cause is mechanical: `class_weight="balanced"` multiplies the minority class by
approximately 89,936/6,534 = 13.8, shifting every score upward by roughly that odds factor.
The *ranking* is untouched (calibration is monotone), which is why AP and ROC-AUC do not change,
but any rule expressed in probability units - including the cost arithmetic in section 7 -
inherits the inflation.

**Measured fix, fitted on validation data only:** isotonic regression on the validation scores
moves validation ECE from 0.3928 to 0.0000 and test ECE from **0.4736 to 0.0203**, leaving test
AP and ROC-AUC identical. The shipped bundle deliberately does not include it: the reported test
score is the single-shot score of the frozen artefact, and swapping the decision rule afterwards
would invalidate it. It is therefore recorded here as the first recommended change.


### 5.3 The frozen decision point, with uncertainty

Threshold chosen on validation data: **0.6300** (max-F1; validation F1 0.2005, precision 0.1340,
recall 0.3978). The cost-optimal threshold on validation would have been 0.7200.

| metric (test period, n = 14,471, 620 late) | value | 95% bootstrap interval | n draws |
| --- | ---: | --- | ---: |
| average precision | **0.0711** | [0.0619, 0.0829] | 500 |
| ROC-AUC (optimistic under imbalance) | 0.6475 | [0.6276, 0.6652] | 500 |
| precision | 0.0625 | [0.0546, 0.0710] | 500 |
| recall | 0.3500 | [0.3120, 0.3895] | 500 |
| F1 | 0.1061 | [0.0931, 0.1193] | 500 |
| Brier | 0.2847 | [0.2821, 0.2871] | 500 |
| accuracy (reported only to show how useless it is) | 0.7472 | - | - |

Confusion at the frozen threshold: TP 217, FP 3,255, FN 403, TN 10,596.

The average-precision interval **[0.0619, 0.0829] lies entirely above the base rate (0.0428)**,
so the ranking skill is not sampling noise. It is, however, small: the model separates late from
on-time orders only about 65% of the time by ROC-AUC.

### 5.4 Label sensitivity

The same frozen probabilities under both label definitions:

| definition | prevalence | average precision | ROC-AUC | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `date_based` (primary) | 0.0428 | 0.0711 | 0.6475 | 0.0625 | 0.3500 |
| `timestamp_based` | 0.0661 | 0.1002 | 0.6453 | 0.0907 | 0.3292 |

The ranking quality is essentially identical (ROC-AUC 0.6475 versus 0.6453); the headline
average precision moves only because the positive class grows. The conclusions do not depend on
which definition is chosen, which is what a sensitivity check is for.

## 6. Error analysis

### 6.1 Where the model fails, by segment

Worst recall among segments with at least 500 test orders (frozen threshold):

| dimension | level | n | late rate | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: |
| customer_state | PR | 743 | 0.0215 | 0.013 | 0.063 |
| customer_state | SP | 6,750 | 0.0576 | 0.142 | 0.154 |
| route_bucket | same-state seller | 5,925 | 0.0651 | 0.189 | 0.176 |
| items_bucket | 2+ items | 1,328 | 0.0271 | 0.031 | 0.194 |
| customer_state | MG | 1,581 | 0.0127 | 0.024 | 0.200 |
| month_bucket | June 2018 | 1,964 | 0.0097 | 0.029 | 0.263 |
| lead_time_bucket | <=7 days | 1,260 | **0.1690** | **0.219** | 0.268 |

Read the precision and late-rate columns together, not the recall column alone: recall is low in
PR and MG (0.063, 0.200) mostly because almost nothing is late there (2.2% and 1.3% of orders),
so a low recall in those segments is not a failure - it is the sensible behaviour of a model
that concentrates flags on risk. Where the late rate is genuinely high, the model's precision
rises with it (tens of percent in the `<=7 days` segment), which is the signature of a model
whose *ranking* is doing real work while its *absolute* level is set by the base rate.

No segment in this run was flagged `below_min_n`, and the subgroup table keeps only the twelve
largest levels per dimension to stop small segments from being quoted as findings.

### 6.2 What the model relies on

Permutation importance on the test period (mean drop in average precision):

| feature | importance | std |
| --- | ---: | ---: |
| `lead_time_days` | 0.03147 | 0.00101 |
| `seller_state` | 0.00309 | 0.00036 |
| `customer_state` | 0.00277 | 0.00197 |
| `mean_weight_g` | 0.00141 | 0.00077 |
| `freight_ratio` | 0.00093 | 0.00017 |
| `n_sellers` | 0.00079 | 0.00040 |

The promise dominates by an order of magnitude (0.031 versus 0.003): shipping a promise too
short is the main driver, followed by geography (seller and customer state, i.e. route) and
weight. This is a sensible, checkable story about how parcel networks behave, and it agrees with
the descriptive split (cross-state 8.05% late versus same-state 4.50%; `<=7`-day promises 16.9%
late).


## 7. The business case: where the model stops paying for itself

### 7.1 The arithmetic

The config holds two assumptions - a missed delay costs **25**, an unnecessary proactive contact
costs **4**. The break-even precision follows directly:

> flagging an order is worth it only if `TP x 25 > FP x 4`, i.e. precision must exceed
> 4/(4+25) = **13.79%**

The model's precision at its frozen threshold is **6.25%** - about half of what it needs. Cost
per 1,000 test orders:

| rule | flags | flag rate | precision | recall | cost / 1,000 | saving vs doing nothing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| do nothing | 0 | 0% | - | 0% | 1,071.11 | 0.00 |
| flag everything | 14,471 | 100% | 4.3% | 100% | 3,828.62 | -2,757.52 |
| break-even rule on **raw** scores | 14,273 | 98.6% | 4.3% | 99.8% | 3,775.90 | -2,704.79 |
| model at 0.5 | 8,041 | 55.6% | 6.1% | 78.6% | 2,317.81 | -1,246.70 |
| **model at the frozen threshold 0.63** | 3,472 | 24.0% | 6.25% | 35.0% | 1,595.95 | **-524.84** |
| break-even rule on **calibrated** scores | 1,596 | 11.0% | 6.8% | 17.4% | 1,295.83 | -224.73 |
| **rule: promise <= 7 days** | 1,260 | 8.7% | **16.9%** | 34.4% | **992.54** | **+78.57** |

### 7.2 The finding that matters

**A one-line rule beats the model at every comparable budget.** The `<=7 days` segment has a late
rate of 16.9%, which is *above* the 13.79% break-even before any model is applied: flagging all
1,260 of those orders (8.7% of the period) is profitable (**+78.57** per 1,000). Given exactly
the same 8.7% budget, the model's own top picks reach only 7.5% precision and lose 157.90.

Why the model loses to it: the model's lift is roughly constant (~1.5-1.7x) across the risk
spectrum, while the late rate varies by a factor of ten between segments. A rule that keys
directly on the highest-base-rate segment captures more bad orders per flag than a global ranking
does, because the global ranking dilutes its flags with orders from low-risk segments where it is
still "relatively" risky.

Where the model does add something: used as a *top-up* on the rule (rule plus the model's
next-most-risky orders up to a 12% budget), precision is **14.9%** against **13.2%** for a random
top-up of the same size - the ranking carries real information, just not enough to justify
expanding the flag budget much beyond the rule's own segment:

| rule | flag rate | precision | cost / 1,000 | saving |
| --- | ---: | ---: | ---: | ---: |
| promise <= 7 days | 8.7% | 16.9% | 992.54 | +78.57 |
| rule + model top-up to 12% | 12.0% | 14.9% | 1,033.93 | +37.18 |
| rule + random top-up to 12% | 12.0% | 13.2% | 1,092.05 | -20.94 |
| rule + model top-up to 16% | 16.0% | 12.5% | 1,129.85 | -58.74 |
| rule + model top-up to 20% | 20.0% | 10.9% | 1,241.79 | -170.69 |

### 7.3 What the recommendation depends on

Everything above rests on `cost_false_negative / cost_false_positive = 6.25`. The requirement at
each operating point is the inverse-precision ratio, and it is remarkably flat:

| operating point | precision | needs c_fn/c_fp above |
| --- | ---: | ---: |
| model at the frozen threshold | 6.25% | 15.00 |
| calibrated model at the break-even probability | 6.8% | 13.78 |
| calibrated model at 0.05 (48% flagged) | 6.5% | 14.48 |
| `<=7 days` segment alone | 16.9% | 4.92 |

So the decision is a business question with a precise translation: **if a missed delay really
costs about 15x an unnecessary contact (or the flag budget can be cut to the riskiest ~9% of
orders), the model becomes worth deploying. At the assumed 6.25:1, it is not.** That is a more
useful sentence than "the model achieved an ROC-AUC of 0.65", and it is why the cost assumptions
live in the config as parameters rather than as constants in the code.


## 8. What did not work, and what was rejected

* **The random forest.** Best cross-validated score of the three (0.1481 against the linear
  model's 0.1355) and a test ROC-AUC of 0.4955. It was rejected by validation average precision;
  in hindsight the generalisation gap was the more informative diagnostic.
* **The distance feature.** The public geolocation table offers only noisy median coordinates per
  zip prefix, so no defensible booking-time distance could be computed. Excluded, with the reason
  recorded in the feature audit rather than quietly approximated.
* **Payment features.** Chosen at checkout but written during approval, so their availability at
  the booking instant is uncertain; off by default and switchable in the config.
* **The max-F1 threshold objective.** Not cost-optimal (it flags more than the cost-optimal 0.72
  on validation), and on the test period the entire family of global thresholds loses money. The
  flag *budget* may be the more natural business control than a probability cut.
* **Uncalibrated probabilities.** Inflated by the class weighting; any rule expressed in
  probability units behaves badly (section 5.2 and the third row of section 7.1).
* **The un-cross-validated dummy baseline.** An early version reported the baseline with a `nan`
  cross-validation score, which made a placeholder look like an evaluation. Fixed: the baseline is
  now scored on the same `TimeSeriesSplit` folds, so its floor is measured rather than assumed.
* **A silently empty column in the app.** The Streamlit tab read `ci_low`/`ci_high` while the
  evaluation wrote `lower`/`upper`, so the confidence-interval columns rendered blank - a bug that
  cannot be seen without looking for it. Fixed, and `tests/test_app_runtime.py` now asserts the
  intervals are populated.
* **The synthetic generator's seasonal term.** An early version produced a ~50% late rate in
  November and December, which would have produced a beautiful, meaningless model. Fixed, with
  guard tests at both the statistical and the constant level (data card, section 8).
* **The full search grid on 96k rows.** Kept, but the forest's three-fold, four-configuration grid
  costs 5.5 minutes and lost. Tuning the cheap model first and escalating only if the gap
  justifies it would have saved most of the runtime.

## 9. Limitations and what would change the conclusion

1. **The regime problem is the big one.** The model trains through the Nov 2017-Mar 2018
   disruption and is tested on a calmer 2018 period. A production system must handle this
   continuously: rolling re-training, drift monitoring on the late rate *and* on the feature
   distributions, and challenger-shadow deployment before any change.
2. **Non-stationarity inside the test period itself.** Late rate runs 0.97% (June) to 6.19%
   (August). A fixed threshold is therefore a bet on the level of the next month; a
   capacity-based rule ("flag the riskiest k%") is more robust to that than a probability cut.
3. **The costs are assumptions, not measurements.** The whole business conclusion flips if the
   real ratio is nearer 15:1 than 6.25:1. That is the single most valuable number the business
   could supply.
4. **No intervention data.** Nothing here measures whether a proactive contact actually reduces
   lateness or churn; without a holdout group the "saving" is expected-cost arithmetic, not
   evidence. A randomised or switchback trial is the way to price it properly.
5. **Excluded orders.** Canceled and unresolved orders (3.0%) are dropped rather than modelled. In
   production they need their own decision rule, and the drop itself is a selection effect.
6. **No counterfactual promise modelling.** `lead_time_days` is a decision variable the business
   controls. This project predicts lateness *given* the promise; the more valuable question - what
   happens to lateness if we lengthen the promise? - needs an interventional design.
7. **A single split.** One train/validation/test cut gives one test number. Rolling-origin
   backtesting over several cut points would give a distribution, which is what a deployment
   decision actually needs.
8. **Scale.** Olist is one Brazilian marketplace; its promise mechanics, network and seasonality do
   not transfer to another market without re-validation.


## 10. Recommendation

1. **Deploy the segment rule now** (`promise <= 7 days`, optionally extended to cross-state
   routes): profitable under the assumed costs, interpretable, and free to run.
2. **Calibrate the model** (isotonic, fitted on validation) and re-express the decision as a flag
   *budget* rather than a probability cut, then measure it in shadow mode.
3. **Answer the cost question.** With a verified cost ratio above roughly 15:1, or a much smaller
   flag budget, the model's ranking becomes the better rule.
4. **Move to rolling-origin evaluation** before any production claim, and add drift monitoring on
   both the labels and the features.
5. **Only then invest in better features** (carrier, warehouse, distance, promise-policy
   interactions): the current ceiling is set by the information available at booking, not by the
   choice of algorithm.

## 11. Reproducing this report

```bash
python -m src.prepare      # 96,470 orders, 6.77% late, fingerprints + feature audit
python -m src.train        # tunes three candidates, freezes the threshold, scores test once
python -m src.evaluate     # bootstrap intervals, subgroups, calibration, cost, 13 figures
python -m pytest           # the invariants: leakage, split order, maturity, app rendering
```

Every number above is reproducible from `reports/evaluation_summary.json`,
`reports/model_comparison.csv`, `reports/test_calibration.csv`,
`reports/permutation_importance.csv`, `reports/test_subgroups.csv` and
`reports/run_manifest.json`. The rule comparisons in section 7 come from scoring the saved bundle
(`models/best_model.joblib`) on the frozen folds: they add no new model, only new decision rules
applied to the same probabilities.

