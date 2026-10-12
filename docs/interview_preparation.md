# Interview preparation

Every question below is one this project invites. The answers quote the frozen run's artefacts, so
they can be defended with a file path.

---

### Q1. "Your model's precision is 6%. Isn't that useless?"

**Short answer:** the model is not useless, but at the assumed costs the *decision* it supports is.

**Long answer:** precision is meaningless without the base rate. The test period was 4.28% late, so
6.25% precision is a 1.46x lift in precision and a 1.66x lift in average precision (0.0711 against
0.0428, interval [0.0619, 0.0829] - chance is excluded). For a decision, what matters is whether
precision clears the break-even implied by the costs: with a missed delay at 25 and a false alarm at
4 you need 13.79% precision, and the model gives 6.25%. So the correct statement is "real but
insufficient skill at these costs", not "no skill". The fix is visible in the numbers: a one-line
rule (`promise <= 7 days`) reaches 16.9% precision on 8.7% of orders, because the late rate varies
ten-fold across segments while the model's lift is roughly constant.

### Q2. "Why average precision rather than ROC-AUC or accuracy?"

* Accuracy: predicting "on time" for every order scores 93%; the metric cannot see the class you
  care about.
* ROC-AUC: dominated by the many easy true negatives; the model's test ROC-AUC (0.65) looks much
  better than its actual usefulness. It is reported, explicitly flagged as optimistic, and never used
  for selection.
* Average precision: depends only on how the rare positive class is ranked, which is exactly what
  "spend the flag budget on the riskiest orders" needs. It is the search, selection and headline
  metric.

### Q3. "How do you know there is no leakage?"

Three layers:

1. a written audit (`src/features.py:FEATURE_AUDIT`) classifying every candidate feature with whether
   it exists at booking and why it is used or rejected;
2. a machine-checked prohibitions list (`PROHIBITED_MODEL_COLUMNS`: delivery timestamps, carrier
   hand-over, approval timestamp, order status, payment rows, identifiers, the label itself) with
   tests that fail if any of them reaches the feature set, or if a *used* feature is marked
   unavailable at booking;
3. a chronological split plus a label-maturity filter, so the model never trains on a row whose
   delivery was not yet known when the next period began.

`order_status` is the interesting one: the most predictive column in the dataset, and it *is* the
outcome, so it is prohibited outright.

### Q4. "Why not shuffle the split? Everyone uses `train_test_split`."

Shuffling lets the model train on orders booked *after* the ones it is scored on. With a known
regime shift in this data (Nov 2017-Mar 2018 runs at 12-15% late against 1.8-5% in mid-2018) a
shuffled split would leak the easy future regime into training and inflate every metric. The test
period here is a real future - 2018-06-21 to 2018-08-29 - scored once.

### Q5. "What is label maturity and why does it matter?"

A training order booked on the last day of the training window has not been delivered yet when the
validation period begins, so its label would not have been observable at the moment we claim to be
training. I drop those rows from train (5%) and validation (11%), and never from test: the test
period *is* the future, so filtering it would be circular. Without this, "no shuffling" is still
look-ahead bias in disguise.

### Q6. "The random forest had the better cross-validated score. Why ship a logistic regression?"

Cross-validated AP: forest 0.1481, linear 0.1355. Validation AP: forest 0.1025, linear **0.1126** -
and validation is the split used for selection. The forest's train-to-validation gap was -0.239
versus -0.038 for the linear model, and on the test period it collapsed to AP 0.0430 (the base rate)
with ROC-AUC 0.4955 - pure chance. The honest nuance: the selection margin (0.010) sat inside the
bootstrap noise (~0.021), so selection did not *prove* anything; the generalisation gap was the
better warning sign, and the report says so.

### Q7. "Your calibration error is 0.47. How can you trust the model?"

I don't trust the *probabilities*; I trust the *ranking*, and I say so. The cause is
`class_weight="balanced"`, which inflates scores by roughly the class ratio (13.8x). Ranking metrics
are invariant to a monotone transform, which is why AP and ROC-AUC are unaffected - but any
probability-unit rule inherits the inflation: applying the cost break-even (0.1379) to the raw scores
flags 98.6% of orders and loses 2,705 per 1,000. I measured the fix instead of guessing: isotonic
calibration fitted on validation data only takes test ECE from 0.4736 to 0.0203 with identical
ranking metrics. It is not in the shipped bundle because adding a decision step after the single-shot
test evaluation would invalidate the reported score.

### Q8. "Why didn't you use the geolocation table for distance?"

Because no *defensible* booking-time distance can be computed from it: the public table holds noisy
median coordinates per zip prefix, and a naive great-circle number would be an invented feature
dressed up as data. I excluded it and recorded the reason in the feature audit. I would rather defend
a smaller feature set than a slightly better metric built on an indefensible input.

### Q9. "Why is the test score lower than validation?"

Because the process is non-stationary, not because something is broken. Validation (Apr-Jun 2018)
scores AP 0.1126 and test (Jun-Aug 2018) 0.0711, although the class balance barely moves (4.23%
versus 4.28%). The late rate inside the test window itself runs 0.97% (June), 3.38% (July), 6.19%
(August), while the training window contains the Nov 2017-Mar 2018 disruption at 6.81% late. The
feature-to-lateness *relationship* moved. That is the main reason my recommendation is
rolling-origin evaluation and drift monitoring before any production claim.

### Q10. "What would you do with another month?"

1. Rolling-origin backtesting over several cut points, to get a distribution of test metrics instead
   of one number.
2. Isotonic calibration in the bundle plus a *capacity-based* decision rule ("flag the riskiest k%"),
   which is far more robust to a drifting base rate than a fixed probability cut.
3. Test whether promise policy is the real lever: `lead_time_days` is a business decision variable,
   so the interventional question ("what if we lengthen the promise by two days?") is worth more than
   another algorithm.
4. Push the two things the data lacks - carrier and warehouse attributes, and a holdout group so the
   intervention's effect on lateness and churn can be measured rather than assumed.

### Q11. "How would you deploy this, and how would you know it still works?"

Serve the frozen bundle behind a feature-assembly layer that is the same code path as training; log
every scored order with its features, score and decision; keep the segment rule as the incumbent and
run the model in shadow mode; monitor (a) the late rate by week, (b) the score distribution (PSI/KS),
(c) realised precision at the operating point and (d) per-segment recall for fairness. Retrain on a
rolling window, and require a challenger to beat the incumbent on the most recent *out-of-time*
window before promotion. The app in this repository already refuses to re-train or re-choose
anything, which is the right shape for that kind of server-side control.

### Q12. "What did you get wrong during the project?"

A concrete list, all found and either fixed or documented:

* the dummy baseline was reported with a `nan` cross-validation score, which made a placeholder look
  like an evaluation - now scored on the same `TimeSeriesSplit` folds;
* the Streamlit tab read `ci_low`/`ci_high` while the evaluation wrote `lower`/`upper`, so the
  confidence-interval columns rendered silently blank: a bug you only find by asserting on rendered
  output, which the test suite now does;
* the synthetic generator added the seasonal peak to the promise far more weakly than to the actual
  journey, producing a 50% late rate in November - which would have produced a beautiful, meaningless
  model. Fixed, and now guarded by tests at both the statistical and the constant level;
* an in-memory pandas frame with Arrow-backed string columns containing `pd.NA` crashed
  scikit-learn's indexing, while the same data re-read from CSV did not - which is why the tests now
  fit models on the dtypes production actually loads.

### Q13. "Why logistic regression and not gradient boosting or a neural net?"

Three reasons, in order of importance: the signal is dominated by one smooth effect (the promise)
plus a handful of categorical ones, which a linear model in a one-hot design matrix captures; a
400-tree forest already overfit this data badly, so higher capacity is a big bet for a small return;
and a linear model's coefficients are inspectable, which matters when the output affects customers.
If the feature set grows to include carrier and warehouse attributes, boosting deserves a re-test -
capacity is not the current bottleneck, information is.

### Q14. "How do you handle a category you have never seen before?"

The encoder uses `min_frequency` merging for rare levels and `handle_unknown="infrequent_if_exist"`,
so a level invented after deployment is routed to the infrequent bucket instead of raising. The app
additionally offers an explicit "(other / rare)" option, so a user who knows the level is new is not
forced to pick a real one. Both behaviours are covered by tests.

### Q15. "What is the one number you would want from the business?"

The cost ratio - specifically whether a missed delay costs more like 6x or more like 15x an
unnecessary proactive contact. At 6.25:1 the model should not be deployed at a global threshold; if
the ratio is above roughly 15:1, its precision clears the break-even and it becomes the better rule.
The break-even probability `p* = c_fp / (c_fp + c_fn)` is printed by the app precisely so that
conversation can happen without reverse-engineering the model.

### Q16. "How is any of this reproducible?"

One config file holds every tunable decision; a seed drives Python, NumPy and every estimator; the
raw files, the derived label and the prepared table all get SHA-256 fingerprints; the run writes a
manifest with library versions and the git commit; and the tests include determinism checks (two
generations of the synthetic data are byte-identical; the split and threshold selection are seeded).
The generator even carries a fingerprint of its own source, so a changed generator cannot silently
reuse stale CSVs.

### Q17. "If you had to defend this as a business decision tomorrow, what would you say?"

"Spend the first month on the rule, not the model. Flagging orders whose promised lead time is seven
days or fewer costs nothing to build, affects 8.7% of orders and is expected to save around 79 per
1,000 orders under our stated costs. The model currently costs 525 per 1,000 at its frozen threshold,
because its precision (6.25%) is half the break-even (13.79%). Keep it in shadow mode, calibrate it,
and revisit when we know the real cost of a missed delay: above roughly 15:1 the model's ranking
becomes the better rule, and inside the risky segment it already lifts precision from 13.2% to 14.9%
at the same budget."

### Q18. "What is the most interesting thing you found?"

That the *framing* beat the *modelling*. The dataset contains a strong, simple signal (how much
slack the promise carries) and a weak, complex one (everything else). A correctly-framed decision
analysis - expected cost, break-even precision, a flag budget - turned a "0.65 ROC-AUC, is that good?"
question into a precise statement: at these costs, the profitable action lives in the top 8.7% of
orders, and no global threshold achieves it. Most of the value of the project came from that
translation, not from the estimator.
