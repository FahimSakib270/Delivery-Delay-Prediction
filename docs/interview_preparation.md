# Interview preparation

Use `reports/results_summary.md` and `reports/run_manifest.json` for the current
measured results, selected parameters and sample counts. These answers describe the
implemented protocol; do not memorize stale numbers from an earlier run.

### 1. What question does this project answer?
Can information available at booking help rank completed orders by the risk of
arriving after their promised calendar date?

### 2. Why classification rather than regression?
The initial question is late versus on time. Predicting actual duration would be a
separate regression problem with its own objective and evaluation.

### 3. What is the label?
One if actual delivery calendar date is later than the promised calendar date;
otherwise zero. Delivery on the promised day is on time.

### 4. What is leakage here?
Using future delivery information, post-delivery reviews, or outcomes unavailable
at a historical training boundary. Outcome timestamps only construct labels and
check their availability; they are excluded from model features.

### 5. Why split chronologically?
A live model predicts future bookings from past information. Random splitting can
mix future regimes into training. Tied booking timestamps stay in the same period.

### 6. Why check label maturity inside CV?
An order can be booked before a fold boundary but delivered afterward. Its outcome
was unavailable when predictions at that boundary began. The fold excludes it.
The filter uses the actual timestamp, not a delivery date rounded to midnight.

### 7. What does logistic regression do?
It combines encoded features linearly in log-odds space. The fitted coefficients
support a simple ranking model; they are associations, not causal effects.

### 8. How is a random forest different?
It averages many decision trees and can learn nonlinear interactions. More complex
models may overfit or generalize poorly across changing operational conditions.

### 9. How did you select the model and threshold?
Hyperparameters use training-period label-aware CV. The model is chosen by validation
average precision; the primary threshold maximizes validation F1. Candidate test
metrics are calculated after selection and cannot affect that selection.

### 10. Why is accuracy insufficient?
Most orders are on time. A majority-class predictor can have high accuracy while
catching no delays. Precision, recall and average precision expose this failure.

### 11. What do precision, recall, AP and ROC-AUC mean?
Precision is the delayed fraction among flagged orders. Recall is the fraction of
all delays flagged. AP summarizes the precision-recall ranking and depends on
prevalence. ROC-AUC describes positive-negative ranking; imbalance does not
automatically inflate it, but it does not directly show low-prevalence precision.

### 12. Why isn't the demo score a probability?
Class-weighted training can distort probability estimates. The serving model is
uncalibrated, so the demo shows a score and an experimental flag without cost advice.
Aggregate metric bootstrap intervals are not confidence intervals for one order.

### 13. What did the calibration experiment do?
It fitted isotonic regression using validation scores and outcomes, then evaluated
it retrospectively on test. A non-decreasing mapping can create ties, so AP and
ROC-AUC are recalculated. This exploratory calibrator is not in the serving bundle.

### 14. Can you claim the test is untouched or a rule is profitable?
No. Automated selection excludes test outcomes, but the historical set was inspected
before the corrected rerun. Supplementary rules and budgets are fixed exploratory
comparisons. Their hypothetical penalties are not measured profit or causal benefit.
Use new holdouts, rolling-origin evaluation and a controlled pilot before deployment.

### 15. How would this connect to ProFast?
First record trustworthy booking-time snapshots and delivery outcomes from real
operations, validate transfer to Bangladesh, and define the actual decision and
intervention costs. Only then consider a backend scoring endpoint. The current
prototype is standalone and does not claim integration with my thesis system.
