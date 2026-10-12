# CV project entry

**Delivery-delay prediction at booking time — Python, scikit-learn, pandas, Streamlit**

- Built an order-level ML pipeline using public Olist e-commerce data, with audited
  booking-time features and training-fitted preprocessing.
- Compared a majority-class baseline, logistic regression and random forest using
  chronological splits and delivery-time label-maturity checks inside cross-validation.
- Evaluated ranking, classification errors, calibration and subgroup performance;
  built a local risk-score demo and documented limitations and reproducible exploratory analyses.

## Optional measured result

Use the current `reports/results_summary.md` if adding a metric. Include the dataset,
late-order prevalence and retrospective chronological evaluation context. Do not use
accuracy alone, describe an uncalibrated score as a probability, or claim publication,
novel algorithms, production integration, measured profit or deployment readiness.

## Professor-facing description

I developed a delivery-delay prediction prototype inspired by my ProFast undergraduate
package-management system. Using public Brazilian marketplace data, I compared simple
models under temporal evaluation with observable-label checks and investigated errors
and calibration. The model has limited retrospective ranking performance. The useful
outcome is a reproducible analysis of the model's limitations and of simple candidate
rules that need further validation. It has not been validated on Bangladesh operations.

## Three points to explain in an interview

1. Delivery outcomes create the label but never enter the model as features.
2. Chronological booking order alone is insufficient: a fold's training outcomes must
   already have been observed when its validation period begins.
3. A model score, a calibrated probability, and an operational decision are different;
   hypothetical error costs do not prove that an intervention will improve deliveries.
