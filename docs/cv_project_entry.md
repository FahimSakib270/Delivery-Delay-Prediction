# Portfolio project description

**Delivery-delay prediction — Python, scikit-learn, pandas, Streamlit**

- Built an order-level pipeline using public Olist e-commerce data and booking-time features.
- Compared a baseline, logistic regression and random forest with chronological
  evaluation, training-fitted preprocessing and delivery-time label-maturity checks.
- Built a local risk-score demo and analyzed errors, calibration and subgroup performance.

## Short description

I built a delivery-delay prediction prototype inspired by my ProFast undergraduate
project. On a retrospective chronological Olist test period, logistic regression
achieved average precision of 0.0711 against a late-order prevalence of 0.0428.
Ranking performance was modest, and the chosen threshold cost more than flagging
nothing under the assumed error penalties. The project has not been validated for
Bangladesh couriers or live operations.

See [generated results](../reports/results_summary.md) for the full evaluation.
Scores are uncalibrated; hypothetical costs do not show measured profit.

## Interview points

1. Delivery outcomes define labels but never enter the model as features.
2. Chronological splits also need label-maturity checks: training deliveries must
   be known before validation starts, including inside cross-validation.
3. Risk scores, calibrated probabilities and operational decisions have different
   meanings. Any proposed intervention needs further validation.
