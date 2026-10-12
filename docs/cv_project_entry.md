# CV project entry

## Two lines (CV / LinkedIn)

> **Delivery-delay prediction at booking time (Python, scikit-learn, pandas, Streamlit).** Built
> an end-to-end pipeline on 96,470 real Olist orders that scores order risk before any delivery
> timestamp exists, with chronological splitting, a label-maturity filter, leakage tests and
> bootstrap intervals; showed that a one-line promise-length rule beat the learned model once the
> cost of a false alarm was priced in, and recommended deployment of the rule instead.

## One paragraph (portfolio / cover letter)

> I built an end-to-end machine-learning prototype that predicts, at the moment an order is
> booked, whether it will arrive after the promised delivery date. Working on 96,470 real orders
> from the public Olist dataset, I enforced booking-time-only features (with a machine-checked
> prohibited-column list), split strictly by time, and filtered training rows whose labels were not
> yet observable when the next period began. The selected model - a class-weighted logistic
> regression chosen over a random forest that overfit its training regime - reached average
> precision 0.071 with a bootstrap interval of [0.062, 0.083] against a 0.043 base rate on a
> single-shot test period. The more useful outcome was negative: once the intervention cost was
> made explicit, the model's 6.3% precision fell short of the 13.8% break-even, while a one-line
> rule (flag promises of seven days or fewer) was profitable at 16.9% precision. I calibrated the
> scores to cut the calibration error from 0.47 to 0.02, quantified what the ranking *does* add
> (14.9% versus 13.2% precision inside the risky segment), and documented the whole thing as a data
> card, model card and technical report with a reproducible artefact trail.

## Skill keywords to draw from

`Python` · `pandas` · `scikit-learn` · `Pipeline`/`ColumnTransformer` · `TimeSeriesSplit` ·
`GridSearchCV` · average precision vs ROC-AUC · class imbalance · probability calibration
(isotonic) · bootstrap confidence intervals · permutation importance · subgroup error analysis ·
expected-cost threshold analysis · leakage prevention · label maturity / look-ahead bias ·
reproducibility (seeds, fingerprints, manifests) · Streamlit · pytest · data/model cards ·
technical writing

## The three sentences that make this project memorable in an interview

1. "I predicted at booking time, which meant deleting every delivery timestamp from the feature
   set - and I wrote a test that fails if one ever comes back."
2. "The model's ranking was statistically real but not economically useful: at a 6.25:1 cost
   ratio, its precision was half the break-even, so I recommended a one-line rule instead."
3. "Then I measured how close the model came: calibrated, it lifted precision to 14.9% inside the
   risky segment against 13.2% for random picks - so it is a challenger, not a product."
