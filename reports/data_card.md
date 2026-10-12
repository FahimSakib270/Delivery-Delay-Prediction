# Data card - ProFast delivery-delay prediction

This documents the data behind the shipped artefacts: what it is, what was removed from
it, how the label is defined, and what it cannot support. Written to be read *before* the
results, not after.

---

## 1. Identity and licence

| field | value |
| --- | --- |
| name | Brazilian E-Commerce Public Dataset by Olist |
| publisher | Olist (Brazilian marketplace), published on Kaggle |
| slug | `olistbr/brazilian-ecommerce` |
| licence | **CC BY-NC-SA 4.0** - non-commercial, share-alike, attribution required |
| version used | dataset as published, last updated 2021-10-01, 126,186,995 uncompressed bytes |
| format | 9 CSV files, ~100k orders, 2016-09-04 to 2018-10-17 |

The licence is the practical constraint to check before any commercial use: the *code* in
this repository is free to reuse, the *data* is not. The dataset is downloaded in place and
never re-hosted here.

```python
import urllib.request
urllib.request.urlretrieve(
    "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce",
    "olist.zip",
)   # unzip into data/raw/olist/
```

### Files used, with SHA-256 (first 16 hex characters of the shipped run)

| table | file | rows | sha256 (head) |
| --- | --- | ---: | --- |
| orders | `olist_orders_dataset.csv` | 99,441 | `8df58ef3d2d7e994` |
| items | `olist_order_items_dataset.csv` | 112,650 | `0bc4d068c4fe38cb` |
| products | `olist_products_dataset.csv` | 32,951 | `3e6569628a17fbc7` |
| customers | `olist_customers_dataset.csv` | 99,441 | `983a422239e1712d` |
| sellers | `olist_sellers_dataset.csv` | 3,095 | `1f643d2b950373b8` |
| payments (optional) | `olist_order_payments_dataset.csv` | 103,886 | `4f713964f2815dbb` |
| translation (optional) | `product_category_name_translation.csv` | 71 | `a81f0d1f27b27e72` |

Full digests, plus the SHA-256 of the prepared table, are in
`data/processed/dataset_metadata.json` under `provenance.raw_file_sha256` and
`provenance.processed_sha256`. The run's environment and git commit are in
`reports/run_manifest.json`.

## 2. What the data is, and what it is not

**Is:** a real, messy, event-level record of 99,441 marketplace orders with their items,
sellers, customers, freight, payments and reviews. Every order has a booking timestamp and a
promised delivery date; almost all have a recorded delivery timestamp.

**Is not:**

* not a carrier or warehouse dataset - no carrier identity, no warehouse, no route, no
  dispatch queue, so nothing here measures *how* a parcel moved;
* not customer demographics - only an anonymised `customer_unique_id`, a city and a state;
* not a promise-quality record - the promised date is Olist's own estimate, so this project
  predicts *deviation from the promise*, which is not the same as *slow delivery*. An order
  delivered in 30 days against a 35-day promise is "on time" and labelled 0;
* not balanced over time - 2016 holds 267 orders, 2017 holds 43,426 and the first eight
  months of 2018 hold 52,777, so any per-year comparison rests on very unequal support;
* not free of a known logistics episode - see section 6.



## 3. Row accounting - every order is either used or explicitly excluded

The preparation funnel is fully counted, so a reader can audit the exclusions rather than
trust a headline row count. All numbers come from `data/processed/dataset_metadata.json`.

| step | orders | reason |
| --- | ---: | --- |
| raw orders | 99,441 | as published |
| duplicate `order_id` removed | 0 | none found |
| missing booking or promised date | 0 | the promise is mandatory: it defines the label |
| promise earlier than booking | 0 | would be a data error, none present |
| canceled / unavailable | -1,234 | no delivery outcome exists to learn from |
| delivered date absent | -1,737 | unresolved: not yet delivered, or lost record |
| delivered before booking | 0 | impossible, none present |
| **usable** | **96,470** | pairs of (promise, delivery) that can be labelled |
| without item rows | 0 | all usable orders have items |
| **final modelling rows** | **96,470** | one row per order |

The 2,971 excluded orders (3.0%) are excluded *before* any split, so the test period is not
a filtered subset of one fold: the same rule applies everywhere. This is deliberately
conservative: canceled and unresolved orders are exactly the ones a live system would have
to handle, and the right way to handle them is a separate decision rule, not a smuggled-in
label. Section 8 of the technical report says what a production system would do instead.

## 4. The label

```python
promised_date  = order_estimated_delivery_date.date()
delivered_date = order_delivered_customer_date.date()
is_late        = 1 if delivered_date > promised_date else 0        # primary
is_late_ts     = 1 if order_delivered_customer_date > order_estimated_delivery_date else 0
```

**Why calendar dates for the primary label.** The promised date is communicated to the
customer as a day ("arrives by 12 March"). The `order_estimated_delivery_date` column
carries a time of day that is a system artefact. Comparing raw timestamps would treat a
parcel delivered at 09:00 on the promised day as on time and the same parcel at 23:00 as
late, which is not a difference the business or the customer can act on. The
timestamp-based definition is kept as a sensitivity check and reported alongside, because a
definition that flips the results would be a finding in itself.

| definition | late | on time | late rate |
| --- | ---: | ---: | ---: |
| `date_based` (primary) | 6,534 | 89,936 | **6.77%** |
| `timestamp_based` (sensitivity) | 7,826 | 88,644 | 8.11% |

The class is genuinely rare, which drives three downstream choices: average precision as
the primary metric, a threshold chosen on validation data rather than fixed at 0.5, and
bootstrap intervals on every headline number.

Useful scale for interpretation: the median *promised* lead time is **24 days** while the
median *actual* journey is **10 days**, so the promise carries a median of about two weeks
of slack. When an order *is* late, it is late by a median of **7 days** (90th percentile: 22
days). The model is therefore predicting the tail of a process that is usually comfortable -
Which is precisely why the base rate is low and the decision threshold matters.

## 5. Features: what is in, what is out, and why

The full table is machine-generated into `reports/feature_audit.csv` (18 model inputs and
15 rejected candidates, each with a reason). Summary of the rules:

**In (18 features, all knowable at booking):**

| group | features |
| --- | --- |
| when booked | `purchase_month`, `purchase_weekday`, `purchase_hour` |
| the promise | `lead_time_days` (promised date minus booking date) |
| basket | `n_items`, `n_products`, `n_sellers`, `price_sum`, `freight_sum`, `order_value`, `freight_ratio` |
| physical | `total_weight_g`, `mean_weight_g`, `total_volume_cm3` |
| geography | `customer_state`, `seller_state`, `same_state` |
| product | `product_category` |

**Out, with the reason:**

| candidate | why it is out |
| --- | --- |
| `order_delivered_customer_date`, `order_delivered_carrier_date` | future events; used only for the label and the maturity mask |
| `order_approved_at` | happens after booking |
| `order_status` | the outcome itself; using it leaks the target |
| `shipping_limit_date` | filled by Olist while processing; booking-time availability is unclear |
| `payment_type`, `payment_installments` | chosen at checkout but written during approval, so their availability at the prediction instant is uncertain - excluded from the primary experiment, and switchable via `features.include_payment_features` |
| `review_score`, `review_comment` | written after delivery |
| `order_id`, `customer_id`, `product_id`, `seller_id`, `customer_unique_id` | identifiers: not predictive of a physical process, and high-cardinality ids encourage memorisation |
| `customer_city`, `seller_city` | ~4,100 cities would create a very large sparse matrix; the state already carries most of the regional signal at this budget |
| geographical distance | the public geolocation table only offers noisy median coordinates per zip prefix, so no *defensible* booking-time distance can be computed without extra assumptions. Skipped rather than guessed. |

The list is enforced, not documented: `src/data_prep.py:PROHIBITED_MODEL_COLUMNS` plus
`tests/test_data_prep.py` fail the build if any of those columns reaches the feature set.

## 6. Known biases and one important episode

These are properties of the data that limit what the results mean. All are reported rather
than smoothed away.

* **November 2017 - March 2018 is a different regime.** The late rate by booking month is
  5.4% (Jan) but **15.1% in March 2018** and **12.4% in November 2017** (Black Friday
  volume), against **1.8% in June**. A large share of the late-rate signal in any time-based
  split is that episode, so a model tuned here partly learns "2018, not 2017".
* **Tight promises are the risky ones.** Late rate by promised lead time: `<=7 days`
  **16.9%**, `8-15 days` 4.8%, `16-25 days` 7.7%, `>25 days` 6.0%. The relationship is
  non-monotonic, which is a useful reality check: a simple "more slack is safer" rule would
  be wrong.
* **Cross-state routes are worse:** 8.05% late versus 4.50% for same-state sellers.
* **Missing category:** 1,332 orders (1.4%) have no product category in the translation
  table; the encoder imputes the most frequent level for them and the missingness is
  recorded in the metadata.
* **Geographic concentration:** SP accounts for 40,494 orders (42%) and RJ/MG a further
  24%. Subgroup results for small states rest on few orders and are flagged as such.
* **No counterfactual:** nothing in the data says what would have happened to an order if
  the business had intervened, so every "saving" in the cost section is an assumption, not
  a measurement.

## 7. The split, and why a training row can be dropped for being *too recent*

The split is a cut on booking time, never a shuffle:

| fold | booking window | rows after the maturity filter |
| --- | --- | ---: |
| train | 2016-09-15 to 2018-04-15 | 64,430 (from 67,529) |
| validation | 2018-04-15 to 2018-06-21 | 12,847 (from 14,470) |
| test | 2018-06-21 to 2018-08-29 | 14,471 (never filtered) |

**Label maturity.** A training order booked on the last day of the training window has not
been delivered yet when the validation period begins, so its label would not have been
observable at the moment we claim to be training. Those rows are removed:

* training rows are kept only if `delivered_date <` the first booking of the validation period;
* validation rows are kept only if `delivered_date <` the first booking of the test period;
* test rows are never filtered - the test period *is* the future, and filtering it would be
  circular. The number of test rows whose delivery falls after the last booking instant in
  the dataset is recorded in the metadata instead.

The cost is visible and small (5-11% of each early fold). The benefit is that no fold is
trained on information from the future.

## 8. The synthetic fallback (what it is for, and what it is not for)

`src/synthetic.py` generates Olist-shaped tables when `data/raw/olist/` is absent, so the
whole pipeline - and the test suite - can run on any machine, including offline.

* It is **deterministic**: one seed produces byte-identical CSVs
  (`tests/test_synthetic.py` compares two generations).
* It is **self-declaring**: `PROVENANCE.json` records `kind: synthetic`, the seed, the
  requested row count and a SHA-256 fingerprint of the generator file. If the generator
  changes, `ensure_synthetic_raw` notices the stale fingerprint and regenerates instead of
  silently reusing old CSVs, and the app shows a warning banner **before** any metric.
* It is **not a replica of Olist.** Its seasonal effect is deliberately mild (Nov/Dec
  roughly 8.7% late versus 6.7% elsewhere), while the real data has a much sharper
  Nov-Mar episode. An earlier version of the generator had a bug that made ~50% of
  November orders late; `tests/test_synthetic.py` now guards the magnitude of that term
  both statistically and at the level of the constants. The lesson is recorded here because
  it is the exact failure mode a synthetic fallback invites: metrics that look fine because
  the data was built to make them look fine.
* Numbers computed on it are a **software smoke test**. They are never reported as evidence
  about real deliveries. The shipped artefacts in this repository were produced from the
  real Olist CSVs (`kind: olist`).

## 9. Reproducibility record

| what | where |
| --- | --- |
| random seed (Python + NumPy + every estimator) | `project.random_seed` in `config/config.yaml`, recorded in the manifest |
| dataset fingerprint | `provenance.dataset_fingerprint` (order-insensitive hash of `order_id`, booking timestamp, label) |
| prepared-table SHA-256 | `provenance.processed_sha256` |
| raw-file SHA-256 (per table) | `provenance.raw_file_sha256` |
| library versions | `run_manifest.json:environment` |
| git commit | `run_manifest.json:git.commit` |
| every modelling decision | `run_manifest.json:config` (the resolved config, not the file on disk) |

Re-running `python -m src.prepare && python -m src.train && python -m src.evaluate` on the
same CSVs reproduces the same fingerprints and the same numbers; the split, the tuning and
the threshold selection are all seeded and all deterministic.
