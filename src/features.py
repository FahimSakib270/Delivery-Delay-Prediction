"""Feature schema, preprocessing pipeline and the feature-audit table.

All preprocessing is expressed as scikit-learn transformers so that every learned
quantity (medians, most-frequent categories, one-hot vocabulary, scaler statistics)
is fitted **only on the training fold** and is stored inside the saved pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

# --------------------------------------------------------------------------
# Feature audit:  Feature | Meaning | Source | Available at booking? | Used? | Reason
# --------------------------------------------------------------------------
FEATURE_AUDIT: List[Dict[str, str]] = [
    {"Feature": "purchase_month", "Meaning": "Calendar month of booking (1-12)",
     "Source": "order_purchase_timestamp", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Known at checkout; captures Brazilian peak season (Nov/Dec)."},
    {"Feature": "purchase_weekday", "Meaning": "Day of week of booking (0=Mon)",
     "Source": "order_purchase_timestamp", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Weekend bookings may be processed later."},
    {"Feature": "purchase_hour", "Meaning": "Hour of day of booking (0-23)",
     "Source": "order_purchase_timestamp", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Late-evening bookings miss same-day handling."},
    {"Feature": "lead_time_days", "Meaning": "Days from booking date to promised delivery date",
     "Source": "order_estimated_delivery_date - order_purchase_timestamp",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "The promise is assigned at booking; the single most informative input."},
    {"Feature": "n_items", "Meaning": "Number of item units in the order",
     "Source": "olist_order_items_dataset", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Sets total weight/volume and picking complexity."},
    {"Feature": "n_products", "Meaning": "Distinct products in the order",
     "Source": "olist_order_items_dataset", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Multi-product orders may take longer to consolidate."},
    {"Feature": "n_sellers", "Meaning": "Distinct sellers in the order",
     "Source": "olist_order_items_dataset", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Multi-seller orders need several hand-offs."},
    {"Feature": "price_sum", "Meaning": "Total goods value (BRL)",
     "Source": "olist_order_items_dataset.price", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Proxy for basket size / service level."},
    {"Feature": "freight_sum", "Meaning": "Total freight charged (BRL)",
     "Source": "olist_order_items_dataset.freight_value", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Priced at checkout; correlates with distance and weight."},
    {"Feature": "order_value", "Meaning": "price_sum + freight_sum (BRL)",
     "Source": "derived", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Overall order size."},
    {"Feature": "freight_ratio", "Meaning": "freight_sum / order_value",
     "Source": "derived", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Relative shipping burden; a compact distance/weight proxy."},
    {"Feature": "total_weight_g", "Meaning": "Total package weight of the order (grams)",
     "Source": "olist_products_dataset.product_weight_g",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Catalog attribute; heavy parcels move more slowly."},
    {"Feature": "mean_weight_g", "Meaning": "Mean item weight (grams)",
     "Source": "olist_products_dataset.product_weight_g",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Distinguishes one heavy item from several light ones."},
    {"Feature": "total_volume_cm3", "Meaning": "Sum of product length x height x width (cm^3)",
     "Source": "olist_products_dataset dimensions",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Volumetric weight; bulky items are harder to route."},
    {"Feature": "product_category", "Meaning": "English category of the primary item",
     "Source": "olist_products_dataset + product_category_name_translation",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Logistics behaviour varies by category (fragile, bulky, oversize)."},
    {"Feature": "customer_state", "Meaning": "Customer's 2-letter Brazilian state",
     "Source": "olist_customers_dataset.customer_state",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Destination region is known from the delivery address."},
    {"Feature": "seller_state", "Meaning": "2-letter state of the primary item's seller",
     "Source": "olist_sellers_dataset.seller_state",
     "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Origin region is known from the seller record at booking."},
    {"Feature": "same_state", "Meaning": "1 if the primary seller is in the customer's state",
     "Source": "derived", "Available at booking?": "Yes", "Used?": "Yes",
     "Reason": "Intra-state lanes are typically faster and less variable."},
]

FEATURE_AUDIT += [
    {"Feature": "order_delivered_customer_date", "Meaning": "Actual delivery timestamp",
     "Source": "olist_orders_dataset", "Available at booking?": "No", "Used?": "No",
     "Reason": "Future event. Used only to build the label and the label-maturity mask."},
    {"Feature": "order_delivered_carrier_date", "Meaning": "Carrier hand-over timestamp",
     "Source": "olist_orders_dataset", "Available at booking?": "No", "Used?": "No",
     "Reason": "Post-booking event; simply not available at the prediction moment."},
    {"Feature": "order_approved_at", "Meaning": "Payment approval timestamp",
     "Source": "olist_orders_dataset", "Available at booking?": "No", "Used?": "No",
     "Reason": "Happens after booking, so it cannot be an input to a booking-time model."},
    {"Feature": "shipping_limit_date", "Meaning": "Seller shipping deadline",
     "Source": "olist_order_items_dataset", "Available at booking?": "No", "Used?": "No",
     "Reason": "Olist fills this while processing the order; booking-time availability unclear."},
    {"Feature": "order_status", "Meaning": "Final order status",
     "Source": "olist_orders_dataset", "Available at booking?": "No", "Used?": "No",
     "Reason": "Outcome variable. Knowing it would trivially leak the target."},
    {"Feature": "payment_type / payment_installments",
     "Meaning": "Payment method and number of instalments",
     "Source": "olist_order_payments_dataset", "Available at booking?": "Uncertain", "Used?": "No",
     "Reason": ("Chosen at checkout, but the payment rows are written during approval, i.e. after "
                "booking. Availability at the prediction instant is uncertain, so excluded from the "
                "primary experiment (config flag features.include_payment_features).")},
    {"Feature": "review_score / review_comment", "Meaning": "Post-delivery customer review",
     "Source": "olist_order_reviews_dataset", "Available at booking?": "No", "Used?": "No",
     "Reason": "Written after delivery; would leak the outcome."},
    {"Feature": "order_id / customer_id / product_id / seller_id",
     "Meaning": "Row identifiers", "Source": "all tables",
     "Available at booking?": "order_id only", "Used?": "No",
     "Reason": ("Not predictive of a physical process, and high-cardinality identifiers encourage "
                "memorisation. order_id is retained in the output for analysis only.")},
    {"Feature": "customer_city / seller_city", "Meaning": "City names",
     "Source": "olist_customers_dataset / olist_sellers_dataset",
     "Available at booking?": "Yes", "Used?": "No",
     "Reason": ("~4,100 cities would create a very large sparse design matrix; the state already "
                "captures most of the regional signal within this project's budget.")},
    {"Feature": "geographical distance", "Meaning": "Great-circle distance seller -> customer",
     "Source": "could be derived from olist_geolocation_dataset",
     "Available at booking?": "Technically yes", "Used?": "No",
     "Reason": ("The geolocation table only offers noisy median coordinates per zip prefix; there "
                "is no defensible booking-time distance without extra assumptions. Skipped.")},
    {"Feature": "customer_unique_id", "Meaning": "Customer identity across orders",
     "Source": "olist_customers_dataset", "Available at booking?": "Yes", "Used?": "No",
     "Reason": ("Identifier; excluded by the identifier rule, and a repeat-customer signal would "
                "not transfer to new customers.")},
]


def feature_audit_table() -> pd.DataFrame:
    """Return the feature audit as a DataFrame (used in the report and the data card)."""
    return pd.DataFrame(FEATURE_AUDIT, columns=[
        "Feature", "Meaning", "Source", "Available at booking?", "Used?", "Reason",
    ])


def feature_frame(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Select the model-input columns (never a superset of :data:`MODEL_FEATURES`).

    Payment features are one-to-many per order and are excluded from the persisted
    modelling table, so requesting them via ``features.include_payment_features``
    raises instead of silently returning a smaller matrix than the config promised.
    """
    if cfg is not None and bool(cfg["features"].get("include_payment_features")):
        raise NotImplementedError(
            "features.include_payment_features=True is not supported: payment rows are "
            "one-to-many per order and are deliberately absent from the modelling table "
            "(order_value already carries what payment rows contributed)."
        )
    cols = list(features if features is not None else MODEL_FEATURES)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in the prepared dataset: {missing}")
    return df.loc[:, cols].copy()


def build_preprocessor(
    cfg: Dict[str, Any],
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build the preprocessing transformer.

    * numeric: median imputation (fitted on train only) + optional standardisation
      (required by the linear model, pointless for trees, so it is switched on per model);
    * categorical: most-frequent imputation, then one-hot encoding with
      ``min_frequency`` merging rare levels and ``handle_unknown="infrequent_if_exist"``
      so that a category never seen during training can still be scored in production.
    """
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    n_min_freq = int(cfg["features"]["min_category_frequency"])
    n_max_levels = int(cfg["features"]["max_category_levels"])

    numeric_steps: List[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=n_min_freq,
            max_categories=n_max_levels,
            sparse_output=True,
            dtype=np.float64,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _as_list(value: Any) -> List[Any]:
    """Normalise a config value that may be a scalar or a list into a list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def model_param_grid(cfg: Dict[str, Any], model_name: str) -> Dict[str, List[Any]]:
    """Translate the YAML search space into an sklearn ``param_grid`` (nested names).

    Only knobs with more than one candidate become a searched axis; single-valued
    entries (``max_iter``, ``n_jobs``, ``class_weight``) are treated as fixed
    hyper-parameters and are applied by :func:`build_model_pipeline` instead.
    """
    base = cfg["modeling"][model_name]
    grid: Dict[str, List[Any]] = {}
    for key, value in base.items():
        values = _as_list(value)
        if len(values) > 1:
            grid[f"model__{key}"] = values
    return grid


def build_model_pipeline(
    model_name: str,
    cfg: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Pipeline:
    """Create ``Pipeline(preprocessor -> model)`` for one of the three candidates.

    ``params`` uses the flat sklearn convention (``model__max_depth`` etc.) and is
    merged over the YAML defaults, which lets the grid search override single values.
    """
    seed = int(seed if seed is not None else cfg["project"]["random_seed"])
    cfg_model = cfg["modeling"][model_name]
    params = dict(params or {})
    tuned = {}
    for key, value in params.items():
        if not key.startswith("model__"):
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        tuned[key] = value

    if model_name == "dummy":
        model = DummyClassifier(
            strategy=tuned.get("model__strategy", cfg_model["strategy"]),
            random_state=seed,
        )
        scale = False
    elif model_name == "logistic_regression":
        model = LogisticRegression(
            C=float(tuned.get("model__C", _as_list(cfg_model["C"])[0])),
            max_iter=int(tuned.get("model__max_iter", cfg_model["max_iter"])),
            class_weight=tuned.get("model__class_weight", cfg_model["class_weight"]),
            solver="liblinear",
            random_state=seed,
        )
        scale = True
    elif model_name == "random_forest":
        depth = tuned.get("model__max_depth", _as_list(cfg_model["max_depth"])[0])
        leaf = tuned.get("model__min_samples_leaf", _as_list(cfg_model["min_samples_leaf"])[0])
        model = RandomForestClassifier(
            n_estimators=int(tuned.get("model__n_estimators", cfg_model["n_estimators"])),
            max_depth=None if depth in (None, "None", "null") else int(depth),
            min_samples_leaf=int(leaf),
            max_features=tuned.get("model__max_features", _as_list(cfg_model["max_features"])[0]),
            class_weight=tuned.get("model__class_weight", cfg_model["class_weight"]),
            n_jobs=int(tuned.get("model__n_jobs", cfg_model["n_jobs"])),
            random_state=seed,
        )
        scale = False
    else:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected 'dummy', "
            "'logistic_regression' or 'random_forest'."
        )

    return Pipeline([
        ("preprocessor", build_preprocessor(cfg, scale_numeric=scale)),
        ("model", model),
    ])


def transformed_feature_names(pipeline: Pipeline) -> List[str]:
    """Names of the columns the model actually sees (after imputation + one-hot)."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
