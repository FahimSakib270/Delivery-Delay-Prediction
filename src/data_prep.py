"""Data loading, validation, joining, labelling and feature construction.

Design rules enforced here
--------------------------
1. **One row per order.** All many-to-one joins (items -> order, products/sellers
   -> item) are validated for cardinality so multi-item orders can never silently
   duplicate training examples.
2. **No future information.** Only fields known at the booking instant (plus the
   promised delivery date, which is assigned at booking) may become model inputs.
   ``order_approved_at`` (post-booking), ``order_delivered_carrier_date``,
   ``order_delivered_customer_date``, ``order_status`` and payment rows are
   blocked from the feature set. The actual delivery timestamp is used only to
   build the label and the label-maturity mask.
3. **Everything is counted.** Every row dropped is recorded in a statistics dict
   that ends up in the run manifest, so the reported sample size is auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .utils import dataframe_fingerprint, ensure_dir, get_logger, save_json, set_seed

logger = get_logger("data_prep")

# File names as published by the "Brazilian E-Commerce Public Dataset by Olist".
TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "translation": "product_category_name_translation.csv",
}
# Only these are strictly required; payments/translation are optional extras.
REQUIRED_TABLES = ("orders", "items", "products", "customers", "sellers")

# Columns that must NEVER reach the model, kept here as a testable list.
PROHIBITED_MODEL_COLUMNS = (
    "order_id", "customer_id", "customer_unique_id", "product_id", "seller_id",
    "order_status", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "shipping_limit_date", "payment_sequential", "payment_type",
    "payment_installments", "payment_value",
    "is_late", "is_late_ts", "delay_days", "delivered_date",
)


# --------------------------------------------------------------------------
# Locating and reading the raw tables
# --------------------------------------------------------------------------
def find_raw_dir(cfg: Dict[str, Any]) -> Tuple[Path, str]:
    """Return ``(directory, kind)`` for the raw data, honouring ``data_source.mode``.

    ``kind`` is ``"olist"`` for the real public dataset and ``"synthetic"`` for the
    generated demonstration data. The distinction is carried through the whole
    pipeline so results can never be confused with one another.
    """
    olist_dir = resolve(cfg, "paths", "raw_dir") / "olist"
    synthetic_dir = resolve(cfg, "paths", "synthetic_raw_dir")
    mode = str(cfg["data_source"]["mode"]).lower()

    olist_ok = (olist_dir / TABLE_FILES["orders"]).exists()
    if mode == "olist":
        if not olist_ok:
            raise FileNotFoundError(
                f"data_source.mode='olist' but {olist_dir / TABLE_FILES['orders']} was not found. "
                "Download the Kaggle dataset and place the CSV files in data/raw/olist/ "
                "(see README.md, section 'Getting the real data')."
            )
        return olist_dir, "olist"
    if mode == "synthetic":
        return synthetic_dir, "synthetic"
    if mode != "auto":
        raise ValueError(f"Unknown data_source.mode: {mode!r}")
    return (olist_dir, "olist") if olist_ok else (synthetic_dir, "synthetic")


def ensure_synthetic_raw(raw_dir: Path, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Generate the synthetic tables if they are missing *or stale*.

    Returns ``True`` when the CSVs were (re)generated. The staleness check matters:
    the CSVs are cached on disk, so without comparing the generator fingerprint and
    the requested ``(seed, n_orders)`` a change to ``src/synthetic.py`` would be
    silently ignored and the pipeline would keep reporting numbers from a generator
    that no longer exists.
    """
    cfg = cfg or load_config()
    from .synthetic import generator_fingerprint, read_provenance, write_synthetic_dataset

    n_orders = int(cfg["data_source"]["synthetic_n_orders"])
    seed = int(cfg["data_source"]["synthetic_seed"])
    marker_exists = (raw_dir / TABLE_FILES["orders"]).exists()

    reason = None
    if not marker_exists:
        reason = "no raw data found"
    else:
        provenance = read_provenance(raw_dir)
        if provenance.get("generator_fingerprint") != generator_fingerprint():
            reason = "data/raw/synthetic was written by a different version of src/synthetic.py"
        elif int(provenance.get("seed", -1)) != seed:
            reason = f"dataset seed changed ({provenance.get('seed')} -> {seed})"
        elif int(provenance.get("n_orders_requested", -1)) != n_orders:
            reason = f"synthetic_n_orders changed ({provenance.get('n_orders_requested')} -> {n_orders})"

    if reason is None:
        return False

    logger.warning("Generating the SYNTHETIC demonstration dataset (%s).", reason)
    write_synthetic_dataset(raw_dir, n_orders=n_orders, seed=seed)
    return True


def load_raw(raw_dir: Path, kind: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """Read the CSV tables from ``raw_dir``, generating synthetic data if needed."""
    cfg = cfg or load_config()
    if kind == "synthetic":
        ensure_synthetic_raw(raw_dir, cfg)

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            if key in REQUIRED_TABLES:
                raise FileNotFoundError(f"Required table missing: {path}")
            logger.warning("Optional table missing, continuing without it: %s", path)
            continue
        tables[key] = pd.read_csv(path, dtype_backend="numpy_nullable")
        logger.info(
            "Loaded %-12s %8d rows x %2d cols from %s",
            key, len(tables[key]), tables[key].shape[1], filename,
        )
    return tables


# --------------------------------------------------------------------------
# Order-level cleaning and target construction
# --------------------------------------------------------------------------
def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse a timestamp column, turning unparsable values into ``NaT``.

    ``errors="coerce"`` is used deliberately: malformed values are *quarantined*
    (counted and dropped) rather than silently becoming the epoch.
    """
    return pd.to_datetime(series, errors="coerce", format="mixed")


def clean_orders(orders_raw: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Validate order rows and drop unusable records, counting each exclusion."""
    df = orders_raw.copy()
    stats["raw_orders"] = int(len(df))

    # --- duplicate order ids -------------------------------------------------
    dup_mask = df["order_id"].duplicated(keep="first")
    stats["dropped_duplicate_order_id"] = int(dup_mask.sum())
    df = df.loc[~dup_mask]

    # --- timestamp parsing ---------------------------------------------------
    time_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in time_cols:
        if col in df.columns:
            before_na = int(df[col].isna().sum())
            df[col] = _to_datetime(df[col])
            stats[f"unparsable_{col}"] = int(df[col].isna().sum()) - before_na
        else:
            df[col] = pd.NaT

    # --- booking timestamp and promised date are mandatory --------------------
    bad_dates = df["order_purchase_timestamp"].isna() | df["order_estimated_delivery_date"].isna()
    stats["dropped_missing_booking_or_promise_date"] = int(bad_dates.sum())
    df = df.loc[~bad_dates]

    # --- the promise must not precede the booking ----------------------------
    promise_before_booking = df["order_estimated_delivery_date"] < df["order_purchase_timestamp"]
    stats["dropped_promise_before_booking"] = int(promise_before_booking.sum())
    df = df.loc[~promise_before_booking]

    # --- terminated orders have no outcome to learn from ---------------------
    status = df["order_status"].astype("string").str.lower()
    terminated = status.isin(["canceled", "cancelled", "unavailable"])
    stats["dropped_canceled_or_unavailable"] = int(terminated.sum())
    df = df.loc[~terminated]

    # --- unresolved orders (no delivery timestamp) ---------------------------
    unresolved = df["order_delivered_customer_date"].isna()
    stats["dropped_unresolved_no_delivery_date"] = int(unresolved.sum())
    df = df.loc[~unresolved]

    # --- impossible delivery timestamps --------------------------------------
    delivered_before_booking = df["order_delivered_customer_date"] < df["order_purchase_timestamp"]
    stats["dropped_delivery_before_booking"] = int(delivered_before_booking.sum())
    df = df.loc[~delivered_before_booking]

    stats["usable_orders"] = int(len(df))
    return df.reset_index(drop=True)


def add_target(df: pd.DataFrame, label_definition: str = "date_based") -> pd.DataFrame:
    """Attach the binary delay target and the analysis-only delay magnitude.

    Primary definition (``date_based``)
    -----------------------------------
    ``is_late = 1`` iff the **calendar date** of the actual customer delivery is
    strictly later than the **calendar date** promised at booking. A parcel that
    arrives on the promised day counts as 0 (on time).

    Why calendar dates? The promised delivery date is published to the customer as
    a *day* ("arrives by 12 March"), and Olist's estimated-delivery column carries a
    time-of-day that is a system artefact rather than a customer-facing promise.
    Comparing raw timestamps would treat a parcel delivered at 09:00 on the
    promised day as on time but the same parcel at 23:00 as late, which is not a
    meaningful distinction for this research question. The timestamp comparison is
    kept as a documented sensitivity check (``is_late_ts``).
    """
    if label_definition not in {"date_based", "timestamp_based"}:
        raise ValueError(f"Unknown label definition: {label_definition!r}")

    out = df.copy()
    out["promised_date"] = out["order_estimated_delivery_date"].dt.normalize()
    out["delivered_date"] = out["order_delivered_customer_date"].dt.normalize()

    out["is_late"] = (out["delivered_date"] > out["promised_date"]).astype("int8")
    out["is_late_ts"] = (
        out["order_delivered_customer_date"] > out["order_estimated_delivery_date"]
    ).astype("int8")
    out["delay_days"] = (out["delivered_date"] - out["promised_date"]).dt.days.astype("float64")

    # Booking-time quantities that follow directly from the promise.
    out["lead_time_days"] = (
        out["promised_date"] - out["order_purchase_timestamp"].dt.normalize()
    ).dt.days
    out["purchase_month"] = out["order_purchase_timestamp"].dt.month.astype("int16")
    out["purchase_weekday"] = out["order_purchase_timestamp"].dt.dayofweek.astype("int16")
    out["purchase_hour"] = out["order_purchase_timestamp"].dt.hour.astype("int16")
    return out


# --------------------------------------------------------------------------
# Item -> order aggregation (this is where multi-item orders must not duplicate)
# --------------------------------------------------------------------------
def _report_duplicate_keys(df: pd.DataFrame, key: str, name: str, stats: Dict[str, Any]) -> pd.DataFrame:
    """Warn about, and remove, duplicate join keys on the *right-hand* side."""
    n_dup = int(df[key].duplicated().sum())
    stats[f"duplicate_keys_{name}"] = n_dup
    if n_dup:
        logger.warning("%s has %d duplicate %r values; keeping the first of each.", name, n_dup, key)
        df = df.drop_duplicates(subset=[key], keep="first")
    return df


def aggregate_items(
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    translation: Optional[pd.DataFrame],
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Collapse the order-items table to exactly one row per ``order_id``.

    Aggregation rule for orders with several products/sellers (documented because
    it is a modelling decision, not an implementation detail):

    * ``n_items``  : number of item rows (units bought).
    * ``n_products`` / ``n_sellers`` : distinct products / distinct sellers.
    * ``price_sum``, ``freight_sum``, ``total_weight_g``, ``total_volume_cm3`` : sums.
    * ``mean_weight_g`` : mean item weight.
    * ``product_category`` : category of the item with the lowest ``order_item_id``
      (the "primary" item). A single dominant category keeps cardinality low; the
      alternative (one row per item) would duplicate the order and is forbidden.
    * ``seller_state`` : state of the primary item's seller, so ``same_state`` means
      "the primary seller is in the same state as the customer".
    """
    it = items.copy()
    n_before = len(it)

    prod = _report_duplicate_keys(
        products[["product_id", "product_category_name", "product_weight_g",
                  "product_length_cm", "product_height_cm", "product_width_cm"]],
        "product_id", "products", stats,
    )
    it = it.merge(prod, on="product_id", how="left", validate="many_to_one")
    if len(it) != n_before:
        raise AssertionError("products join changed the item row count")

    sell = _report_duplicate_keys(sellers[["seller_id", "seller_state"]], "seller_id", "sellers", stats)
    it = it.merge(sell, on="seller_id", how="left", validate="many_to_one")
    if len(it) != n_before:
        raise AssertionError("sellers join changed the item row count")

    if translation is not None and {"product_category_name", "product_category_name_english"} <= set(translation.columns):
        tr = _report_duplicate_keys(translation, "product_category_name", "translation", stats)
        it = it.merge(tr, on="product_category_name", how="left", validate="many_to_one")
        it["product_category"] = it["product_category_name_english"].fillna(it["product_category_name"])
    else:
        it["product_category"] = it["product_category_name"]

    # Keep the English label when available; leave NaN for the imputer to handle.
    it["product_category"] = it["product_category"].astype("string")

    it["volume_cm3"] = (
        pd.to_numeric(it["product_length_cm"], errors="coerce")
        * pd.to_numeric(it["product_height_cm"], errors="coerce")
        * pd.to_numeric(it["product_width_cm"], errors="coerce")
    )

    it = it.sort_values(["order_id", "order_item_id"], kind="stable")
    grouped = it.groupby("order_id", sort=True)

    agg = grouped.agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        mean_weight_g=("product_weight_g", "mean"),
        total_volume_cm3=("volume_cm3", "sum"),
    )
    # First non-null value inside each order == the primary item's value, because
    # the frame was sorted by order_item_id before grouping.
    primary = grouped[["seller_state", "product_category"]].first()
    agg = agg.join(primary)

    assert agg.index.is_unique, "aggregation must produce exactly one row per order"
    stats["item_rows_in"] = int(n_before)
    stats["orders_after_item_aggregation"] = int(len(agg))
    stats["multi_item_orders"] = int((agg["n_items"] > 1).sum())
    stats["multi_seller_orders"] = int((agg["n_sellers"] > 1).sum())
    return agg.reset_index()


# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------
#: Numeric model inputs. Every one of these is knowable at the booking instant.
NUMERIC_FEATURES = [
    "purchase_month",       # 1-12, seasonality
    "purchase_weekday",     # 0=Monday
    "purchase_hour",        # 0-23, local ordering hour
    "lead_time_days",       # promised date - booking date (the headline feature)
    "n_items",              # units in the order
    "n_products",           # distinct products
    "n_sellers",            # distinct sellers
    "price_sum",            # BRL, goods value
    "freight_sum",          # BRL, freight charged
    "order_value",          # BRL, price_sum + freight_sum
    "freight_ratio",        # freight_sum / order_value
    "total_weight_g",       # grams
    "mean_weight_g",        # grams
    "total_volume_cm3",     # cm^3
    "same_state",           # 1 if the primary seller is in the customer's state
]

#: Categorical model inputs (encoded by the pipeline, unseen levels handled).
CATEGORICAL_FEATURES = [
    "customer_state",       # 2-letter Brazilian state
    "seller_state",         # 2-letter Brazilian state of the primary seller
    "product_category",     # English category of the primary item
]

#: Analysis-only columns: needed for the chronological split, the label-maturity
#: filter and the error analysis, but never fed to the model.
ANALYSIS_COLUMNS = [
    "order_id", "customer_id", "customer_state",
    "order_status", "order_purchase_timestamp",
    "order_estimated_delivery_date", "order_delivered_customer_date",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "is_late"


def build_order_dataset(
    tables: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Run the full one-row-per-order preparation and return the modelling frame."""
    label_definition = str(cfg["label"]["primary"])

    orders = clean_orders(tables["orders"], stats)
    orders = add_target(orders, label_definition=label_definition)

    # --- customers: 1:1 per order in the Olist schema -------------------------
    customers = tables["customers"]
    keep = [c for c in ["customer_id", "customer_state", "customer_city", "customer_zip_code_prefix"]
            if c in customers.columns]
    customers = _report_duplicate_keys(customers[keep], "customer_id", "customers", stats)

    n_before = len(orders)
    orders = orders.merge(customers, on="customer_id", how="left", validate="one_to_one")
    if len(orders) != n_before:
        raise AssertionError("customers join changed the order row count")
    stats["orders_missing_customer_row"] = int(orders["customer_state"].isna().sum())
    # A join keyed on the wrong column fails silently: nothing raises, the feature is
    # simply always missing. That is how a whole attribute can disappear from a model
    # without anyone noticing, so the share is audited loudly here.
    share_missing_customer = stats["orders_missing_customer_row"] / max(1, len(orders))
    if share_missing_customer > 0.01:
        logger.warning(
            "%.1f%% of orders have no matching customer row - check the customer join key",
            100 * share_missing_customer,
        )

    # --- item aggregates ------------------------------------------------------
    agg = aggregate_items(
        items=tables["items"],
        products=tables["products"],
        sellers=tables["sellers"],
        translation=tables.get("translation"),
        stats=stats,
    )
    orders = orders.merge(agg, on="order_id", how="left", validate="one_to_one")

    no_items = orders["n_items"].isna()
    stats["dropped_orders_without_items"] = int(no_items.sum())
    orders = orders.loc[~no_items].copy()

    # --- derived booking-time quantities --------------------------------------
    orders["price_sum"] = pd.to_numeric(orders["price_sum"], errors="coerce")
    orders["freight_sum"] = pd.to_numeric(orders["freight_sum"], errors="coerce")
    orders["order_value"] = orders["price_sum"] + orders["freight_sum"]
    orders["freight_ratio"] = np.where(
        orders["order_value"] > 0, orders["freight_sum"] / orders["order_value"], np.nan
    )
    # ``same_state`` is a 0/1 booking-time feature. A missing customer or seller row
    # yields ``<NA>`` from the comparison, which cannot be cast to an integer feature
    # (pandas raises for nullable/Arrow booleans), so it is resolved to 0 with an
    # explicit comment: "unknown route" is treated the same as "different state",
    # never as "same state".
    same_state = (
        orders["customer_state"].astype("string") == orders["seller_state"].astype("string")
    )
    orders["same_state"] = same_state.fillna(False).astype("int8")

    orders = orders.sort_values("order_purchase_timestamp", kind="stable").reset_index(drop=True)
    stats["final_orders"] = int(len(orders))
    stats["final_late_rate"] = float(orders["is_late"].mean())
    stats["final_late_rate_timestamp_definition"] = float(orders["is_late_ts"].mean())
    stats["date_range_start"] = orders["order_purchase_timestamp"].min()
    stats["date_range_end"] = orders["order_purchase_timestamp"].max()
    return orders


def audit_missingness(df: pd.DataFrame, columns: list[str]) -> Dict[str, Dict[str, float]]:
    """Missing-value audit for the model inputs (reporting only, no filling here)."""
    audit: Dict[str, Dict[str, float]] = {}
    for col in columns:
        n_missing = int(df[col].isna().sum())
        audit[col] = {
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / max(1, len(df)), 4),
        }
    return audit


def chronological_bounds(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience helper returning ISO booking-time bounds of a frame."""
    ts = df["order_purchase_timestamp"]
    return {"start": ts.min().isoformat(), "end": ts.max().isoformat(), "n": int(len(df))}


# --------------------------------------------------------------------------
# Top-level preparation entry point
# --------------------------------------------------------------------------
def prepare_dataset(
    cfg: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load -> clean -> join -> label -> audit -> persist. Returns ``(df, metadata)``."""
    from .utils import file_sha256  # local import keeps the module header short

    cfg = cfg or load_config()
    set_seed(int(cfg["project"]["random_seed"]))
    stats: Dict[str, Any] = {}

    raw_dir, kind = find_raw_dir(cfg)
    stats["data_source_kind"] = kind
    stats["raw_dir"] = str(raw_dir)
    stats["label_definition"] = str(cfg["label"]["primary"])

    tables = load_raw(raw_dir, kind, cfg)
    # Synthetic files may have been generated by load_raw; hash the actual inputs
    # after generation rather than recording null checksums on a clean run.
    stats["raw_file_sha256"] = {
        key: file_sha256(raw_dir / filename) for key, filename in TABLE_FILES.items()
    }
    stats["raw_row_counts"] = {key: int(len(frame)) for key, frame in tables.items()}

    df = build_order_dataset(tables, cfg, stats)

    stats["missingness_model_features"] = audit_missingness(df, MODEL_FEATURES)
    stats["model_features"] = list(MODEL_FEATURES)
    stats["numeric_features"] = list(NUMERIC_FEATURES)
    stats["categorical_features"] = list(CATEGORICAL_FEATURES)
    stats["class_distribution"] = {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()}
    stats["late_rate"] = float(df[TARGET_COLUMN].mean())
    stats["dataset_fingerprint"] = dataframe_fingerprint(
        df, ["order_id", "order_purchase_timestamp", TARGET_COLUMN]
    )
    stats["booking_period"] = chronological_bounds(df)
    stats["synthetic"] = kind == "synthetic"

    out_dir = ensure_dir(output_dir or resolve(cfg, "paths", "processed_dir"))
    data_path = out_dir / "orders_prepared.csv"
    df.to_csv(data_path, index=False)
    stats["processed_path"] = str(data_path)
    stats["processed_sha256"] = file_sha256(data_path)

    meta = {
        "provenance": {
            "data_source_kind": kind,
            "synthetic": kind == "synthetic",
            "raw_dir": str(raw_dir),
            "raw_file_sha256": stats["raw_file_sha256"],
            "processed_sha256": stats["processed_sha256"],
            "dataset_fingerprint": stats["dataset_fingerprint"],
        },
        "config": cfg,
        "statistics": stats,
    }
    save_json(meta, out_dir / "dataset_metadata.json")

    logger.info("Prepared dataset: %d orders, late rate %.4f, booking window %s .. %s",
                len(df), stats["late_rate"],
                stats["booking_period"]["start"][:10], stats["booking_period"]["end"][:10])
    return df, meta


def load_processed(path: Optional[Path] = None, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load the prepared CSV, raising a friendly error if preparation has not run."""
    cfg = cfg or load_config()
    data_path = Path(path) if path is not None else resolve(cfg, "paths", "processed_dir") / "orders_prepared.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run 'python -m src.prepare' first."
        )
    df = pd.read_csv(data_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["promised_date"] = pd.to_datetime(df["promised_date"], errors="coerce")
    df["delivered_date"] = pd.to_datetime(df["delivered_date"], errors="coerce")
    return df


def main(cfg_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """CLI entry point: ``python -m src.prepare``."""
    cfg = load_config(cfg_path)
    return prepare_dataset(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    parser = argparse.ArgumentParser(description="Prepare the delivery-delay dataset.")
    parser.add_argument("--config", default=None, help="Path to an alternative config YAML.")
    _args = parser.parse_args()
    main(_args.config)
