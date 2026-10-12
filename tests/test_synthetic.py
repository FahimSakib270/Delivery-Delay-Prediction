"""Tests for the synthetic data generator.

The generator is a *fallback*, but the numbers computed on it are still printed in
reports, so it has to be honest about three things: it must be deterministic, it must
be Olist-shaped, and its late rate must be plausible. The plausibility tests exist
because an earlier version of this file contained a bug that made ~50% of
November/December orders late: results were still produced, they were just nonsense.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import synthetic

#: ``generate_tables`` keys its output by CSV file name, which is also the name the
#: loader looks for. This is the only key the tests here need by hand.
ORDERS = "olist_orders_dataset"
ITEMS = "olist_order_items_dataset"


# --------------------------------------------------------------------------- schema
def test_generate_tables_returns_every_expected_table(synthetic_tables):
    """The generator must emit exactly the tables the loader expects.

    ``generate_tables`` keys its output by CSV file name (``olist_orders_dataset``),
    and :data:`src.data_prep.TABLE_FILES` maps those files to the logical names the
    pipeline speaks (``orders``). If the two ever disagree, either the CSVs are written
    with the wrong names or a table silently disappears - so the contract is asserted
    here rather than discovered later.
    """
    from src.data_prep import TABLE_FILES

    for name in TABLE_FILES:
        assert name in synthetic_tables, f"missing logical table: {name}"
        assert len(synthetic_tables[name]) > 0

    raw = synthetic.generate_tables(n_orders=200, seed=4)
    assert set(raw) == {Path(filename).stem for filename in TABLE_FILES.values()}, (
        "the generator's keys must be exactly the CSV stems the loader looks for"
    )
    assert "order_id" in raw["olist_orders_dataset"].columns


def test_orders_table_has_the_olist_columns_the_pipeline_reads(synthetic_tables):
    orders = synthetic_tables["orders"]
    for column in (
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        assert column in orders.columns, f"missing order column: {column}"


def test_keys_are_unique_and_foreign_keys_resolve(synthetic_tables):
    orders = synthetic_tables["orders"]
    items = synthetic_tables["items"]
    customers = synthetic_tables["customers"]

    assert orders["order_id"].is_unique
    assert customers["customer_id"].is_unique, "customers must be 1:1 with orders"
    assert set(orders["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(items["order_id"]).issubset(set(orders["order_id"]))
    assert set(items["product_id"]).issubset(set(synthetic_tables["products"]["product_id"]))
    assert set(items["seller_id"]).issubset(set(synthetic_tables["sellers"]["seller_id"]))


def test_customer_and_seller_id_spaces_are_disjoint(synthetic_tables):
    """A vendor-ID collision would let a join silently return wrong rows.

    ``orders.customer_id`` must join ``customers.customer_id`` and *only* that, so the
    generator uses distinct id prefixes per entity, exactly like Olist does.
    """
    customer_ids = set(synthetic_tables["customers"]["customer_id"])
    seller_ids = set(synthetic_tables["sellers"]["seller_id"])
    assert not (customer_ids & seller_ids)


# --------------------------------------------------------------------------- physics
def test_timestamp_ordering_is_internally_consistent(synthetic_tables):
    orders = synthetic_tables["orders"]
    purchase = pd.to_datetime(orders["order_purchase_timestamp"])
    approved = pd.to_datetime(orders["order_approved_at"])
    carrier = pd.to_datetime(orders["order_delivered_carrier_date"])
    delivered = pd.to_datetime(orders["order_delivered_customer_date"])
    promised = pd.to_datetime(orders["order_estimated_delivery_date"])

    # Some timestamps are deliberately missing (canceled orders, unresolved deliveries),
    # which is what makes ``clean_orders`` worth having. Only complete rows can be
    # checked for ordering, and the missingness must stay a small minority.
    complete = approved.notna() & carrier.notna() & delivered.notna()
    assert complete.mean() > 0.95, "more than 5% of orders are missing a timestamp"
    assert promised.notna().all(), "the promise is mandatory: it defines the label"

    assert (approved[complete] >= purchase[complete]).all(), "approval cannot precede booking"
    assert (carrier[complete] >= approved[complete]).all(), "carrier hand-over cannot precede approval"
    assert (delivered[complete] >= carrier[complete]).all(), "delivery cannot precede the carrier hand-over"
    assert (promised[complete] > purchase[complete]).all(), "the promise is made after the booking"


def test_promised_dates_are_plausible_days_ahead(synthetic_tables):
    orders = synthetic_tables["orders"]
    lead = (
        pd.to_datetime(orders["order_estimated_delivery_date"])
        - pd.to_datetime(orders["order_purchase_timestamp"])
    ).dt.total_seconds() / 86400.0
    assert lead.min() > 0
    assert 8.0 <= lead.median() <= 45.0, f"median promise lead {lead.median():.1f} days is implausible"


def test_november_december_is_not_a_different_dataset(big_tables):
    """Regression guard for the 50%-late-in-November bug.

    The peak effect on the journey must stay in the same order of magnitude as the
    promise padding; when it does not, one seasonal term dominates every metric.
    """
    orders = big_tables["orders"]
    purchase = pd.to_datetime(orders["order_purchase_timestamp"])
    promised = pd.to_datetime(orders["order_estimated_delivery_date"])
    delivered = pd.to_datetime(orders["order_delivered_customer_date"])
    ok = delivered.notna() & promised.notna()
    late = (delivered[ok].dt.normalize() > promised[ok].dt.normalize()).to_numpy()
    month = purchase[ok].dt.month.to_numpy()

    peak = late[np.isin(month, (11, 12))]
    off_peak = late[~np.isin(month, (11, 12))]
    peak_rate, off_peak_rate = float(peak.mean()), float(off_peak.mean())

    assert 0.02 <= off_peak_rate <= 0.20, f"off-peak late rate {off_peak_rate:.3f} is implausible"
    assert peak_rate < 0.30, f"peak late rate {peak_rate:.3f} means the peak term dominates everything"
    assert peak_rate > 0, "no peak season at all would change the reported story"


def test_the_peak_term_stays_the_same_order_of_magnitude_as_the_padding():
    """A unit-level guard: the peak effect is expressed on both scales."""
    journey_days = synthetic.PEAK_JOURNEY_WEIGHT * synthetic.TRANSIT_DIFFICULTY_DAYS
    assert 0.0 < journey_days <= 2.5 * synthetic.PROMISE_PEAK_DAYS, (
        "the November/December journey penalty must not dwarf the promise padding "
        f"({journey_days:.1f} transit days vs {synthetic.PROMISE_PEAK_DAYS:.1f} padded days)"
    )


def test_the_generator_produces_a_minority_late_class(big_tables):
    orders = big_tables["orders"]
    promised = pd.to_datetime(orders["order_estimated_delivery_date"])
    delivered = pd.to_datetime(orders["order_delivered_customer_date"])
    ok = delivered.notna() & promised.notna()
    late_rate = float((delivered[ok].dt.normalize() > promised[ok].dt.normalize()).mean())
    assert 0.02 <= late_rate <= 0.20, f"late rate {late_rate:.4f} is not a plausible delayed-delivery rate"


# --------------------------------------------------------------------------- determinism
def test_generation_is_reproducible_for_a_fixed_seed():
    first = synthetic.generate_tables(n_orders=300, seed=99)
    second = synthetic.generate_tables(n_orders=300, seed=99)
    assert first[ORDERS].equals(second[ORDERS])
    assert first[ITEMS].equals(second[ITEMS])


def test_a_different_seed_changes_the_data():
    first = synthetic.generate_tables(n_orders=300, seed=1)
    second = synthetic.generate_tables(n_orders=300, seed=2)
    assert not first[ORDERS].equals(second[ORDERS])


# --------------------------------------------------------------------------- provenance
def test_generator_fingerprint_is_the_file_hash_and_is_stable():
    from src.utils import file_sha256

    fingerprint = synthetic.generator_fingerprint()
    assert len(fingerprint) == 64
    assert fingerprint == file_sha256(synthetic.__file__)
    assert synthetic.generator_fingerprint() == fingerprint


def test_write_synthetic_dataset_writes_csvs_and_a_provenance_marker(tmp_path):
    from src.data_prep import TABLE_FILES

    out = synthetic.write_synthetic_dataset(tmp_path / "synth", n_orders=400, seed=3)

    for filename in TABLE_FILES.values():
        assert (out / filename).is_file(), f"{filename} was not written"

    provenance = synthetic.read_provenance(out)
    assert provenance["kind"] == "synthetic"
    assert provenance["seed"] == 3
    assert provenance["n_orders_requested"] == 400
    assert provenance["generator_fingerprint"] == synthetic.generator_fingerprint()
    assert "SIMULATED" in provenance["warning"]


def test_read_provenance_returns_an_empty_mapping_when_absent(tmp_path):
    assert synthetic.read_provenance(tmp_path) == {}
