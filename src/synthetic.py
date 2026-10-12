"""Synthetic demonstration dataset with the *exact* Olist table schema.

Why this file exists
--------------------
The project prefers the real Brazilian Olist e-commerce data. If that data is not
available in ``data/raw/olist/`` the code must still be runnable end-to-end and
the tests must still be meaningful. This generator produces tables using the real
Olist file names and column names (including Olist's original typos such as
``product_name_lenght``), so the *identical* preparation code path runs.

IMPORTANT HONESTY NOTE
----------------------
These rows are simulated. They encode an invented delay mechanism and therefore
say nothing about ProFast, Bangladesh, or Brazilian logistics. Any metric
obtained on this dataset is a **software smoke test**, not evidence of
real-world predictive skill. The synthetic dataset is labelled as such
everywhere: in ``PROVENANCE.json``, the run manifest, the reports and the app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .utils import ensure_dir, file_sha256, get_logger, load_json, save_json

logger = get_logger("synthetic")

# --- Brazilian states with a rough "logistics difficulty" weight --------------
# 0 = well connected, 1 = remote/hard. This ordering is invented for realism and
# is NOT a measured logistics index. It only shapes the simulated data.
STATES: Dict[str, float] = {
    "SP": 0.05, "RJ": 0.15, "MG": 0.15, "PR": 0.15, "SC": 0.18, "RS": 0.22,
    "ES": 0.25, "GO": 0.30, "DF": 0.30, "MT": 0.40, "MS": 0.40, "BA": 0.40,
    "PE": 0.48, "CE": 0.50, "RN": 0.55, "PB": 0.55, "AL": 0.58, "SE": 0.58,
    "PI": 0.62, "MA": 0.68, "TO": 0.65, "PA": 0.72, "RO": 0.78, "AC": 0.85,
    "AM": 0.88, "AP": 0.90, "RR": 0.92,
}
STATE_NAMES = list(STATES)
# Sampling weights roughly follow the population/economic concentration of the
# real Olist sample (Sao Paulo dominates).
STATE_WEIGHTS = np.array([
    0.42, 0.13, 0.11, 0.05, 0.04, 0.04, 0.02, 0.03, 0.02, 0.02, 0.01, 0.03,
    0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005,
    0.01, 0.005, 0.002,
])
STATE_WEIGHTS = STATE_WEIGHTS / STATE_WEIGHTS.sum()

CITIES = {
    "SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte",
    "PR": "curitiba", "SC": "florianopolis", "RS": "porto alegre",
    "ES": "vitoria", "GO": "goiania", "DF": "brasilia", "MT": "cuiaba",
    "MS": "campo grande", "BA": "salvador", "PE": "recife", "CE": "fortaleza",
    "RN": "natal", "PB": "joao pessoa", "AL": "maceio", "SE": "aracaju",
    "PI": "teresina", "MA": "sao luis", "TO": "palmas", "PA": "belem",
    "RO": "porto velho", "AC": "rio branco", "AM": "manaus", "AP": "macapa",
    "RR": "boa vista",
}

# Real Olist category names, paired with the official English translation.
CATEGORIES = [
    ("beleza_saude", "health_beauty"), ("esporte_lazer", "sports_leisure"),
    ("moveis_decoracao", "furniture_decor"), ("informatica_acessorios", "computers_accessories"),
    ("utilidades_domesticas", "housewares"), ("relogios_presentes", "watches_gifts"),
    ("telefonia", "telephony"), ("ferramentas_jardim", "garden_tools"),
    ("automotivo", "auto"), ("brinquedos", "toys"),
    ("cool_stuff", "cool_stuff"), ("perfumaria", "perfumery"),
    ("bebes", "baby"), ("eletronicos", "electronics"),
    ("papelaria", "stationery"), ("fashion_bolsas_e_acessorios", "fashion_bags_accessories"),
    ("pet_shop", "pet_shop"), ("moveis_escritorio", "office_furniture"),
    ("consoles_games", "consoles_games"),
    ("construcao_ferramentas_construcao", "construction_tools_construction"),
    ("malas_acessorios", "luggage_accessories"), ("casa_construcao", "home_construction"),
    ("livros_interesse_geral", "books_general_interest"),
    ("instrumentos_musicais", "musical_instruments"),
    ("eletrodomesticos", "home_appliances"),
    ("agro_industria_e_comercio", "agro_industry_and_commerce"),
    ("industria_comercio_e_negocios", "industry_commerce_and_business"),
    ("audio", "audio"), ("livros_tecnicos", "books_technical"),
    ("artigos_de_festas", "party_supplies"),
]
# Typical packed weight per unit for each category above (kilograms).
CATEGORY_WEIGHT_KG = np.array([
    0.6, 1.2, 9.0, 0.7, 1.5, 0.3, 0.4, 3.5, 4.0, 1.0,
    1.0, 0.4, 2.0, 1.5, 0.4, 0.9, 2.5, 12.0, 0.8, 5.0,
    1.0, 6.0, 0.5, 3.0, 15.0, 8.0, 6.0, 1.0, 0.5, 0.8,
])

BASE_FREIGHT = 13.0        # BRL base handling + first leg, roughly Olist-like
FREIGHT_PER_KG = 2.4       # BRL per kilogram
CROSS_STATE_PENALTY = 18.0 # BRL added when the seller ships out of state

# --- promised ("estimated") date knobs ---------------------------------------
# The generator is only useful if its late rate is plausible; otherwise every
# downstream metric is noise dressed up as a result. These constants shape the
# promise and the true journey. PROMISE_INTERCEPT_DAYS was fixed by a one-off grid
# search over the resulting late rate (recorded in reports/data_card.md):
#     12.0 -> 24.5%   14.0 -> 14.5%   15.5 -> 9.7%   16.0 -> 8.2%   16.5 -> 6.9%
# 16.5 was chosen: overall late rate 6.9% (6.7% outside November/December, 8.7%
# inside it), median promise ~21 days, median actual delivery ~13 days, which is the
# same broad shape as public Olist analyses. The mechanism itself - harder routes get
# a slightly longer promise but take a *disproportionately* longer journey, so delay
# stays predictable in principle - is invented, not measured.
PROMISE_INTERCEPT_DAYS = 16.5      # calibrated: see the grid above
PROMISE_DIFFICULTY_DAYS = 14.0     # promise stretches only partly with difficulty
PROMISE_PEAK_DAYS = 3.5            # extra padding during the Nov/Dec peak
PROMISE_NOISE_DAYS = 2.5           # promise jitter around the latent expectation

# --- true journey knobs (what actually happens) ------------------------------
TRANSIT_BASE_DAYS = 4.0
TRANSIT_DIFFICULTY_DAYS = 26.0     # steeper than the promise slope on purpose
# Nov/Dec effect on the *journey*, in the same "difficulty" units as ``journey``
# below: 0.17 x 26 days ~= 4.4 days of extra transit against 3.5 days of extra
# promise padding, so peak months come out slightly worse - the realistic direction -
# without dominating the dataset. An earlier version used 0.55 (+14 transit days)
# and produced a 50% late rate in November/December against 2% elsewhere: not a
# plausible dataset but a disguised bug.
PEAK_JOURNEY_WEIGHT = 0.17

def generate_tables(
    n_orders: int = 60000,
    seed: int = 20240101,
    start: str = "2017-01-01",
    end: str = "2018-08-31",
) -> Dict[str, pd.DataFrame]:
    """Generate Olist-shaped tables. Returns a mapping table-name -> DataFrame."""
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- sellers --
    n_sellers = 300
    seller_state = rng.choice(STATE_NAMES, size=n_sellers, p=STATE_WEIGHTS)
    sellers = pd.DataFrame({
        "seller_id": [f"synth_seller_{i:05d}" for i in range(n_sellers)],
        "seller_zip_code_prefix": rng.integers(1000, 99999, n_sellers),
        "seller_city": [CITIES[s] for s in seller_state],
        "seller_state": seller_state,
    })
    seller_difficulty = pd.Series(seller_state).map(STATES).to_numpy()
    seller_load = rng.beta(2.0, 5.0, size=n_sellers)  # persistent per-seller workload

    # --------------------------------------------------------------- products --
    n_products = 3000
    cat_idx = rng.integers(0, len(CATEGORIES), size=n_products)
    weight_kg = np.clip(CATEGORY_WEIGHT_KG[cat_idx] * rng.lognormal(0.0, 0.45, n_products), 0.05, 40.0)
    length = np.clip(rng.normal(30, 12, n_products), 7, 105)
    height = np.clip(rng.normal(18, 9, n_products), 2, 105)
    width = np.clip(rng.normal(24, 11, n_products), 6, 105)
    products = pd.DataFrame({
        "product_id": [f"synth_product_{i:05d}" for i in range(n_products)],
        "product_category_name": [CATEGORIES[c][0] for c in cat_idx],
        "product_name_lenght": rng.integers(20, 60, n_products),
        "product_description_lenght": rng.integers(200, 2000, n_products),
        "product_photos_qty": rng.integers(1, 8, n_products),
        "product_weight_g": np.round(weight_kg * 1000).astype(int),
        "product_length_cm": np.round(length).astype(int),
        "product_height_cm": np.round(height).astype(int),
        "product_width_cm": np.round(width).astype(int),
    })
    translation = pd.DataFrame(CATEGORIES, columns=[
        "product_category_name", "product_category_name_english",
    ])

    # --------------------------------------------------- order booking stream --
    span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    purchase_date = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, span_days, n_orders), unit="D")
    weekday = purchase_date.dayofweek.to_numpy()
    weekend = (weekday >= 5).astype(float)
    # Hourly booking weights: a plausible Brazilian e-commerce daily shape (low
    # overnight, lunch and evening peaks). Normalised explicitly because NumPy
    # requires probabilities to sum to 1 exactly (>=2.0 raises otherwise).
    hour_weights = np.array(
        [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 5, 5, 5, 5, 6, 6, 6, 6, 5, 4, 3, 2],
        dtype=float,
    )
    hour = rng.choice(np.arange(24), size=n_orders, p=hour_weights / hour_weights.sum())
    purchase_ts = (
        purchase_date
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(rng.integers(0, 60, n_orders), unit="m")
        + pd.to_timedelta(rng.integers(0, 60, n_orders), unit="s")
    )
    month = purchase_ts.month.to_numpy()  # DatetimeIndex: no .dt accessor
    peak = np.isin(month, [11, 12]).astype(float)  # Brazilian peak season

    # ------------------------------------------------------- line items (all) --
    items_per_order = np.where(rng.random(n_orders) < 0.88, 1, rng.integers(2, 5, n_orders))
    sellers_per_order = np.where(rng.random(n_orders) < 0.97, 1, 2)
    order_row = np.repeat(np.arange(n_orders), items_per_order)
    total_items = order_row.size
    item_id = np.arange(total_items) - np.repeat(np.cumsum(items_per_order) - items_per_order, items_per_order) + 1

    primary_seller = rng.integers(0, n_sellers, n_orders)
    secondary_seller = (primary_seller + rng.integers(1, n_sellers, n_orders)) % n_sellers
    half = np.ceil(items_per_order / 2)
    use_secondary = (item_id > np.repeat(half, items_per_order)) & (np.repeat(sellers_per_order, items_per_order) == 2)
    seller_idx = np.where(use_secondary, np.repeat(secondary_seller, items_per_order),
                          np.repeat(primary_seller, items_per_order))
    product_idx = rng.integers(0, n_products, total_items)

    items = pd.DataFrame({
        "order_row": order_row,
        "order_item_id": item_id,
        "product_idx": product_idx,
        "seller_idx": seller_idx,
        "weight_kg": weight_kg[product_idx],
        "volume_cm3": (length * height * width)[product_idx],
        "seller_difficulty": seller_difficulty[seller_idx],
        "seller_load": seller_load[seller_idx],
    })
    items["price"] = np.clip(np.round(rng.lognormal(3.6, 0.55, total_items), 2), 2.5, 2000.0)

    # ------------------------------------------------- customer / seller state --
    order_state = rng.choice(STATE_NAMES, size=n_orders, p=STATE_WEIGHTS)
    items["seller_state"] = pd.Series(sellers["seller_state"].to_numpy()[seller_idx]).to_numpy()
    items["cross_state"] = (items["seller_state"].to_numpy() != order_state[order_row]).astype(float)

    # ------------------------------------------------- order-level aggregation --
    agg = items.groupby("order_row").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_idx", "nunique"),
        n_products=("product_idx", "nunique"),
        price_sum=("price", "sum"),
        weight_kg_sum=("weight_kg", "sum"),
        seller_difficulty=("seller_difficulty", "mean"),
        seller_load=("seller_load", "mean"),
    ).reindex(range(n_orders))
    agg["n_items"] = agg["n_items"].fillna(1)
    agg = agg.fillna(0.0)

    # Freight depends on weight, out-of-state shipping and a little noise.
    items["freight_value"] = np.round(
        BASE_FREIGHT
        + FREIGHT_PER_KG * items["weight_kg"]
        + CROSS_STATE_PENALTY * items["cross_state"]
        + rng.normal(0, 3.0, total_items),
        2,
    ).clip(6.0, 420.0)
    freight_sum = items.groupby("order_row")["freight_value"].sum().reindex(range(n_orders)).fillna(0.0).to_numpy()
    cross_per_order = items.groupby("order_row")["cross_state"].mean().reindex(range(n_orders)).fillna(0.0).to_numpy()

    items["shipping_limit_date"] = (
        pd.Series(purchase_ts.to_numpy()[order_row]) + pd.to_timedelta(rng.integers(2, 8, total_items), unit="D")
    ).to_numpy()

    # ----------------------------------------- latent difficulty of the journey --
    cust_difficulty = pd.Series(order_state).map(STATES).to_numpy()
    order_difficulty = np.clip(
        0.55 * cust_difficulty
        + 0.35 * agg["seller_difficulty"].to_numpy()
        + 0.10 * agg["seller_load"].to_numpy()
        + 0.15 * cross_per_order,
        0.0, 1.2,
    )

    # ---- promised date: deliberately generous, more so in the peak season ----
    promised_days = np.clip(
        PROMISE_INTERCEPT_DAYS
        + PROMISE_DIFFICULTY_DAYS * order_difficulty
        + PROMISE_PEAK_DAYS * peak
        + rng.normal(0, PROMISE_NOISE_DAYS, n_orders),
        2.0, None,
    )
    promised_ts = (
        purchase_date
        + pd.to_timedelta(np.round(promised_days).astype(int), unit="D")
        + pd.to_timedelta(rng.integers(0, 24, n_orders), unit="h")
    )

    # ---- latent difficulty of the *actual* journey (correlated with the promise) --
    journey = (
        0.90 * order_difficulty
        + PEAK_JOURNEY_WEIGHT * peak
        + 0.06 * weekend
        + 0.10 * np.clip(agg["weight_kg_sum"].to_numpy() / 12.0, 0, 1.5)
        - 0.25 * np.clip(freight_sum / 120.0, 0, 1.5)
        + rng.normal(0, 0.16, n_orders)
    )
    transit_days = np.clip(
        TRANSIT_BASE_DAYS
        + TRANSIT_DIFFICULTY_DAYS * np.clip(journey, 0, 1.4)
        + rng.lognormal(0.0, 0.42, n_orders),
        1.0, None,
    )
    approval_ts = purchase_ts + pd.to_timedelta(np.round(rng.exponential(0.35, n_orders), 4), unit="D")
    carrier_ts = approval_ts + pd.to_timedelta(rng.lognormal(0.7, 0.5, n_orders), unit="D")
    delivered_ts = (
        carrier_ts
        + pd.to_timedelta(np.clip(transit_days - 2.0, 0.2, None), unit="D")
        + pd.to_timedelta(rng.integers(0, 24 * 3600, n_orders), unit="s")
    )

    # ------------------------------------------- status + realistic missingness --
    status_draw = rng.random(n_orders)
    status = np.where(
        status_draw < 0.00625, "canceled",
        np.where(status_draw < 0.00690, "unavailable", "delivered"),
    )
    lost_delivery = (status == "delivered") & (rng.random(n_orders) < 0.0021)
    lost_carrier = (status == "delivered") & (rng.random(n_orders) < 0.0045)
    lost_approval = rng.random(n_orders) < 0.0021

    delivered_series = pd.Series(delivered_ts)
    delivered_series[lost_delivery] = pd.NaT
    carried_series = pd.Series(carrier_ts)
    carried_series[lost_carrier | (status != "delivered")] = pd.NaT
    approval_series = pd.Series(approval_ts)
    approval_series[lost_approval] = pd.NaT

    order_ids = np.array([f"synth_order_{i:06d}" for i in range(n_orders)])
    # ``customer_id`` is the *order-level* key: in the Olist schema it is unique per
    # order and is the column ``orders.customer_id`` joins to. ``customer_unique_id``
    # is the person, so repeat buyers reappear there. Getting this backwards joins
    # nothing at all (every ``customer_state`` becomes missing and silently drops a
    # feature), so the two are built from explicitly different id spaces.
    customer_ids = np.array([f"synth_customer_{i:06d}" for i in range(n_orders)])
    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "order_status": status,
        "order_purchase_timestamp": purchase_ts,
        "order_approved_at": approval_series,
        "order_delivered_carrier_date": carried_series,
        "order_delivered_customer_date": delivered_series,
        "order_estimated_delivery_date": promised_ts,
    })

    # ------------------------------------------------------------- customers --
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_unique_id": [f"synth_uid_{v:06d}" for v in rng.integers(0, max(2, int(n_orders * 0.97)), n_orders)],
        "customer_zip_code_prefix": rng.integers(1000, 99999, n_orders),
        "customer_city": [CITIES[s] for s in order_state],
        "customer_state": order_state,
    })

    # ----------------------------------------------------------------- items --
    items["order_id"] = order_ids[items["order_row"].to_numpy()]
    items["product_id"] = products["product_id"].to_numpy()[items["product_idx"].to_numpy()]
    items["seller_id"] = sellers["seller_id"].to_numpy()[items["seller_idx"].to_numpy()]
    items = items[[
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ]].reset_index(drop=True)

    # -------------------------------------------------------------- payments --
    paid_orders = rng.choice(n_orders, size=int(round(n_orders * 0.98)), replace=False)
    n_pay_per_order = np.where(rng.random(paid_orders.size) < 0.97, 1, 2)
    pay_order_row = np.repeat(paid_orders, n_pay_per_order)
    pay_seq = np.arange(pay_order_row.size) - np.repeat(np.cumsum(n_pay_per_order) - n_pay_per_order, n_pay_per_order) + 1
    pay_type = rng.choice(["credit_card", "boleto", "voucher", "debit_card"],
                          size=pay_order_row.size, p=[0.735, 0.19, 0.06, 0.015])
    pay_inst = np.where(pay_type == "credit_card", rng.integers(1, 13, pay_order_row.size), 1)
    payments = pd.DataFrame({
        "order_id": order_ids[pay_order_row],
        "payment_sequential": pay_seq,
        "payment_type": pay_type,
        "payment_installments": pay_inst,
        "payment_value": np.round(rng.lognormal(4.0, 0.6, pay_order_row.size), 2),
    })

    # ---- product-level missingness (Olist has ~1.8% products with no category) ----
    products.loc[rng.random(n_products) < 0.018, "product_category_name"] = None
    missing_measure = rng.random(n_products) < 0.0025
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        products.loc[missing_measure, col] = np.nan

    tables = {
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": items,
        "olist_products_dataset": products,
        "olist_customers_dataset": customers,
        "olist_sellers_dataset": sellers,
        "olist_order_payments_dataset": payments,
        "product_category_name_translation": translation,
    }
    logger.info("Generated synthetic tables: %s", {k: len(v) for k, v in tables.items()})
    return tables


def generator_fingerprint() -> str:
    """SHA-256 of this file, so stale synthetic CSVs can be detected automatically.

    The synthetic CSVs are cached on disk (generating them costs a few seconds).
    Without this fingerprint, editing a knob above would leave the old CSVs in place
    and the pipeline would silently keep reporting results from a generator that no
    longer exists - a reproducibility trap this function exists to close.
    """
    return file_sha256(Path(__file__)) or "unknown"


def read_provenance(out_dir: str | Path) -> Dict[str, Any]:
    """Read the provenance marker of a synthetic dataset, ``{}`` if absent."""
    path = Path(out_dir) / "PROVENANCE.json"
    if not path.exists():
        return {}
    try:
        return load_json(path) or {}
    except (json.JSONDecodeError, OSError):  # unreadable marker -> treat as absent
        return {}


def write_synthetic_dataset(out_dir: str | Path, n_orders: int, seed: int) -> Path:
    """Generate and persist the synthetic tables as Olist-named CSV files."""
    out = ensure_dir(out_dir)
    tables = generate_tables(n_orders=n_orders, seed=seed)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)

    # A provenance marker so the loader (and the app) can never mistake these
    # CSV files for the real dataset.
    save_json(
        {
            "kind": "synthetic",
            "generator": "src/synthetic.py",
            "generator_fingerprint": generator_fingerprint(),
            "seed": seed,
            "n_orders_requested": n_orders,
            "warning": (
                "SIMULATED DATA. Metrics measured on this dataset are a software smoke "
                "test only and must never be reported as real-world predictive skill."
            ),
        },
        out / "PROVENANCE.json",
    )
    logger.info("Wrote synthetic dataset to %s", out)
    return out


def main(n_orders: int | None = None, seed: int | None = None) -> Path:
    """CLI entry point: ``python -m src.synthetic``."""
    cfg = load_config()
    n = n_orders if n_orders is not None else int(cfg["data_source"]["synthetic_n_orders"])
    s = seed if seed is not None else int(cfg["data_source"]["synthetic_seed"])
    return write_synthetic_dataset(resolve(cfg, "paths", "synthetic_raw_dir"), n, s)


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic demonstration dataset.")
    parser.add_argument("--n-orders", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    _args = parser.parse_args()
    main(_args.n_orders, _args.seed)
