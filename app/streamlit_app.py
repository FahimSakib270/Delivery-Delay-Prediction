"""Research demo: score a single order at booking time.

This app is a *demo*, not a product. It reads the artefacts written by
``python -m src.prepare`` / ``src.train`` / ``src.evaluate`` and deliberately refuses to
hide where the numbers came from: if the dataset behind the model is the synthetic
fallback, a banner says so at the top of every page.

Design notes worth knowing before changing anything:

* the decision threshold comes **from the saved model bundle**, never from the UI, so
  what the app shows is exactly what was evaluated;
* the app never re-trains and never touches the test set to pick anything;
* uncalibrated scores are not presented as probabilities or operational advice.

Run with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

# Streamlit executes this file with the *script* directory on sys.path, so the
# repository root has to be added explicitly before importing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data_prep import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from src.evaluation import predict_positive_proba  # noqa: E402

#: One-line explanations shown under each input. Written for a non-specialist reader:
#: these are the definitions the model was trained on, not marketing copy.
FEATURE_HELP: Dict[str, str] = {
    "purchase_month": "Month the order was placed (1-12). Seasonality proxy.",
    "purchase_weekday": "0 = Monday ... 6 = Sunday.",
    "purchase_hour": "Hour of the booking timestamp (0-23).",
    "lead_time_days": "Promised delivery date minus booking date, in days. The single "
                      "most informative booking-time feature in public Olist analyses.",
    "n_items": "Number of item rows on the order.",
    "n_products": "Distinct products on the order.",
    "n_sellers": "Distinct sellers that must ship (more sellers = more handovers).",
    "price_sum": "Sum of item prices, BRL.",
    "freight_sum": "Sum of freight charged, BRL.",
    "order_value": "price_sum + freight_sum, BRL.",
    "freight_ratio": "freight_sum / order_value: how much of the bill is shipping.",
    "total_weight_g": "Total packed weight of the order, grams.",
    "mean_weight_g": "Mean packed weight per item, grams.",
    "total_volume_cm3": "Total packed volume, cubic centimetres.",
    "same_state": "1 when the customer and the (primary) seller are in the same state.",
    "customer_state": "Destination state of the customer.",
    "seller_state": "State of the primary seller.",
    "product_category": "English category of the primary item.",
}

# Sentinel shown in the categorical dropdowns for "a level this dataset never showed
# me". The encoder maps it to its infrequent bucket, exactly like a rare level.
OTHER_LABEL = "(other / rare)"

# Fallback ranges, used only when data/processed/orders_prepared.csv is absent.
# They are round numbers around the scale of the prepared dataset, not measured
# statistics, so the helper that derives real ranges is always preferred.
FALLBACK_NUMERIC: Dict[str, Tuple[float, float, float]] = {
    "purchase_month": (1, 12, 6),
    "purchase_weekday": (0, 6, 2),
    "purchase_hour": (0, 23, 15),
    "lead_time_days": (2, 60, 15),
    "n_items": (1, 5, 1),
    "n_products": (1, 5, 1),
    "n_sellers": (1, 3, 1),
    "price_sum": (10.0, 500.0, 90.0),
    "freight_sum": (6.0, 200.0, 22.0),
    "order_value": (16.0, 700.0, 112.0),
    "freight_ratio": (0.05, 0.6, 0.2),
    "total_weight_g": (100.0, 20000.0, 1500.0),
    "mean_weight_g": (100.0, 20000.0, 1500.0),
    "total_volume_cm3": (500.0, 120000.0, 20000.0),
    "same_state": (0, 1, 1),
}
FALLBACK_CATEGORICAL: Dict[str, List[str]] = {
    "customer_state": ["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "PE", "CE", "DF"],
    "seller_state": ["SP", "RJ", "MG", "PR", "SC", "RS", "GO", "BA", "CE", "DF"],
    "product_category": [
        "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
        "computers_accessories", "housewares", "watches_gifts", "telephony",
        "garden_tools", "auto", "toys", "cool_stuff",
    ],
}


# ---------------------------------------------------------------------------
# Pure helpers (imported and unit-tested without a Streamlit runtime)
# ---------------------------------------------------------------------------
def assemble_features(values: Mapping[str, Any]) -> pd.DataFrame:
    """Turn a mapping of UI values into the single-row frame the pipeline expects.

    Columns are emitted in :data:`MODEL_FEATURES` order and missing keys raise a
    ``KeyError`` rather than silently becoming NaN, because a silently-imputed
    feature would make the demo's probability impossible to explain.
    """
    missing = [name for name in MODEL_FEATURES if name not in values]
    if missing:
        raise KeyError(f"missing model input(s): {missing}")
    row = {name: [values[name]] for name in MODEL_FEATURES}
    return pd.DataFrame(row, columns=list(MODEL_FEATURES))


def break_even_probability(cost_false_negative: float, cost_false_positive: float) -> float:
    """Probability at which flagging and ignoring an order cost the same.

    With a linear cost model the expected cost of ignoring an order is
    ``p * cost_false_negative`` and the expected cost of flagging it is
    ``(1 - p) * cost_false_positive``.  Setting them equal gives
    ``p* = cost_false_positive / (cost_false_positive + cost_false_negative)``,
    which depends only on the cost assumption - never on the model.  The model's job
    is therefore to be *calibrated* around ``p*``, not to be good on average.
    """
    total = float(cost_false_positive) + float(cost_false_negative)
    if total <= 0:
        return float("nan")
    return float(cost_false_positive) / total


def flagging_value(
    probability: float,
    cost_false_negative: float,
    cost_false_positive: float,
) -> Dict[str, Any]:
    """Expected cost per order of flagging vs ignoring, and the resulting advice."""
    p = float(probability)
    cost_if_ignore = p * float(cost_false_negative)
    cost_if_flag = (1.0 - p) * float(cost_false_positive)
    p_star = break_even_probability(cost_false_negative, cost_false_positive)
    return {
        "probability": p,
        "break_even_probability": p_star,
        "expected_cost_if_ignore": cost_if_ignore,
        "expected_cost_if_flag": cost_if_flag,
        "saving_if_flag": cost_if_ignore - cost_if_flag,
        "recommendation": "flag for proactive contact" if cost_if_flag < cost_if_ignore
        else "leave alone",
    }


# ---------------------------------------------------------------------------
# Artefact access
# ---------------------------------------------------------------------------
def project_paths() -> Dict[str, Path]:
    """Resolved output directories, read from config/config.yaml (never hard-coded)."""
    cfg = load_config(os.environ.get("PROFAST_CONFIG"))
    return {
        "processed": resolve(cfg, "paths", "processed_dir"),
        "models": resolve(cfg, "paths", "models_dir"),
        "reports": resolve(cfg, "paths", "reports_dir"),
        "figures": resolve(cfg, "paths", "figures_dir"),
    }


def read_json(path: Path) -> Dict[str, Any]:
    """Load a JSON artefact, returning ``{}`` when it is missing or unreadable."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str) -> Optional[Dict[str, Any]]:
    """Load the fitted pipeline bundle (cached: unpickling is not free)."""
    path = Path(path_str)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner="Loading the prepared dataset ...")
def load_table(path_str: str, usecols: Optional[Tuple[str, ...]] = None) -> Optional[pd.DataFrame]:
    """Read a CSV artefact (optionally a column subset), ``None`` if absent."""
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=list(usecols) if usecols else None)


# ---------------------------------------------------------------------------
# Input widgets: ranges come from the data, not from hard-coded guesses
# ---------------------------------------------------------------------------
INTEGER_INPUTS = {
    "purchase_month", "purchase_weekday", "purchase_hour",
    "n_items", "n_products", "n_sellers", "same_state",
}


def numeric_widget_specs(df: Optional[pd.DataFrame]) -> Dict[str, Tuple[float, float, float]]:
    """``{feature: (min, max, default)}`` derived from the prepared dataset.

    The 1st/99th percentiles are used as bounds so that the sliders can reach the
    realistic extremes without being dragged out by a handful of outliers. Falls back to
    :data:`FALLBACK_NUMERIC` when the dataset (or a column) is unavailable.
    """
    specs: Dict[str, Tuple[float, float, float]] = {}
    for name in NUMERIC_FEATURES:
        lo, hi, default = FALLBACK_NUMERIC.get(name, (0.0, 1.0, 0.0))
        if df is not None and name in df.columns:
            column = pd.to_numeric(df[name], errors="coerce").dropna()
            if len(column) > 20:
                lo = float(np.floor(column.quantile(0.01)))
                hi = float(np.ceil(column.quantile(0.99)))
                default = float(np.round(column.median(), 2))
                if name in INTEGER_INPUTS:
                    default = float(np.round(column.median()))
        if hi <= lo:
            hi = lo + 1.0
        specs[name] = (lo, hi, min(max(default, lo), hi))
    return specs


def categorical_widget_options(df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """``{feature: [levels]}`` with the long tail collapsed into one sentinel level.

    The sentinel is an unseen category as far as the encoder is concerned, so the
    pipeline routes it to its ``infrequent`` bucket - the same treatment a rare level
    would have received. Keeping it visible (rather than silently picking a common
    state) is what stops the demo from inventing information the user did not give.
    """
    options: Dict[str, List[str]] = {}
    for name in CATEGORICAL_FEATURES:
        levels = list(FALLBACK_CATEGORICAL.get(name, [OTHER_LABEL]))
        if df is not None and name in df.columns:
            counts = df[name].astype("string").fillna(OTHER_LABEL).value_counts()
            levels = [str(value) for value in counts.head(40).index]
            if len(counts) > 40:
                levels.append(OTHER_LABEL)
        options[name] = levels
    return options


def risk_band(probability: float, threshold: float) -> str:
    """Plain-language band around the frozen decision threshold."""
    if probability >= max(threshold, 0.5):
        return "high"
    if probability >= threshold:
        return "elevated"
    if probability >= threshold / 2.0:
        return "low"
    return "very low"


def format_currency(value: float) -> str:
    """BRL-style formatting for the (assumed) cost arithmetic."""
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# The app itself
# ---------------------------------------------------------------------------
def render_provenance_banner(metadata: Mapping[str, Any]) -> None:
    """Say where the numbers come from before showing any number."""
    provenance = dict(metadata.get("provenance", {})) if metadata else {}
    kind = provenance.get("data_source_kind", "unknown")
    if provenance.get("synthetic", True):
        st.error(
            "**SYNTHETIC DEMONSTRATION DATA.** The model behind this page was trained on "
            "data *generated* by `src/synthetic.py`, not on real ProFast or Olist "
            "deliveries. It exists so the whole pipeline can be run and tested end to end. "
            "Every number here is a software smoke test, **not** evidence about real "
            "delivery performance.",
            icon="⚠️",
        )
    else:
        st.success(f"Data source: real Olist CSVs (`{kind}`).", icon="✅")


def sidebar_summary(bundle, metadata, manifest) -> None:
    """Show provenance, never convert uncalibrated scores into business advice."""
    with st.sidebar:
        st.header("Model")
        st.write(f"**Algorithm:** `{bundle.get('model_name', '?')}`")
        st.write(f"**Frozen experimental threshold:** `{float(bundle.get('threshold', 0.5)):.4f}`")
        st.write(f"**Training rows:** {int(bundle.get('training_rows', 0)):,}")
        st.write(f"**Trained at (UTC):** {str(bundle.get('trained_at', '?'))[:19]}")
        st.write(f"**Label:** {bundle.get('label_definition', '?')}")
        st.caption(f"processed sha256 `{str(bundle.get('dataset_sha256', '?'))[:16]}...`")
        st.caption(f"run manifest commit `{manifest.get('git_commit') or 'unknown'}`")
        st.warning("Uncalibrated risk scores. This prototype does not recommend operational actions.")


def order_input_form(
    specs: Mapping[str, Tuple[float, float, float]],
    options: Mapping[str, List[str]],
    prefill: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Render the booking-time inputs. Returns the values when the form is submitted."""
    def default_for(name: str, fallback: Any) -> Any:
        if prefill is None or name not in prefill:
            return fallback
        value = prefill[name]
        if isinstance(value, float) and not np.isfinite(value):
            return fallback
        return value

    with st.form("order_form", border=True):
        st.subheader("Order at booking time")
        st.caption(
            "Only information that exists at the moment of booking is accepted. Delivery "
            "dates, carrier handover dates and payment rows are deliberately absent: at "
            "booking time they are unknown, and using them would make the model useless "
            "for the decision it is meant to support."
        )
        values: Dict[str, Any] = {}
        numeric_columns = st.columns(3)
        for index, name in enumerate(NUMERIC_FEATURES):
            lo, hi, default = specs[name]
            with numeric_columns[index % 3]:
                if name in INTEGER_INPUTS:
                    lo_i = int(lo)
                    hi_i = max(int(hi), lo_i + 1)
                    values[name] = int(st.slider(
                        name.replace("_", " "), lo_i, hi_i,
                        int(min(max(default_for(name, default), lo_i), hi_i)),
                        step=1, help=FEATURE_HELP.get(name),
                    ))
                else:
                    step = max(round((hi - lo) / 100.0, 4), 0.01)
                    values[name] = float(st.slider(
                        name.replace("_", " "), float(lo), float(hi),
                        float(min(max(default_for(name, default), lo), hi)),
                        step=step, help=FEATURE_HELP.get(name),
                    ))
        categorical_columns = st.columns(3)
        for index, name in enumerate(CATEGORICAL_FEATURES):
            levels = options[name]
            with categorical_columns[index % 3]:
                fallback = levels[0]
                chosen = str(default_for(name, fallback))
                index_of = levels.index(chosen) if chosen in levels else min(1, len(levels) - 1)
                values[name] = st.selectbox(
                    name.replace("_", " "), levels, index=index_of,
                    help=FEATURE_HELP.get(name),
                )
        submitted = st.form_submit_button("Score this order", type="primary")
    return values if submitted else None



def render_score(bundle: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Display a ranking score and the frozen experimental flag."""
    frame = assemble_features(values)
    try:
        score = float(predict_positive_proba(bundle["pipeline"], frame[MODEL_FEATURES])[0])
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Unable to score these inputs: {exc}")
        return
    threshold = float(bundle["threshold"])
    st.subheader("Prediction")
    left, right = st.columns(2)
    left.metric("Uncalibrated delay-risk score", f"{score:.3f}")
    right.metric("Frozen experimental threshold", f"{threshold:.3f}")
    st.progress(min(max(score, 0.0), 1.0))
    st.info("Experimental risk flag: " + ("above threshold" if score >= threshold else "below threshold"))
    st.warning(
        "This score is not a calibrated probability. A score of 0.8 does not mean an 80% "
        "chance of delay. The flag is an experimental classification, not advice to contact "
        "a customer or change a delivery."
    )
    st.caption(
        "The threshold was selected on validation data. Reported bootstrap intervals describe "
        "aggregate test metrics, not uncertainty for this individual order. Changing an input "
        "does not establish a causal effect."
    )


def render_performance_tab(summary: Mapping[str, Any], reports_dir: Path) -> None:
    """Everything the frozen model achieved, with its uncertainty, in one place."""
    if not summary:
        st.info("Run `python -m src.evaluate` to populate this tab.", icon="ℹ️")
        return
    metrics = dict(summary.get("test_metrics", {}))
    intervals = dict(summary.get("bootstrap_95", {}))
    st.subheader("Retrospective test period - excluded from automated selection")
    rows = []
    for name, value in metrics.items():
        if name in {"threshold", "n", "n_positive", "n_negative"}:
            continue
        interval = intervals.get(name, {})
        # ``bootstrap_ci`` names its bounds ``lower``/``upper``; the legacy aliases are
        # accepted so the tab keeps working against an older summary file. Getting this
        # wrong does not crash - it silently empties the interval columns, which is worse.
        rows.append({
            "metric": name,
            "value": round(float(value), 4),
            "ci_low": interval.get("lower", interval.get("ci_low")),
            "ci_high": interval.get("upper", interval.get("ci_high")),
            "n_effective": interval.get("n_effective"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Model `{summary.get('model_name', '?')}`, frozen threshold "
        f"{float(summary.get('threshold', float('nan'))):.4f}, calibration ECE "
        f"{float(summary.get('calibration_ece', float('nan'))):.4f}. Bootstrap intervals "
        "resample test rows; a draw containing a single class is skipped and counted in "
        "`n_effective`."
    )
    st.caption(
        "ROC-AUC describes ranking; precision-recall metrics reveal rare-delay detection. "
        "Average precision is the selection metric; it is not trapezoidal PR-AUC."
    )

    cost = dict(summary.get("cost", {}))
    if cost:
        frozen = dict(cost.get("at_frozen_threshold", {}))
        half = dict(cost.get("at_threshold_0_5", {}))
        assumptions = dict(cost.get("assumptions", {}))
        st.markdown(
            "**Hypothetical error penalty per 1 000 orders** - at the frozen threshold "
            f"{float(frozen.get('cost_per_1000_orders', float('nan'))):,.2f}; at 0.5 "
            f"{float(half.get('cost_per_1000_orders', float('nan'))):,.2f}; flagging nothing "
            f"{float(frozen.get('cost_if_never_flagging_per_1000', float('nan'))):,.2f}. "
            f"Assumes {assumptions.get('cost_false_negative')} per missed delay and "
            f"{assumptions.get('cost_false_positive')} per false alarm."
        )

    subgroups = load_table(str(reports_dir / "test_subgroups.csv"))
    if subgroups is not None and not subgroups.empty:
        st.subheader("Subgroups (recall at the frozen threshold)")
        st.dataframe(
            subgroups.loc[~subgroups["below_min_n"]].sort_values("recall").head(15),
            hide_index=True, width="stretch",
        )
        st.caption(
            f"{int(subgroups['below_min_n'].sum())} subgroups are hidden because they hold "
            f"fewer than {int(summary.get('subgroups_min_n', 0))} test orders; a recall "
            "estimated from a handful of rows is noise, not a finding."
        )

    figures = dict(summary.get("figures", {}))
    if figures:
        st.subheader("Figures")
        for name, path in figures.items():
            figure_path = Path(str(path))
            if not figure_path.is_absolute():
                figure_path = reports_dir / figure_path
            if figure_path.exists():
                st.image(str(figure_path), caption=name.replace("_", " "), width="stretch")


def render_provenance_tab(
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    reports_dir: Path,
) -> None:
    """Where the data came from, what was forbidden, and how to reproduce the run."""
    provenance = dict(metadata.get("provenance", {}))
    statistics = dict(metadata.get("statistics", {}))
    st.subheader("Data")
    st.write(f"**Source kind:** `{provenance.get('data_source_kind', 'unknown')}`")
    st.write(f"**Synthetic:** `{provenance.get('synthetic', True)}`")
    st.write(f"**Raw directory:** `{provenance.get('raw_dir', '?')}`")
    st.write(f"**Rows:** {int(statistics.get('final_orders', 0)):,}")
    st.write(
        f"**Late rate (primary label):** "
        f"{float(statistics.get('late_rate', float('nan'))):.4f}"
    )
    st.caption(
        "`late` means the delivery timestamp is strictly after the promised date, at "
        "calendar-date granularity for the primary label; a timestamp-based variant is "
        "reported as a sensitivity check."
    )

    st.subheader("Checksums")
    st.json({
        "dataset_fingerprint": provenance.get("dataset_fingerprint"),
        "processed_sha256": provenance.get("processed_sha256"),
        "raw_file_sha256": provenance.get("raw_file_sha256"),
    }, expanded=False)

    audit = load_table(str(reports_dir / "feature_audit.csv"))
    if audit is not None and not audit.empty:
        st.subheader("Feature audit")
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption(
            "Every candidate feature, whether it exists at booking time, and why it was used "
            "or rejected. Delivery timestamps never enter the model: they define the label "
            "and the maturity filter only."
        )

    if manifest:
        st.subheader("Run manifest")
        st.json({
            "git_commit": manifest.get("git_commit"),
            "environment": manifest.get("environment"),
            "split": manifest.get("split"),
            "random_seed": manifest.get("random_seed"),
            "config": manifest.get("config"),
        }, expanded=False)

    summary_md = reports_dir / "results_summary.md"
    if summary_md.exists():
        st.subheader("Machine-written results summary")
        st.markdown(summary_md.read_text(encoding="utf-8"))
        st.caption(
            f"Generated from this run's own artefacts at `{summary_md}`, so it cannot drift "
            "away from the numbers above."
        )


def main() -> None:
    """Entry point: assemble the page from whatever artefacts exist right now."""
    st.set_page_config(page_title="ProFast delivery-delay risk", page_icon="📦", layout="wide")
    paths = project_paths()
    metadata = read_json(paths["processed"] / "dataset_metadata.json")
    manifest = read_json(paths["reports"] / "run_manifest.json")
    summary = read_json(paths["reports"] / "evaluation_summary.json")

    st.title("📦 ProFast - will this order be delivered late?")
    st.caption(
        "A decision-support prototype that scores an order **at booking time**, before any "
        "delivery timestamps exist. Built from the public Brazilian Olist dataset (or from "
        "the synthetic fallback when that dataset is unavailable)."
    )
    render_provenance_banner(metadata)

    bundle = load_bundle(str(paths["models"] / "best_model.joblib"))
    if bundle is None:
        st.warning(
            "No trained model found. Run the pipeline first:\n\n"
            "```\npython -m src.prepare\npython -m src.train\npython -m src.evaluate\n```",
            icon="🛠️",
        )
        st.stop()

    sidebar_summary(bundle, metadata, manifest)

    prepared = load_table(str(paths["models"] / "training_inputs.csv"), tuple(MODEL_FEATURES))
    if prepared is None:
        st.warning(
            "`models/training_inputs.csv` is missing, so input ranges fall back to "
            "generic values. Run `python -m src.prepare` for data-driven ranges.",
            icon="ℹ️",
        )
    specs = numeric_widget_specs(prepared)
    options = categorical_widget_options(prepared)

    test_rows = load_table(
        str(paths["reports"] / "test_predictions.csv"),
        tuple(list(MODEL_FEATURES) + ["probability"]),
    )
    if "prefill" not in st.session_state:
        st.session_state["prefill"] = None

    button_columns = st.columns([1, 1, 4])
    with button_columns[0]:
        if st.button("Load a test-period order", disabled=test_rows is None or test_rows.empty):
            st.session_state["prefill"] = (
                test_rows.sample(1).iloc[0].to_dict() if test_rows is not None else None
            )
            st.rerun()
    with button_columns[1]:
        if st.button("Clear", disabled=st.session_state["prefill"] is None):
            st.session_state["prefill"] = None
            st.rerun()

    tab_score, tab_performance, tab_provenance = st.tabs(
        ["Score an order", "Model performance", "Data & provenance"]
    )
    with tab_score:
        submitted = order_input_form(specs, options, st.session_state["prefill"])
        if submitted is not None:
            render_score(bundle, submitted)
        else:
            st.info(
                "Set the booking-time inputs and press **Score this order**. The sliders "
                "start at the median of the training data; **Load a test-period order** "
                "fills them from a real row of the held-out period.",
                icon="👉",
            )
        st.caption(
            "This demo uses uncalibrated risk scores and provides no operational recommendation."
        )
    with tab_performance:
        render_performance_tab(summary, paths["reports"])
    with tab_provenance:
        render_provenance_tab(metadata, manifest, paths["reports"])


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    main()
