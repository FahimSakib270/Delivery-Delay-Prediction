"""Metric computation, threshold selection, bootstrap intervals and error analysis.

Two evaluation rules are enforced throughout:

* The **decision threshold is chosen on the validation split and then frozen**; the
  test split is scored once with that frozen threshold. Re-selecting the threshold
  on the test set would turn a reported metric into an optimistic guess.
* Because delays are the minority class, the primary ranking metric is **average
  precision (PR-AUC)**, which only cares about the positive class. **ROC-AUC is
  reported as well but explicitly flagged as optimistic** for very imbalanced data,
  because it is dominated by the many easy true negatives. Accuracy is reported only
  to show how useless it is here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_prep import TARGET_COLUMN
from .utils import get_logger

logger = get_logger("evaluation")

#: Probabilities are clipped before log-loss/Brier so a single confident mistake
#: cannot produce ``inf`` and destroy an otherwise comparable table.
MIN_PROB, MAX_PROB = 1e-6, 1.0 - 1e-6

#: Ranking metrics whose resampled value is undefined when a bootstrap draw happens
#: to contain a single class; those draws are skipped and counted.
RANKING_METRICS = ("roc_auc", "average_precision")
DEFAULT_METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1", "brier")


def predict_positive_proba(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """Probability of the delayed class, robust to estimators that drop a class.

    ``DummyClassifier(strategy="most_frequent")`` only predicts one class and
    therefore returns a single column; this helper returns zeros for such models
    instead of raising an index error.
    """
    proba = np.asarray(pipeline.predict_proba(X), dtype=float)
    classes = list(getattr(pipeline.named_steps["model"], "classes_", [0, 1]))
    if proba.ndim == 1:  # pragma: no cover - very old sklearn behaves this way
        proba = np.column_stack([1.0 - proba, proba])
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return proba[:, classes.index(1)]


def _safe(metric: Callable[..., float], *args: Any, **kwargs: Any) -> float:
    """Return ``nan`` instead of raising when a metric is undefined."""
    try:
        value = float(metric(*args, **kwargs))
    except (ValueError, ZeroDivisionError):  # single-class fold, no positives, ...
        return float("nan")
    return value


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    """Full metric panel for one split at one fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), MIN_PROB, MAX_PROB)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")

    metrics: Dict[str, float] = {
        "n": int(len(y_true)),
        "n_late": int(y_true.sum()),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
        "average_precision": _safe(average_precision_score, y_true, y_prob),
        "brier": _safe(brier_score_loss, y_true, y_prob),
        "log_loss": _safe(log_loss, y_true, y_prob, labels=[0, 1]),
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "predicted_positive_rate": float(y_pred.mean()),
    }
    metrics["precision_lift_over_prevalence"] = (
        float(metrics["precision"] / prevalence)
        if prevalence and np.isfinite(metrics["precision"]) else float("nan")
    )
    # A model that never predicts "late" has recall 0 and precision 0 by convention,
    # which would make the lift above look like a divide-by-zero artefact; the
    # explicit counts make the situation unambiguous in the report.
    return metrics


def threshold_grid(low: float = 0.01, high: float = 0.99, n: int = 197) -> np.ndarray:
    """Candidate decision thresholds: fine enough that the choice is not grid-limited."""
    return np.unique(np.round(np.linspace(low, high, n), 4))


def threshold_table(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    grid: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Metrics for every candidate threshold (used for selection and plotting)."""
    rows = [
        compute_metrics(y_true, y_prob, threshold=float(t))
        for t in (threshold_grid() if grid is None else grid)
    ]
    return pd.DataFrame(rows)


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    objective: str = "max_f1",
    min_precision_floor: float = 0.30,
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """Pick the operating threshold on the *validation* split.

    ``max_f1``
        Threshold maximising F1 for the delayed class.
    ``recall_at_prec``
        Highest recall whose precision is at least
        ``max(prevalence, min_precision_floor)``, i.e. "never do worse than the base
        rate, and ideally beat 30% precision". If no threshold qualifies, the
        function falls back to ``max_f1`` and records that in the returned info
        block instead of silently returning something arbitrary.
    """
    table = threshold_table(y_true, y_prob)
    prevalence = float(np.mean(np.asarray(y_true).astype(int)))
    info: Dict[str, Any] = {"objective": objective, "prevalence": prevalence, "fallback": False}

    if objective == "max_f1":
        best = table.sort_values(["f1", "precision", "threshold"],
                                 ascending=[False, False, True]).iloc[0]
    elif objective == "recall_at_prec":
        floor = max(prevalence, float(min_precision_floor))
        info["precision_floor"] = floor
        eligible = table.loc[table["precision"] >= floor]
        if eligible.empty:
            info["fallback"] = True
            info["note"] = (f"no threshold reached precision >= {floor:.3f}; "
                            "fell back to the max-F1 threshold")
            best = table.sort_values(["f1", "precision", "threshold"],
                                     ascending=[False, False, True]).iloc[0]
        else:
            best = eligible.sort_values(["recall", "precision", "threshold"],
                                        ascending=[False, False, True]).iloc[0]
    else:
        raise ValueError(f"Unknown threshold objective: {objective!r}")

    threshold = float(best["threshold"])
    info["selected"] = {k: v for k, v in best.items()}
    logger.info("threshold selected on validation: %.4f (%s -> f1=%.4f, precision=%.4f, recall=%.4f)",
                threshold, objective, best["f1"], best["precision"], best["recall"])
    return threshold, table, info


def pr_curve_points(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, np.ndarray]:
    """Precision/recall curve plus the class prevalence used as a reference line."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds,
            "prevalence": float(np.mean(np.asarray(y_true).astype(int)))}


def calibration_table(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability table: predicted probability vs observed delay frequency."""
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                          "p": np.asarray(y_prob, dtype=float)})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, n_bins + 1),
                          include_lowest=True, labels=False)
    table = (frame.groupby("bin", dropna=True)
             .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
                  lower=("p", "min"), upper=("p", "max"))
             .reset_index())
    table["abs_gap"] = (table["mean_predicted"] - table["observed_rate"]).abs()
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted mean gap between predicted probability and observed frequency.

    This is the standard ECE of a reliability table: each probability bin contributes
    its absolute gap, weighted by how many orders fell into it. A single number cannot
    replace the reliability plot (it is dominated by whichever bins hold the most rows),
    so the plot is always shown next to it -- the number is only there to justify the
    sentence "the probabilities are / are not usable as probabilities".
    """
    if table is None or len(table) == 0:
        return float("nan")
    total = float(table["n"].sum())
    if total <= 0 or "abs_gap" not in table.columns:
        return float("nan")
    return float((table["abs_gap"] * table["n"]).sum() / total)


def bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap confidence intervals for the test-set metrics.

    Resampling rows (not re-training) answers the question the report needs:
    *how much of the difference between two models could just be sampling noise?*
    Draws that end up single-class cannot produce a ranking metric; they are
    skipped and the number of usable draws is reported so an interval is never
    presented without its sample size.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}

    draws: Dict[str, List[float]] = {m: [] for m in metrics}
    used = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():  # only one class present -> AUC/AP undefined
            if any(m in RANKING_METRICS for m in metrics):
                continue
        panel = compute_metrics(yt, yp, threshold=threshold)
        used += 1
        for name in metrics:
            value = panel.get(name, float("nan"))
            if np.isfinite(value):
                draws[name].append(float(value))

    out: Dict[str, Dict[str, float]] = {}
    for name in metrics:
        values = np.asarray(draws[name], dtype=float)
        if values.size < 2:
            out[name] = {"point": float(compute_metrics(y_true, y_prob, threshold)[name]),
                         "lower": float("nan"), "upper": float("nan"),
                         "n_effective": int(values.size)}
            continue
        out[name] = {
            "point": float(compute_metrics(y_true, y_prob, threshold)[name]),
            "lower": float(np.quantile(values, alpha / 2)),
            "upper": float(np.quantile(values, 1 - alpha / 2)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "n_effective": int(values.size),
        }
    logger.info("bootstrap: %d/%d draws usable for ranking metrics", used, int(n_boot))
    return out


def expected_cost(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    cost_false_negative: float = 25.0,
    cost_false_positive: float = 4.0,
) -> Dict[str, float]:
    """Expected intervention cost per 1 000 orders at a fixed decision rule.

    Cost assumptions are *placeholders for the business to replace* (see
    ``business_case`` in ``config/config.yaml``); the arithmetic is what matters:
    a missed delay costs more than a wasted retention voucher, so the optimal
    threshold is not the one that maximises accuracy.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = float(fn * cost_false_negative + fp * cost_false_positive)
    n = len(y_true)
    never = float(int(y_true.sum()) * cost_false_negative)
    return {
        "false_negatives": fn, "false_positives": fp, "true_positives": tp,
        "cost_total": total,
        "cost_per_1000_orders": total / n * 1000.0 if n else float("nan"),
        "cost_if_never_flagging_per_1000": never / n * 1000.0 if n else float("nan"),
        # Positive value = flagging is cheaper than doing nothing at all.
        "saving_vs_never_flagging_per_1000": (never - total) / n * 1000.0 if n else float("nan"),
    }


def cost_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    cost_false_negative: float = 25.0,
    cost_false_positive: float = 4.0,
    grid: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Cost per 1 000 orders for every candidate threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for t in (threshold_grid() if grid is None else grid):
        panel = expected_cost(y_true, (y_prob >= t).astype(int), cost_false_negative, cost_false_positive)
        panel["threshold"] = float(t)
        rows.append(panel)
    return pd.DataFrame(rows)


def cost_optimal_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    cost_false_negative: float = 25.0,
    cost_false_positive: float = 4.0,
) -> Tuple[float, pd.DataFrame]:
    """Threshold minimising expected cost **on the validation split**."""
    curve = cost_curve(y_true, y_prob, cost_false_negative, cost_false_positive)
    best = curve.sort_values(["cost_per_1000_orders", "threshold"]).iloc[0]
    return float(best["threshold"]), curve


# --------------------------------------------------------------------------
# Error analysis: subgroups and permutation importance
# --------------------------------------------------------------------------
#: Buckets used for the subgroup check. Built from booking-time fields only, so
#: slicing the errors cannot leak the outcome.
LEAD_TIME_BUCKETS = [(-1, 7, "<=7 days"), (7, 15, "8-15 days"), (15, 25, "16-25 days"),
                     (25, 10_000, ">25 days")]

#: Columns the subgroup report iterates over, and how many levels of each to keep.
SUBGROUP_COLUMNS = ("customer_state", "product_category", "lead_time_bucket", "items_bucket",
                    "route_bucket", "month_bucket")
SUBGROUP_TOP_N = 12


def add_analysis_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived grouping columns used by the subgroup error analysis."""
    out = df.copy()
    if "lead_time_days" in out.columns:
        lead = pd.to_numeric(out["lead_time_days"], errors="coerce")
        bucket = pd.Series(pd.NA, index=out.index, dtype="object")
        for low, high, label in LEAD_TIME_BUCKETS:
            bucket = bucket.mask((lead > low) & (lead <= high), label)
        out["lead_time_bucket"] = bucket.astype("string")
    if "n_items" in out.columns:
        out["items_bucket"] = np.where(pd.to_numeric(out["n_items"], errors="coerce") > 1,
                                       "2+ items", "1 item")
    if "same_state" in out.columns:
        out["route_bucket"] = np.where(
            pd.to_numeric(out["same_state"], errors="coerce").fillna(0) > 0,
            "same-state seller", "cross-state seller",
        )
    if "purchase_month" in out.columns:
        out["month_bucket"] = (pd.to_numeric(out["purchase_month"], errors="coerce")
                               .astype("Int64").astype("string"))
    return out


def subgroup_report(
    df: pd.DataFrame,
    y_prob: Sequence[float],
    threshold: float,
    min_n: int = 100,
    columns: Sequence[str] = SUBGROUP_COLUMNS,
) -> pd.DataFrame:
    """Per-segment metrics on a split, computed at the frozen threshold.

    Small segments are still *measured* but flagged ``below_min_n``, so a 94% recall
    that rests on 11 orders cannot be quoted as a finding. That is the whole point of
    the exercise: one headline number hides a segment-wide failure.
    """
    frame = add_analysis_buckets(df).copy()
    frame["_prob"] = np.asarray(y_prob, dtype=float)

    rows: List[Dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for level, group in frame.groupby(column, dropna=False):
            if len(group) == 0:
                continue
            panel = compute_metrics(group[TARGET_COLUMN], group["_prob"], threshold=threshold)
            rows.append({
                "dimension": column,
                "level": str(level),
                "n": panel["n"],
                "n_late": panel["n_late"],
                "late_rate": panel["prevalence"],
                "average_precision": panel["average_precision"],
                "roc_auc": panel["roc_auc"],
                "precision": panel["precision"],
                "recall": panel["recall"],
                "f1": panel["f1"],
                "below_min_n": bool(panel["n"] < min_n),
            })
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["volume_rank"] = report.groupby("dimension")["n"].rank(ascending=False, method="first")
    report = report.loc[report["volume_rank"] <= SUBGROUP_TOP_N]
    return report.sort_values(["dimension", "volume_rank"]).reset_index(drop=True)


def permutation_importance_table(
    pipeline: Any,
    X: pd.DataFrame,
    y: Sequence[int],
    features: Sequence[str],
    n_repeats: int = 5,
    seed: int = 42,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Model-agnostic importance measured on a held-out split.

    Preferred over the Random-Forest impurity importance because impurity importance
    is biased towards high-cardinality features (``product_category``) and cannot be
    computed for a mixed preprocessing pipeline at all. The scoring metric is average
    precision, matching the headline metric.
    """
    result = permutation_importance(
        pipeline, X[list(features)], np.asarray(y).astype(int),
        scoring=scoring, n_repeats=int(n_repeats), random_state=int(seed), n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": list(features),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def error_examples(
    df: pd.DataFrame,
    y_prob: Sequence[float],
    threshold: float,
    n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Concrete false positives / false negatives for the report appendix."""
    cols = [c for c in ["order_id", "order_purchase_timestamp", "promised_date", "delivered_date",
                        "delay_days", "customer_state", "seller_state", "product_category",
                        "lead_time_days", "n_items", "order_value", TARGET_COLUMN] if c in df.columns]
    frame = df[cols].copy()
    frame["probability"] = np.asarray(y_prob, dtype=float)
    late = frame[TARGET_COLUMN].astype(int) == 1
    flagged = frame["probability"] >= float(threshold)
    fp = frame.loc[flagged & ~late].sort_values("probability", ascending=False).head(n)
    fn = frame.loc[~flagged & late].sort_values("probability").head(n)
    return {"false_positives": fp, "false_negatives": fn}


def evaluate_split(
    pipeline: Any,
    frame: pd.DataFrame,
    features: Sequence[str],
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Score one split with a frozen threshold; returns ``(metrics, probabilities)``."""
    probabilities = predict_positive_proba(pipeline, frame[list(features)])
    return compute_metrics(frame[TARGET_COLUMN], probabilities, threshold=threshold), probabilities
