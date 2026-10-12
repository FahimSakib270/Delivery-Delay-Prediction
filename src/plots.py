"""Figures for the report and the Streamlit app.

Every function writes a PNG next to the report and returns its path, so the report
generator can embed the images without knowing anything about matplotlib. Plotting is
kept in its own module because the pages must be readable without re-running a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display is available in CI or on a server

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_prep import TARGET_COLUMN  # noqa: E402
from .utils import ensure_dir  # noqa: E402

try:  # seaborn is optional: it only makes the default styling nicer
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    _HAS_SEABORN = True
except Exception:  # pragma: no cover - styling fallback
    _HAS_SEABORN = False

plt.rcParams.update({
    "figure.autolayout": True,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLOR_PRIMARY = "#1f4e79"
COLOR_ACCENT = "#c0504d"
COLOR_MUTED = "#7f7f7f"
COLOR_OK = "#4f8a4f"


def _save(fig: plt.Figure, path: Path) -> Path:
    """Persist a figure and always release its memory."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_target_balance(df: pd.DataFrame, path: Path) -> Path:
    """Delayed share per booking month: shows the class balance drifting over time."""
    frame = df.copy()
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"])
    monthly = (frame.assign(month=frame["order_purchase_timestamp"].dt.to_period("M").astype(str))
               .groupby("month")
               .agg(orders=(TARGET_COLUMN, "size"), late_rate=(TARGET_COLUMN, "mean"))
               .reset_index())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(monthly["month"], monthly["orders"], color=COLOR_PRIMARY)
    axes[0].set_ylabel("orders booked")
    axes[0].set_title("Order volume and delayed share by booking month")
    axes[1].plot(monthly["month"], monthly["late_rate"] * 100, marker="o", color=COLOR_ACCENT)
    axes[1].axhline(frame[TARGET_COLUMN].mean() * 100, ls="--", color=COLOR_MUTED,
                    label=f"overall {frame[TARGET_COLUMN].mean() * 100:.1f}%")
    axes[1].set_ylabel("% delivered late")
    axes[1].set_xlabel("booking month")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=90)
    return _save(fig, path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path, metric: str = "average_precision") -> Path:
    """Validation vs test ranking quality per candidate, with the dummy floor marked."""
    val = comparison.loc[comparison["split"] == "validation"]
    test = comparison.loc[comparison["split"] == "test"]
    order = val.sort_values(metric, ascending=False)["model"].tolist()
    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [float(val.loc[val["model"] == m, metric].iloc[0]) for m in order],
           width, label="validation", color=COLOR_PRIMARY)
    ax.bar(x + width / 2, [float(test.loc[test["model"] == m, metric].iloc[0]) for m in order],
           width, label="test", color=COLOR_ACCENT)
    floor = float(val.loc[val["model"] == "dummy", metric].iloc[0])
    ax.axhline(floor, ls="--", color=COLOR_MUTED, label=f"dummy floor ({floor:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Candidate comparison (higher is better)")
    ax.legend()
    return _save(fig, path)


def plot_pr_curve(
    curves: Dict[str, Dict[str, np.ndarray]],
    path: Path,
    title: str = "Precision-recall curves (test period)",
) -> Path:
    """PR curves with the prevalence baseline that a useless model would achieve."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    baseline = None
    for name, curve in curves.items():
        baseline = float(curve["prevalence"])
        label = name
        if "ap" in curve and np.isfinite(float(curve["ap"])):
            label = f"{name} (AP={float(curve['ap']):.3f})"
        ax.plot(curve["recall"], curve["precision"], lw=2, label=label)
    if baseline is not None:
        ax.axhline(baseline, ls="--", color=COLOR_MUTED,
                   label=f"prevalence baseline ({baseline:.3f})")
    ax.set_xlabel("recall (share of late orders caught)")
    ax.set_ylabel("precision (share of flags that are truly late)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, path)


def plot_roc_curve(curves: Dict[str, Dict[str, np.ndarray]], path: Path) -> Path:
    """ROC curves, drawn *with* the reminder that they flatter imbalanced problems."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], lw=2,
                label=f"{name} (AUC={curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=COLOR_MUTED, label="random")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("ROC curves (test period)\nflattered by the many easy negatives")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def plot_threshold_curve(
    table: pd.DataFrame,
    path: Path,
    chosen: Optional[float] = None,
    cost_threshold: Optional[float] = None,
) -> Path:
    """Precision / recall / F1 as the decision threshold moves."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for column, color in (("precision", COLOR_PRIMARY), ("recall", COLOR_ACCENT), ("f1", COLOR_OK)):
        ax.plot(table["threshold"], table[column], lw=2, color=color, label=column)
    if chosen is not None:
        ax.axvline(chosen, color="black", ls=":", label=f"chosen {chosen:.3f}")
    if cost_threshold is not None:
        ax.axvline(cost_threshold, color=COLOR_MUTED, ls="-.", label=f"cost-optimal {cost_threshold:.3f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_title("Validation trade-off: what the threshold buys and costs")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_cost_curve(
    curve: pd.DataFrame,
    path: Path,
    chosen: Optional[float] = None,
    cost_threshold: Optional[float] = None,
    assumptions: str = "",
) -> Path:
    """Expected cost per 1 000 orders against the threshold (validation period)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve["threshold"], curve["cost_per_1000_orders"], lw=2, color=COLOR_PRIMARY,
            label="cost of flagging + missing delays")
    if "cost_if_never_flagging_per_1000" in curve.columns:
        ax.axhline(float(curve["cost_if_never_flagging_per_1000"].iloc[0]), ls="--",
                   color=COLOR_MUTED, label="cost of flagging nothing")
    if chosen is not None:
        ax.axvline(chosen, color="black", ls=":", label=f"chosen F1 threshold {chosen:.3f}")
    if cost_threshold is not None:
        ax.axvline(cost_threshold, color=COLOR_ACCENT, ls="-.", label=f"cost-optimal {cost_threshold:.3f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected cost per 1 000 orders")
    ax.set_title("Cost curve\n" + assumptions)
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_calibration(table: pd.DataFrame, path: Path, ece: Optional[float] = None) -> Path:
    """Reliability diagram: are the predicted probabilities usable as probabilities?"""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color=COLOR_MUTED, label="perfect calibration")
    if not table.empty:
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", lw=2,
                color=COLOR_PRIMARY, label="model")
        sizes = 40 + 400 * table["n"] / max(1, table["n"].max())
        ax.scatter(table["mean_predicted"], table["observed_rate"], s=sizes,
                   color=COLOR_PRIMARY, alpha=0.35)
    title = "Calibration (test period)"
    if ece is not None and np.isfinite(ece):
        title += f"\nweighted mean gap (ECE) = {ece:.3f}"
    ax.set_title(title)
    ax.set_xlabel("predicted probability of delay")
    ax.set_ylabel("observed share delayed")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_confusion(metrics: Dict[str, float], path: Path) -> Path:
    """Confusion matrix at the frozen threshold, annotated with counts and rates."""
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    labels = [["true negative", "false positive"], ["false negative", "true positive"]]
    total = matrix.sum()
    for i in range(2):
        for j in range(2):
            share = matrix[i, j] / total if total else 0.0
            ax.text(j, i, f"{labels[i][j]}\n{int(matrix[i, j]):,}\n({share * 100:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#222222")
    ax.set_xticks([0, 1], ["predicted on time", "predicted late"])
    ax.set_yticks([0, 1], ["actually on time", "actually late"])
    ax.set_title(f"Confusion matrix (test)\nthreshold = {metrics['threshold']:.3f}")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return _save(fig, path)


def plot_score_distribution(predictions: pd.DataFrame, path: Path, threshold: float) -> Path:
    """Where flagged and missed orders sit relative to the frozen threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    on_time = predictions.loc[predictions[TARGET_COLUMN] == 0, "probability"]
    late = predictions.loc[predictions[TARGET_COLUMN] == 1, "probability"]
    bins = np.linspace(0, 1, 41)
    ax.hist(on_time, bins=bins, alpha=0.65, label=f"on time (n={len(on_time):,})",
            color=COLOR_PRIMARY, density=True)
    ax.hist(late, bins=bins, alpha=0.65, label=f"delayed (n={len(late):,})",
            color=COLOR_ACCENT, density=True)
    ax.axvline(threshold, color="black", ls=":", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("predicted probability of delay")
    ax.set_ylabel("density")
    ax.set_title("Separation of the two classes (test period)")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_feature_importance(table: pd.DataFrame, path: Path, top_n: int = 15, metric: str = "average precision") -> Path:
    """Horizontal importance bars with the permutation spread as error bars."""
    if table.empty:
        return path
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(top) + 2.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color=COLOR_PRIMARY, error_kw={"ecolor": COLOR_MUTED, "capsize": 3})
    ax.set_xlabel(f"drop in {metric} when the column is shuffled")
    ax.set_title(f"Permutation importance (top {len(top)})")
    return _save(fig, path)


def plot_subgroup_performance(report: pd.DataFrame, path: Path, metric: str = "recall") -> Path:
    """Per-segment performance, marking the segments that are too small to quote."""
    if report.empty:
        return path
    frame = report.copy()
    frame["label"] = frame["dimension"] + " = " + frame["level"]
    frame = frame.sort_values(metric, ascending=True).head(24)
    colors = [COLOR_MUTED if flag else COLOR_PRIMARY for flag in frame["below_min_n"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(frame) + 2.4))
    ax.barh(frame["label"], frame[metric], color=colors)
    overall = float((report["n"] * report[metric]).sum() / report["n"].sum())
    ax.axvline(overall, ls="--", color=COLOR_ACCENT, label=f"volume-weighted {metric} {overall:.3f}")
    ax.set_xlabel(metric)
    ax.set_title(f"{metric.capitalize()} by segment (grey = fewer than min_n rows)")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_lead_time_effect(df: pd.DataFrame, path: Path) -> Path:
    """The single strongest booking-time signal, shown honestly (binned, not smoothed)."""
    frame = df.copy()
    frame["lead_bin"] = pd.cut(pd.to_numeric(frame["lead_time_days"], errors="coerce"),
                               bins=[-1, 7, 10, 14, 20, 25, 30, 40, 60, 10_000])
    grouped = (frame.groupby("lead_bin", observed=True)
               .agg(orders=(TARGET_COLUMN, "size"), late_rate=(TARGET_COLUMN, "mean"))
               .reset_index())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar([str(i) for i in grouped["lead_bin"]], grouped["late_rate"] * 100, color=COLOR_PRIMARY)
    for position, (rate, orders) in enumerate(zip(grouped["late_rate"], grouped["orders"])):
        ax.text(position, rate * 100, f"n={orders:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("promised lead time (days from booking to promised date)")
    ax.set_ylabel("% delivered late")
    ax.set_title("Delay rate by promised lead time")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


def plot_bootstrap_intervals(intervals: Dict[str, Dict[str, float]], path: Path) -> Path:
    """Point estimates with their bootstrap confidence intervals."""
    names = [n for n, values in intervals.items() if np.isfinite(values.get("lower", np.nan))]
    if not names:
        return path
    fig, ax = plt.subplots(figsize=(8, 5))
    for position, name in enumerate(names):
        values = intervals[name]
        ax.plot([values["lower"], values["upper"]], [position, position], lw=3, color=COLOR_PRIMARY)
        ax.plot(values["point"], position, marker="o", color=COLOR_ACCENT)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("metric value (95% percentile bootstrap interval)")
    ax.set_title("Test metrics with sampling uncertainty")
    return _save(fig, path)
