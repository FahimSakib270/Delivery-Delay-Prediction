"""Post-training evaluation: uncertainty, subgroups, calibration, figures and errors.

Trains nothing. Reads ``models/best_model.joblib`` and ``reports/test_predictions.csv``
(written by ``python -m src.train``) so that the numbers quoted in the report can be
regenerated — and challenged — without touching the model or the data split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import load_config, resolve
from .data_prep import TARGET_COLUMN, load_processed
from .evaluation import (
    bootstrap_ci,
    calibration_table,
    compute_metrics,
    cost_curve,
    error_examples,
    evaluate_split,
    expected_calibration_error,
    pr_curve_points,
    subgroup_report,
    permutation_importance_table,
)
from .plots import (
    plot_bootstrap_intervals,
    plot_calibration,
    plot_confusion,
    plot_cost_curve,
    plot_feature_importance,
    plot_lead_time_effect,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_subgroup_performance,
    plot_target_balance,
    plot_threshold_curve,
)
from .utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("evaluate")


def load_artifacts(
    cfg: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return ``(model_bundle, test_predictions, comparison)``."""
    cfg = cfg or load_config()
    models_dir = models_dir or resolve(cfg, "paths", "models_dir")
    reports_dir = reports_dir or resolve(cfg, "paths", "reports_dir")

    model_path = models_dir / "best_model.joblib"
    predictions_path = reports_dir / "test_predictions.csv"
    if not model_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            "Model artefacts not found. Run 'python -m src.train' first "
            f"(expected {model_path} and {predictions_path})."
        )

    bundle = joblib.load(model_path)
    metadata = load_json(resolve(cfg, "paths", "processed_dir") / "dataset_metadata.json")
    if bundle.get("dataset_sha256") != metadata.get("provenance", {}).get("processed_sha256"):
        raise ValueError("Model and prepared dataset checksums differ; rerun training for this dataset")
    predictions = pd.read_csv(predictions_path)
    for column in ("order_purchase_timestamp", "promised_date", "delivered_date"):
        if column in predictions.columns:
            predictions[column] = pd.to_datetime(predictions[column], errors="coerce")
    comparison_path = reports_dir / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    return bundle, predictions, comparison


def label_sensitivity(predictions: pd.DataFrame, threshold: float) -> Dict[str, Dict[str, float]]:
    """Metrics under both label definitions, at the same frozen probabilities.

    The primary definition compares calendar dates; the alternative compares raw
    timestamps. If the model's quality swings wildly between the two, the conclusion
    depends on a definitional choice and the report must say so.
    """
    out: Dict[str, Dict[str, float]] = {}
    for column, name in ((TARGET_COLUMN, "date_based (primary)"), ("is_late_ts", "timestamp_based")):
        if column in predictions.columns:
            metrics = compute_metrics(predictions[column].astype(int),
                                      predictions["probability"], threshold=threshold)
            out[name] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in metrics.items()}
    return out


def evaluate(
    cfg: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Compute every reported number and figure from the saved artefacts."""
    cfg = cfg or load_config()
    reports_dir = ensure_dir(reports_dir or resolve(cfg, "paths", "reports_dir"))
    figures_dir = ensure_dir(figures_dir or resolve(cfg, "paths", "figures_dir"))
    bundle, predictions, comparison = load_artifacts(cfg, models_dir, reports_dir)

    model = bundle["pipeline"]
    features = list(bundle["features"])
    threshold = float(bundle["threshold"])
    min_n = int(cfg["evaluation"]["subgroups_min_n"])
    n_boot = int(cfg["evaluation"]["bootstrap_n"])
    seed = int(cfg["project"]["random_seed"])

    test_metrics, probabilities = evaluate_split(model, predictions, features, threshold)
    metadata = load_json(resolve(cfg, "paths", "processed_dir") / "dataset_metadata.json")
    synthetic = bool(metadata.get("provenance", {}).get("synthetic", False))

    intervals = bootstrap_ci(predictions[TARGET_COLUMN], probabilities, threshold=threshold,
                             n_boot=n_boot, seed=seed)
    calibration = calibration_table(predictions[TARGET_COLUMN], probabilities)
    ece = expected_calibration_error(calibration)
    subgroups = subgroup_report(predictions, probabilities, threshold, min_n=min_n)
    importance = permutation_importance_table(model, predictions, predictions[TARGET_COLUMN],
                                              features, seed=seed)
    examples = error_examples(predictions, probabilities, threshold, n=10)
    sensitivity = label_sensitivity(predictions, threshold)

    cost_assumptions = {
        "cost_false_negative": float(cfg["business_case"]["cost_false_negative"]),
        "cost_false_positive": float(cfg["business_case"]["cost_false_positive"]),
    }
    validation_curve = pd.read_csv(reports_dir / "validation_cost_curve.csv") \
        if (reports_dir / "validation_cost_curve.csv").exists() else pd.DataFrame()
    cost_at_threshold = cost_curve(predictions[TARGET_COLUMN], probabilities,
                                   grid=[threshold], **cost_assumptions).iloc[0].to_dict()
    cost_at_half = cost_curve(predictions[TARGET_COLUMN], probabilities,
                              grid=[0.5], **cost_assumptions).iloc[0].to_dict()

    pr_curve = pr_curve_points(predictions[TARGET_COLUMN], probabilities)
    pr_curve["ap"] = test_metrics["average_precision"]
    fpr, tpr, _ = roc_curve(predictions[TARGET_COLUMN], probabilities)
    roc = {"fpr": fpr, "tpr": tpr, "auc": test_metrics["roc_auc"]}

    summary: Dict[str, Any] = {
        "model_name": bundle["model_name"],
        "threshold": threshold,
        "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in test_metrics.items()},
        "bootstrap_95": intervals,
        "calibration_ece": ece,
        "label_sensitivity": sensitivity,
        "cost": {"assumptions": cost_assumptions,
                 "at_frozen_threshold": cost_at_threshold,
                 "at_threshold_0_5": cost_at_half},
        "duplicate_check": {
            "test_rows": int(len(predictions)),
            "unique_order_ids": int(predictions["order_id"].nunique()) if "order_id" in predictions
            else None,
        },
        "subgroups_min_n": min_n,
        "n_subgroups_flagged_small": int(subgroups["below_min_n"].sum()) if not subgroups.empty else 0,
        "data_source": {"synthetic": synthetic, "kind": metadata.get("provenance", {})
                        .get("data_source_kind")},
        "model_trained_at": bundle.get("trained_at"),
    }

    manifest_path = reports_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    cost_threshold = manifest.get("model", {}).get("cost_optimal_threshold")

    # ---- persistence ---------------------------------------------------------
    subgroups.to_csv(reports_dir / "test_subgroups.csv", index=False)
    importance.to_csv(reports_dir / "permutation_importance.csv", index=False)
    calibration.to_csv(reports_dir / "test_calibration.csv", index=False)
    for name, frame in examples.items():
        frame.to_csv(reports_dir / f"error_examples_{name}.csv", index=False)
    save_json(summary, reports_dir / "evaluation_summary.json")

    if make_plots:
        df = load_processed(cfg=cfg)
        figures: Dict[str, str] = {}
        development = df.loc[df["order_purchase_timestamp"] < pd.Timestamp(bundle["split_info"]["split"]["periods"]["test_start"])].copy()
        figures["target_balance"] = str(plot_target_balance(development, figures_dir / "01_target_balance.png"))
        figures["lead_time_effect"] = str(plot_lead_time_effect(development, figures_dir / "02_lead_time_effect.png"))
        if not comparison.empty:
            figures["model_comparison"] = str(
                plot_model_comparison(comparison, figures_dir / "03_model_comparison.png"))
        figures["pr_curve"] = str(plot_pr_curve({bundle["model_name"]: pr_curve},
                                                figures_dir / "04_precision_recall.png"))
        figures["roc_curve"] = str(plot_roc_curve({bundle["model_name"]: roc},
                                                  figures_dir / "05_roc.png"))
        threshold_table = pd.read_csv(reports_dir / "validation_threshold_curve.csv")
        figures["threshold_curve"] = str(plot_threshold_curve(
            threshold_table, figures_dir / "06_validation_threshold_curve.png",
            chosen=threshold, cost_threshold=cost_threshold))
        if not validation_curve.empty:
            figures["cost_curve"] = str(plot_cost_curve(
                validation_curve, figures_dir / "07_cost_curve.png", chosen=threshold,
                cost_threshold=cost_threshold,
                assumptions=(f"assumes {cost_assumptions['cost_false_negative']:.0f} per missed delay, "
                             f"{cost_assumptions['cost_false_positive']:.0f} per false alarm")))
        figures["calibration"] = str(plot_calibration(calibration, figures_dir / "08_calibration.png", ece=ece))
        figures["confusion"] = str(plot_confusion(test_metrics, figures_dir / "09_confusion_matrix.png"))
        figures["score_distribution"] = str(plot_score_distribution(
            predictions, figures_dir / "10_score_distribution.png", threshold))
        figures["importance"] = str(plot_feature_importance(
            importance, figures_dir / "11_feature_importance.png"))
        figures["subgroups"] = str(plot_subgroup_performance(
            subgroups, figures_dir / "12_subgroup_recall.png"))
        figures["bootstrap"] = str(plot_bootstrap_intervals(
            intervals, figures_dir / "13_bootstrap_intervals.png"))
        summary["figures"] = {name: str(Path(path).relative_to(reports_dir)) for name, path in figures.items()}

    (reports_dir / "results_summary.md").write_text(
        markdown_summary(summary, comparison, subgroups, importance), encoding="utf-8"
    )
    save_json(summary, reports_dir / "evaluation_summary.json")
    from .experiments import run as run_experiments
    from .reporting import write_reports
    exploratory, policies = run_experiments(cfg)
    write_reports(cfg, summary, comparison, exploratory, policies)
    _print_summary(summary, subgroups)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not np.isfinite(number) else f"{number:.{digits}f}"


def _print_summary(summary: Dict[str, Any], subgroups: pd.DataFrame) -> None:
    metrics = summary["test_metrics"]
    intervals = summary["bootstrap_95"]
    print("")
    print("=" * 78)
    print("EVALUATION (test period, frozen threshold)")
    print("=" * 78)
    print(f"  model                : {summary['model_name']}")
    if summary["data_source"]["synthetic"]:
        print("  DATA SOURCE          : SYNTHETIC DEMONSTRATION DATA (not real Olist orders)")
    print(f"  threshold            : {summary['threshold']:.4f}")
    print(f"  rows / late rate     : {metrics['n']:,} / {metrics['prevalence']:.4f}")
    print(f"  average precision    : {_fmt(metrics['average_precision'])}"
          f"   [{_fmt(intervals.get('average_precision', {}).get('lower'))},"
          f" {_fmt(intervals.get('average_precision', {}).get('upper'))}]")
    print(f"  roc-auc              : {_fmt(metrics['roc_auc'])}"
          f"   [{_fmt(intervals.get('roc_auc', {}).get('lower'))},"
          f" {_fmt(intervals.get('roc_auc', {}).get('upper'))}]")
    print(f"  precision / recall   : {_fmt(metrics['precision'])} / {_fmt(metrics['recall'])}"
          f"   (f1 {_fmt(metrics['f1'])})")
    print(f"  accuracy (useless)   : {_fmt(metrics['accuracy'])}")
    print(f"  calibration ECE      : {_fmt(summary['calibration_ece'])}")
    print(f"  cost per 1 000       : "
          f"{_fmt(summary['cost']['at_frozen_threshold']['cost_per_1000_orders'], 2)}"
          f"   (never flagging: "
          f"{_fmt(summary['cost']['at_frozen_threshold']['cost_if_never_flagging_per_1000'], 2)})")
    if not subgroups.empty:
        weak = subgroups.loc[~subgroups["below_min_n"]].sort_values("recall").head(3)
        print("  weakest large groups : " +
              "; ".join(f"{row['dimension']}={row['level']} recall {row['recall']:.3f} "
                        f"(n={int(row['n'])})" for _, row in weak.iterrows()))
    print("=" * 78)


def markdown_summary(
    summary: Dict[str, Any],
    comparison: pd.DataFrame,
    subgroups: pd.DataFrame,
    importance: pd.DataFrame,
) -> str:
    """Machine-generated results block, so the written report cannot drift from the run."""
    metrics = summary["test_metrics"]
    lines = [
        "<!-- generated by python -m src.evaluate; do not edit by hand -->",
        f"# Results summary - {summary['model_name']}",
        "",
        "* data source: **"
        f"{'SYNTHETIC DEMONSTRATION DATA' if summary['data_source']['synthetic'] else summary['data_source']['kind']}**",
        f"* frozen decision threshold: **{summary['threshold']:.3f}**",
        f"* test rows: {metrics['n']:,} ({metrics['n_late']:,} delayed, "
        f"prevalence {metrics['prevalence']:.4f})",
        "",
        "## Headline test metrics",
        "",
        "| metric | value | 95% bootstrap interval |",
        "| --- | --- | --- |",
    ]
    for name in ("average_precision", "roc_auc", "precision", "recall", "f1", "brier", "accuracy"):
        interval = summary["bootstrap_95"].get(name, {})
        lines.append(
            f"| {name} | {_fmt(metrics.get(name, float('nan')))} | "
            f"[{_fmt(interval.get('lower', float('nan')))}, "
            f"{_fmt(interval.get('upper', float('nan')))}] |"
        )
    lines += ["", "## Candidate comparison (average precision per split)", ""]
    if not comparison.empty:
        table = comparison.pivot_table(index="model", columns="split", values="average_precision")
        columns = [c for c in ("train", "validation", "test") if c in table.columns]
        lines.append("| model | " + " | ".join(columns) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
        for model_name, row in table.iterrows():
            lines.append(f"| {model_name} | " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    cost = summary["cost"]
    lines += [
        "",
        "## Error cost (assumptions live in config: `business_case`)",
        "",
        f"* a missed delay is assumed to cost {cost['assumptions']['cost_false_negative']:.0f}, "
        f"a false alarm {cost['assumptions']['cost_false_positive']:.0f}",
        f"* at the frozen threshold: {_fmt(cost['at_frozen_threshold']['cost_per_1000_orders'], 2)} "
        f"per 1 000 orders (threshold 0.5: {_fmt(cost['at_threshold_0_5']['cost_per_1000_orders'], 2)}; "
        f"flagging nothing: {_fmt(cost['at_frozen_threshold']['cost_if_never_flagging_per_1000'], 2)})",
        "",
        "## Weakest large segments (recall at the frozen threshold)",
        "",
    ]
    if not subgroups.empty:
        big = subgroups.loc[~subgroups["below_min_n"]].sort_values("recall").head(8)
        lines.append("| dimension | level | n | recall | precision |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in big.iterrows():
            lines.append(f"| {row['dimension']} | {row['level']} | {int(row['n']):,} | "
                         f"{_fmt(row['recall'])} | {_fmt(row['precision'])} |")
    lines += ["", "## Top permutation importances (test period)", "",
              "| feature | mean drop in average precision | std |", "| --- | --- | --- |"]
    for _, row in importance.head(8).iterrows():
        lines.append(f"| {row['feature']} | {_fmt(row['importance_mean'])} | "
                     f"{_fmt(row['importance_std'])} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m src.evaluate``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate the frozen model artefacts: uncertainty, subgroups, figures.",
    )
    parser.add_argument("--config", default=None, help="Alternative YAML config path.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)
    evaluate(load_config(args.config), make_plots=not args.no_plots)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
