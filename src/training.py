"""Model training, model selection, threshold freezing and artefact persistence.

The sequence implemented here is deliberately constrained:

1. hyper-parameters are tuned **inside the training period** with a *time-series* CV
   (never a shuffled K-fold, which would let a model train on the future);
2. the model is compared on the **validation period** using average precision;
3. the decision threshold is chosen on the validation period and then **frozen**;
4. the test period is scored **once**, after all decisions are made.

Model artefacts, the comparison table, the frozen threshold and the run manifest land in
``models/`` and ``reports/`` so that ``python -m src.evaluate`` and the Streamlit app can
read them without re-training anything.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

from .config import load_config, resolve
from .data_prep import MODEL_FEATURES, TARGET_COLUMN, load_processed
from .evaluation import (
    calibration_table,
    compute_metrics,
    cost_optimal_threshold,
    evaluate_split,
    predict_positive_proba,
    select_threshold,
)
from .features import build_model_pipeline, model_param_grid, transformed_feature_names
from .splitting import split_and_filter
from .utils import environment_info, ensure_dir, file_sha256, get_logger, save_json, set_seed, Timer

logger = get_logger("training")

#: Candidates, cheapest first. ``dummy`` is the floor every model must beat.
CANDIDATE_MODELS = ("dummy", "logistic_regression", "random_forest")

#: Hyper-parameter search is scored with the same metric used for model selection.
SEARCH_SCORING = "average_precision"
PRIMARY_SELECTION_METRIC = "average_precision"


def _search_n_jobs(model_name: str) -> int:
    """Avoid nested parallelism: the forest already uses every core for its trees."""
    return 1 if model_name == "random_forest" else -1


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    cv_splits: int = 3,
    seed: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Tune one candidate inside the training period and return ``(pipeline, info)``.

    ``TimeSeriesSplit`` is used so that every CV fold trains on the past and validates
    on the immediately following stretch of time, exactly like the outer split. With
    the configured search spaces this is at most 3 x 4 forest fits, which is what
    keeps the experiment laptop-sized.
    """
    seed = int(seed if seed is not None else cfg["project"]["random_seed"])
    grid = model_param_grid(cfg, model_name)
    pipeline = build_model_pipeline(model_name, cfg, seed=seed)

    info: Dict[str, Any] = {"model": model_name, "grid": grid, "cv_splits": int(cv_splits)}
    with Timer() as timer:
        if not grid:
            pipeline.fit(X, y)
            fixed_keys = set(cfg["modeling"][model_name])
            info["best_params"] = {k: v for k, v in pipeline.named_steps["model"]
                                   .get_params(deep=False).items() if k in fixed_keys}
            info["search_table"] = pd.DataFrame([{"note": "no searched hyper-parameters"}])
            # Nothing to search, but the baseline is still cross-validated with the same
            # TimeSeriesSplit so its score is a real number rather than a placeholder and
            # so the dummy floor is measured on the same footing as the tuned models.
            try:
                fold_scores = cross_val_score(
                    pipeline, X, y,
                    scoring=SEARCH_SCORING,
                    cv=TimeSeriesSplit(n_splits=int(cv_splits)),
                    error_score=float("nan"),
                )
                info["cv_scores"] = [float(s) for s in fold_scores]
                info["best_cv_score"] = float(np.nanmean(fold_scores))
            except ValueError as exc:  # e.g. a fold with a single class
                logger.warning("could not cross-validate %s: %s", model_name, exc)
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                scoring=SEARCH_SCORING,
                cv=TimeSeriesSplit(n_splits=int(cv_splits)),
                refit=True,
                n_jobs=_search_n_jobs(model_name),
                error_score=float("nan"),
            )
            search.fit(X, y)
            pipeline = search.best_estimator_
            info["best_params"] = dict(search.best_params_)
            info["best_cv_score"] = float(search.best_score_)
            info["search_table"] = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]
            ].sort_values("rank_test_score")
    info["seconds"] = round(timer.seconds, 2)
    cv_score = info.get("best_cv_score")
    logger.info(
        "%-20s tuned in %6.1fs  best CV %s = %s",
        model_name, timer.seconds, SEARCH_SCORING,
        "n/a" if cv_score is None else f"{cv_score:.4f}",
    )
    return pipeline, info


def fit_candidates(
    cfg: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
    cv_splits: int = 3,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Fit every candidate on the training period and tabulate metrics per split."""
    train = frames["train"].sort_values("order_purchase_timestamp", kind="stable")
    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN].astype(int)

    fitted: Dict[str, Any] = {}
    info: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for name in CANDIDATE_MODELS:
        pipeline, model_info = tune_model(name, X_train, y_train, cfg, cv_splits=cv_splits, seed=seed)
        fitted[name] = pipeline
        info[name] = model_info
        for split_name in ("train", "validation", "test"):
            frame = frames[split_name]
            probabilities = predict_positive_proba(pipeline, frame[MODEL_FEATURES])
            panel = compute_metrics(frame[TARGET_COLUMN], probabilities, threshold=0.5)
            rows.append({"model": name, "split": split_name, **panel})

    comparison = pd.DataFrame(rows)
    return fitted, comparison, info


def select_best_model(
    comparison: pd.DataFrame,
    metric: str = PRIMARY_SELECTION_METRIC,
) -> str:
    """Pick the model with the best **validation** average precision.

    Average precision (PR-AUC) is the selection metric because delays are a minority
    class: it depends only on how well the delayed orders are ranked, whereas ROC-AUC
    would be inflated by the many easy on-time orders. Brier score breaks ties towards
    the better-calibrated model.
    """
    validation = comparison.loc[comparison["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("no validation rows in the comparison table")
    ranked = validation.sort_values([metric, "brier"], ascending=[False, True])
    best = str(ranked.iloc[0]["model"])
    dummy_rows = validation.loc[validation["model"] == "dummy", metric]
    floor = float(dummy_rows.iloc[0]) if len(dummy_rows) else float("nan")
    logger.info("model selection by validation %s: %s (%.4f) vs dummy floor %.4f",
                metric, best, float(ranked.iloc[0][metric]), floor)
    if np.isfinite(floor) and float(ranked.iloc[0][metric]) <= floor:
        logger.warning("the selected model does not beat the dummy floor on %s; "
                       "report this instead of hiding it", metric)
    return best


def _git_commit() -> Optional[str]:
    """Best-effort short commit hash for provenance (``None`` outside a repo)."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return out.stdout.strip() or None


def train_and_evaluate(
    cfg: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    cv_splits: int = 3,
    persist: bool = True,
) -> Dict[str, Any]:
    """Run selection -> threshold freezing -> a single scoring of the test period."""
    cfg = cfg or load_config()
    seed = int(cfg["project"]["random_seed"])
    set_seed(seed)
    models_dir = ensure_dir(models_dir or resolve(cfg, "paths", "models_dir"))
    reports_dir = ensure_dir(reports_dir or resolve(cfg, "paths", "reports_dir"))

    df = load_processed(cfg=cfg)
    frames, split_info = split_and_filter(df, cfg)
    for name, frame in frames.items():
        if len(frame) < 20:
            raise ValueError(
                f"the {name} split has only {len(frame)} rows; "
                "check the split fractions and the label-maturity filter"
            )
        if frame[TARGET_COLUMN].nunique() < 2:
            logger.warning("the %s split contains a single class only", name)

    logger.info("folds after split + maturity filter: %s", {k: len(v) for k, v in frames.items()})

    with Timer() as timer:
        fitted, comparison, search_info = fit_candidates(cfg, frames, cv_splits=cv_splits, seed=seed)
    best_name = select_best_model(comparison)
    best = fitted[best_name]

    validation = frames["validation"]
    validation_prob = predict_positive_proba(best, validation[MODEL_FEATURES])
    thr_cfg = cfg["threshold_selection"]
    threshold, threshold_curve, threshold_info = select_threshold(
        validation[TARGET_COLUMN], validation_prob,
        objective=str(thr_cfg["objective"]),
        min_precision_floor=float(thr_cfg["min_precision_floor"]),
    )
    cost_cfg = cfg["business_case"]
    cost_threshold, cost_curve_table = cost_optimal_threshold(
        validation[TARGET_COLUMN], validation_prob,
        cost_false_negative=float(cost_cfg["cost_false_negative"]),
        cost_false_positive=float(cost_cfg["cost_false_positive"]),
    )

    # ---- the test period is scored exactly once, with everything frozen --------
    test = frames["test"]
    test_metrics, test_prob = evaluate_split(best, test, MODEL_FEATURES, threshold)
    test_metrics_at_half = compute_metrics(test[TARGET_COLUMN], test_prob, threshold=0.5)
    logger.info("TEST  AP=%.4f  ROC-AUC=%.4f  P=%.4f  R=%.4f  F1=%.4f  (threshold=%.4f)",
                test_metrics["average_precision"], test_metrics["roc_auc"],
                test_metrics["precision"], test_metrics["recall"], test_metrics["f1"], threshold)

    calibration = calibration_table(test[TARGET_COLUMN], test_prob)
    predictions = test.copy()
    predictions["probability"] = test_prob
    predictions["prediction"] = (test_prob >= threshold).astype(int)
    predictions["split"] = "test"

    model_path = models_dir / "best_model.joblib"
    bundle = {
        "pipeline": best,
        "model_name": best_name,
        "features": list(MODEL_FEATURES),
        "threshold": float(threshold),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(frames["train"])),
        "dataset_sha256": file_sha256(resolve(cfg, "paths", "processed_dir") / "orders_prepared.csv"),
        "label_definition": str(cfg["label"]["primary"]),
    }
    if persist:
        joblib.dump(bundle, model_path)

    results: Dict[str, Any] = {
        "model_name": best_name,
        "model_path": str(model_path) if persist else None,
        "threshold": float(threshold),
        "threshold_info": threshold_info,
        "cost_optimal_threshold": float(cost_threshold),
        "test_metrics": {k: _plain(v) for k, v in test_metrics.items()},
        "test_metrics_at_0_5": {k: _plain(v) for k, v in test_metrics_at_half.items()},
        "comparison": comparison,
        "threshold_curve": threshold_curve,
        "cost_curve": cost_curve_table,
        "calibration": calibration,
        "search_info": {k: {kk: vv for kk, vv in v.items() if kk != "search_table"}
                        for k, v in search_info.items()},
        "split_info": split_info,
        "predictions": predictions,
        "seconds": round(timer.seconds, 2),
    }

    if persist:
        comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
        threshold_curve.to_csv(reports_dir / "validation_threshold_curve.csv", index=False)
        cost_curve_table.to_csv(reports_dir / "validation_cost_curve.csv", index=False)
        calibration.to_csv(reports_dir / "test_calibration.csv", index=False)
        predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
        save_json(results["search_info"], reports_dir / "hyperparameter_search.json")
        save_json(_manifest(cfg, results, model_path, persist), reports_dir / "run_manifest.json")
        logger.info("artefacts written to %s and %s", models_dir, reports_dir)
    return results


def _plain(value: Any) -> Any:
    """Convert NumPy scalars to plain Python so JSON writing cannot fail."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _manifest(
    cfg: Dict[str, Any],
    results: Dict[str, Any],
    model_path: Path,
    persist: bool,
) -> Dict[str, Any]:
    """Everything needed to reproduce (or interrogate) this run."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": environment_info(),
        "config": cfg,
        "model": {
            "name": results["model_name"],
            "path": str(model_path) if persist else None,
            "sha256": file_sha256(model_path) if persist else None,
            "threshold": results["threshold"],
            "threshold_selection": results["threshold_info"],
            "cost_optimal_threshold": results["cost_optimal_threshold"],
            "hyperparameters": results["search_info"],
        },
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "splits": results["split_info"],
        "test_metrics": results["test_metrics"],
        "test_metrics_at_threshold_0_5": results["test_metrics_at_0_5"],
        "runtime_seconds": results["seconds"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m src.train``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Tune the candidates, freeze a threshold, score the test period once.",
    )
    parser.add_argument("--config", default=None, help="Alternative YAML config path.")
    parser.add_argument("--cv-splits", type=int, default=3,
                        help="TimeSeriesSplit folds used inside the training period.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = train_and_evaluate(cfg, cv_splits=args.cv_splits)
    metrics = results["test_metrics"]
    print("")
    print("=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print(f"  selected model      : {results['model_name']}")
    print(f"  frozen threshold    : {results['threshold']:.4f}"
          f"   (validation objective: {results['threshold_info']['objective']})")
    print(f"  cost-optimal (val)  : {results['cost_optimal_threshold']:.4f}")
    print(f"  TEST average prec.  : {metrics['average_precision']:.4f}   "
          f"(prevalence {metrics['prevalence']:.4f})")
    print(f"  TEST ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  TEST precision      : {metrics['precision']:.4f}")
    print(f"  TEST recall         : {metrics['recall']:.4f}")
    print(f"  TEST F1             : {metrics['f1']:.4f}")
    print(f"  runtime             : {results['seconds']:.1f}s")
    print("=" * 78)
    print("Next: python -m src.evaluate   (CIs, subgroups, calibration, figures)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
