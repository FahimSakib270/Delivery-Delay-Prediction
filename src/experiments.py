"""Reproducible exploratory calibration and policy comparisons.

These are retrospective diagnostics, not an untouched confirmation experiment:
the historical test outcomes were already inspected during project development.
Calibration is fitted on validation only; no rule or budget is optimized on test.
Costs are hypothetical classification penalties, not measured intervention savings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import load_config, resolve
from .data_prep import TARGET_COLUMN
from .evaluation import compute_metrics, calibration_table, expected_calibration_error
from .utils import save_json


def policy_panel(name, y, flags, c_fn, c_fp):
    y = np.asarray(y, dtype=int)
    flags = np.asarray(flags, dtype=bool)
    tp = int(np.sum(flags & (y == 1)))
    fp = int(np.sum(flags & (y == 0)))
    fn = int(np.sum(~flags & (y == 1)))
    return {
        "policy": name, "n": len(y), "flagged": int(flags.sum()),
        "flag_rate": float(flags.mean()), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, int(y.sum())),
        "hypothetical_cost_per_1000": (fp * c_fp + fn * c_fn) * 1000 / len(y),
    }


def supplementary_experiments(validation, test, cfg, threshold):
    """Fit calibration on validation, then score fixed policies retrospectively."""
    if validation[TARGET_COLUMN].nunique() < 2:
        raise ValueError("calibration needs both classes in validation")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(validation["probability"], validation[TARGET_COLUMN])
    calibrated = calibrator.predict(test["probability"])
    y = test[TARGET_COLUMN].to_numpy(dtype=int)
    c_fn = float(cfg["business_case"]["cost_false_negative"])
    c_fp = float(cfg["business_case"]["cost_false_positive"])
    p_star = c_fp / (c_fn + c_fp) if c_fn + c_fp else 1.0
    rule_days = int(cfg.get("exploratory", {}).get("promise_rule_days", 7))
    rule = test["lead_time_days"].to_numpy() <= rule_days
    rows = [
        policy_panel("never_flag", y, np.zeros(len(y), dtype=bool), c_fn, c_fp),
        policy_panel("frozen_model_threshold", y, test["probability"] >= threshold, c_fn, c_fp),
        policy_panel(f"promise_le_{rule_days}_days", y, rule, c_fn, c_fp),
        policy_panel("validation_calibrated_cost_threshold", y, calibrated >= p_star, c_fn, c_fp),
    ]
    # Stable order breaks equal-score ties deterministically. Rule members are
    # retained; model top-ups rank only the remaining orders by the RAW score.
    remaining = np.flatnonzero(~rule)
    ranked = remaining[np.argsort(-test["probability"].to_numpy()[remaining], kind="stable")]
    for budget in cfg.get("exploratory", {}).get("top_up_budgets", [0.12, 0.16, 0.20]):
        k = min(len(remaining), max(0, int(np.ceil(len(y) * budget)) - int(rule.sum())))
        flags = rule.copy()
        flags[ranked[:k]] = True
        rows.append(policy_panel(f"rule_plus_model_{budget:.0%}", y, flags, c_fn, c_fp))
        # Exact expectation over uniformly random top-ups, avoiding seed noise.
        expected_tp = float(y[rule].sum()) + k * float(y[remaining].mean()) if len(remaining) else float(y.sum())
        flagged = int(rule.sum()) + k
        expected_fp, expected_fn = flagged - expected_tp, float(y.sum()) - expected_tp
        rows.append({
            "policy": f"rule_plus_random_{budget:.0%}_expectation", "n": len(y),
            "flagged": flagged, "flag_rate": flagged / len(y), "tp": expected_tp,
            "fp": expected_fp, "fn": expected_fn,
            "precision": expected_tp / max(1, flagged),
            "recall": expected_tp / max(1, int(y.sum())),
            "hypothetical_cost_per_1000": (expected_fp * c_fp + expected_fn * c_fn) * 1000 / len(y),
        })
    summary = {
        "status": "exploratory_retrospective_not_confirmatory",
        "calibration_fit_split": "validation_only",
        "rule_days": rule_days,
        "cost_assumptions": {"false_negative": c_fn, "false_positive": c_fp},
        "calibrated_cost_threshold": p_star,
        "raw_test": compute_metrics(y, test["probability"], threshold),
        "calibrated_test": compute_metrics(y, calibrated, p_star),
        "raw_ece": expected_calibration_error(calibration_table(y, test["probability"])),
        "calibrated_ece": expected_calibration_error(calibration_table(y, calibrated)),
        "ranking_note": "Isotonic calibration is non-decreasing but may introduce ties; ranking metrics are recomputed.",
        "cost_note": "Hypothetical error penalties; no measured profit or causal intervention effect.",
    }
    return summary, pd.DataFrame(rows), calibrated


def run(cfg=None):
    cfg = cfg or load_config()
    reports = resolve(cfg, "paths", "reports_dir")
    validation = pd.read_csv(reports / "validation_predictions.csv")
    test = pd.read_csv(reports / "test_predictions.csv")
    import json
    manifest = json.loads((reports / "run_manifest.json").read_text())
    summary, policies, calibrated = supplementary_experiments(
        validation, test, cfg, float(manifest["model"]["threshold"])
    )
    summary["data_source_kind"] = manifest["config"]["data_source"]["mode"]
    save_json(summary, reports / "exploratory_summary.json")
    policies.to_csv(reports / "exploratory_policies.csv", index=False)
    pd.DataFrame({"order_id": test["order_id"], "raw_score": test["probability"],
                  "calibrated_estimate": calibrated, TARGET_COLUMN: test[TARGET_COLUMN]}).to_csv(
        reports / "exploratory_calibrated_predictions.csv", index=False
    )
    return summary, policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    summary, policies = run(load_config(args.config))
    print(summary["status"])
    print(policies.to_string(index=False))


if __name__ == "__main__":
    main()
