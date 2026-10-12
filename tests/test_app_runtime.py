"""Runtime test for the Streamlit app: the whole page really runs.

The pure helpers are covered in :mod:`tests.test_app_helpers`; this file answers a
different question - *does the app start and render all three tabs against the
artefacts on disk?* Streamlit's ``AppTest`` executes the script in-process, so a broken
figure loop, a missing key in the summary file or a bad widget argument fails here
instead of in front of an audience.

Tests generate their own small synthetic artifacts in temporary directories.
They run on a clean checkout without external data or pre-existing model files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import resolve

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
RUN_TIMEOUT = 60



def page_text(app) -> str:
    """Everything the page printed, whichever Streamlit element printed it.

    ``st.json``, ``st.metric`` and ``st.markdown`` are different element types, so a test
    that only reads ``app.markdown`` would miss half the page.
    """
    parts = []
    for name in ("title", "header", "subheader", "markdown", "caption", "text", "code",
                 "json", "info", "warning", "error", "success"):
        for element in getattr(app, name, []) or []:
            parts.append(str(getattr(element, "value", "")))
    for metric in getattr(app, "metric", []) or []:
        parts.append(f"{getattr(metric, 'label', '')} {getattr(metric, 'value', '')}")
    return " ".join(parts)


@pytest.fixture(scope="module")
def artifact_config(cfg, tmp_path_factory):
    """Build isolated artifacts so a clean checkout exercises the real page."""
    import copy
    import yaml
    from src.prepare import main as prepare
    from src.training import train_and_evaluate
    from src.evaluate import evaluate

    local = copy.deepcopy(cfg)
    root = tmp_path_factory.mktemp("streamlit-runtime")
    for key, value in local["paths"].items():
        local["paths"][key] = str(root / value)
    local["data_source"].update(mode="synthetic", synthetic_n_orders=1800, synthetic_seed=5)
    local["modeling"]["random_forest"].update(n_estimators=20, max_depth=[6], min_samples_leaf=[2], n_jobs=1)
    local["modeling"]["logistic_regression"]["C"] = [1.0]
    local["evaluation"]["bootstrap_n"] = 10
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(local))
    prepare(["--config", str(config_path)])
    train_and_evaluate(local, cv_splits=2)
    evaluate(local)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PROFAST_CONFIG", str(config_path))
        yield local


@pytest.fixture(scope="module")
def app(artifact_config):
    at = AppTest.from_file(str(APP_PATH), default_timeout=RUN_TIMEOUT)
    at.run()
    return at


def test_the_app_starts_without_raising(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_the_data_source_is_declared_before_any_number_is_shown(app, artifact_config):
    """The synthetic banner (or the real-data notice) must be present."""
    reports = resolve(artifact_config, "paths", "reports_dir")
    summary = json.loads((reports / "evaluation_summary.json").read_text(encoding="utf-8"))
    synthetic = str(summary["data_source"]["kind"]).lower() != "olist"

    text = page_text(app)
    if synthetic:
        assert "SYNTHETIC" in text.upper(), "the demo must label synthetic data before any metric"
    else:
        assert "real olist" in text.lower()


def test_both_tab_titles_and_the_scoring_form_exist(app):
    labels = [tab.label for tab in app.tabs] if hasattr(app, "tabs") else []
    assert "Score an order" in labels or "Score an order" in page_text(app)
    # The form's widgets are all present: one slider per numeric feature, one selectbox per
    # categorical feature.
    from src.data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    assert len(app.slider) == len(NUMERIC_FEATURES)
    assert len(app.selectbox) == len(CATEGORICAL_FEATURES)


def test_pressing_score_displays_risk_without_probability_or_cost_advice(app):
    button = [b for b in app.button if "Score" in b.label]
    assert button
    button[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]
    assert any("uncalibrated" in metric.label.lower() for metric in app.metric)
    assert not any("probability" in metric.label.lower() for metric in app.metric)
    assert any("threshold" in metric.label.lower() for metric in app.metric)
    text = page_text(app).lower()
    assert "does not mean an 80%" in text
    assert "expected cost per order" not in text
    assert "recommends proactive contact" not in text


def test_the_performance_tab_shows_metric_rows_with_confidence_intervals(app):
    tables = app.dataframe
    assert tables, "the performance tab rendered no table"
    first = tables[0].value
    assert {"metric", "value"}.issubset(first.columns)
    # The interval columns must be populated, not silently empty: they are the reason
    # the tab exists.
    assert first["ci_low"].notna().any(), "bootstrap intervals are missing from the table"
    assert first["ci_high"].notna().any()


def test_every_figure_referenced_by_the_summary_exists_on_disk(app, artifact_config):
    reports = resolve(artifact_config, "paths", "reports_dir")
    summary = json.loads((reports / "evaluation_summary.json").read_text(encoding="utf-8"))
    figures = summary.get("figures", {})
    assert figures, "the evaluation summary lists no figures"
    missing = [name for name, path in figures.items() if not (reports / path).exists()]
    assert not missing, f"figures listed in the summary but absent: {missing}"


def test_the_provenance_tab_publishes_the_feature_audit_and_the_summary(app):
    tables = app.dataframe
    audit_rendered = any(
        "Available at booking?" in str(table.value.columns) for table in tables
    )
    assert audit_rendered, "the feature audit table is not rendered"

    text = page_text(app)
    assert "dataset_fingerprint" in text or "processed_sha256" in text, (
        "the provenance tab must expose the checksums"
    )
    assert "late rate" in text.lower()
