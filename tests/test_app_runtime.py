"""Runtime test for the Streamlit app: the whole page really runs.

The pure helpers are covered in :mod:`tests.test_app_helpers`; this file answers a
different question - *does the app start and render all three tabs against the
artefacts on disk?* Streamlit's ``AppTest`` executes the script in-process, so a broken
figure loop, a missing key in the summary file or a bad widget argument fails here
instead of in front of an audience.

The tests are skipped (with a clear reason) when the pipeline artefacts are missing,
because the app legitimately refuses to render without them.
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


def _artefacts_present() -> bool:
    cfg = {"paths": {"models_dir": "models", "processed_dir": "data/processed",
                     "reports_dir": "reports"}}
    return (
        (resolve(cfg, "paths", "models_dir") / "best_model.joblib").exists()
        and (resolve(cfg, "paths", "processed_dir") / "orders_prepared.csv").exists()
        and (resolve(cfg, "paths", "reports_dir") / "evaluation_summary.json").exists()
    )


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


pytestmark = pytest.mark.skipif(
    not _artefacts_present(),
    reason="pipeline artefacts missing: run src.prepare, src.train and src.evaluate first",
)


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=RUN_TIMEOUT)
    at.run()
    return at


def test_the_app_starts_without_raising(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_the_data_source_is_declared_before_any_number_is_shown(app):
    """The synthetic banner (or the real-data notice) must be present."""
    reports = resolve({"paths": {"reports_dir": "reports"}}, "paths", "reports_dir")
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


def test_pressing_score_produces_a_probability_and_a_cost_comparison(app):
    """The end-to-end user journey, driven by the widget API."""
    button = [b for b in app.button if "Score" in b.label]
    assert button, "the 'Score this order' submit button is missing"
    button[0].click().run()

    assert not app.exception, [str(e) for e in app.exception]
    values = [metric.value for metric in app.metric]
    assert any("%" in value for value in values), f"no probability metric was rendered: {values}"
    assert any("threshold" in metric.label.lower() for metric in app.metric)

    text = page_text(app).lower()
    assert "expected cost" in text
    assert "break-even" in text
    assert "ignore the risk" in text and "flag for proactive contact" in text


def test_the_performance_tab_shows_metric_rows_with_confidence_intervals(app):
    tables = app.dataframe
    assert tables, "the performance tab rendered no table"
    first = tables[0].value
    assert {"metric", "value"}.issubset(first.columns)
    # The interval columns must be populated, not silently empty: they are the reason
    # the tab exists.
    assert first["ci_low"].notna().any(), "bootstrap intervals are missing from the table"
    assert first["ci_high"].notna().any()


def test_every_figure_referenced_by_the_summary_exists_on_disk(app):
    reports = resolve({"paths": {"reports_dir": "reports"}}, "paths", "reports_dir")
    summary = json.loads((reports / "evaluation_summary.json").read_text(encoding="utf-8"))
    figures = summary.get("figures", {})
    assert figures, "the evaluation summary lists no figures"
    missing = [name for name, path in figures.items() if not Path(path).exists()]
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
