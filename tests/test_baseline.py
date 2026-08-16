from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dagster import Failure, build_asset_context

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.core.schema import FEATURE_COLUMNS
from fraud_detection.orchestration.assets import baseline as baseline_module
from fraud_detection.orchestration.assets.baseline import (
    BASELINE_FEATURE_COLUMNS,
    BASELINE_METRICS_PATH,
    BASELINE_MODEL,
    VERTEX_BASELINE_MODEL_ID,
    bqml_baseline,
)
from fraud_detection.orchestration.resources import BigQueryResource, ModelArtifactStore

# BigQuery, GCS and the tracker are all mocked: the suite must not reach any
# cloud service or create an experiment run.


class _RecordingTracker:
    """Captures what would have been logged, without calling Vertex AI.

    Deliberately not a subclass of ExperimentTracker: that is a pydantic
    model, so instance state does not survive plain attribute assignment. The
    asset only ever calls log_run(), which is the whole point of the resource
    exposing one method instead of the SDK.
    """

    def __init__(self) -> None:
        self.logged: dict = {}

    def log_run(self, run_name, params, metrics) -> str:
        self.logged = {"run_name": run_name, "params": params, "metrics": metrics}
        # Mirrors ExperimentTracker, which returns the run name it recorded.
        return run_name


def _scored_rows(n_pos: int = 20, n_neg: int = 80) -> list[dict]:
    # 20 positives ranked above 80 negatives: a perfect separation, so PR-AUC
    # is 1.0 and the random floor is the positive rate, 0.2.
    return [{"y_true": 1, "y_score": 0.9} for _ in range(n_pos)] + [
        {"y_true": 0, "y_score": 0.1} for _ in range(n_neg)
    ]


def _mock_bigquery_client(monkeypatch: pytest.MonkeyPatch, rows=None) -> MagicMock:
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = (
        _scored_rows() if rows is None else rows
    )
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: mock_client)
    return mock_client


def _mock_gcs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_blob = MagicMock()
    mock_client = MagicMock()
    mock_client.bucket.return_value.blob.return_value = mock_blob
    monkeypatch.setattr(baseline_module.storage, "Client", lambda *a, **k: mock_client)
    return mock_blob


def _run(monkeypatch, rows=None, tracker=None):
    mock_client = _mock_bigquery_client(monkeypatch, rows)
    mock_blob = _mock_gcs(monkeypatch)
    result = bqml_baseline(
        build_asset_context(),
        BigQueryResource(project="test-project"),
        tracker or _RecordingTracker(),
        ModelArtifactStore(bucket="test-bucket"),
        split_assignment="test-project.features.splits",
    )
    return result, mock_client, mock_blob


def _queries(mock_client: MagicMock) -> tuple[str, str]:
    (create_sql,), _ = mock_client.query.call_args_list[0]
    (predict_sql,), _ = mock_client.query.call_args_list[1]
    return create_sql, predict_sql


def test_trains_on_the_train_split_and_scores_the_val_split(monkeypatch):
    result, mock_client, _ = _run(monkeypatch)

    create_sql, predict_sql = _queries(mock_client)
    assert f"test-project.features.{BASELINE_MODEL}" in create_sql
    assert "CREATE OR REPLACE MODEL" in create_sql
    assert "s.split = 'train'" in create_sql
    assert "s.split = 'val'" in predict_sql
    assert "ML.PREDICT" in predict_sql

    assert getattr(result.metadata["model_id"], "value", result.metadata["model_id"]) == f"test-project.features.{BASELINE_MODEL}"


def test_baseline_stays_narrow(monkeypatch):
    # The baseline answers "did the engineered features buy anything", so it
    # is the amount plus the seven velocity features -- not a second model
    # competing with LightGBM over all ~440 columns.
    _result, mock_client, _ = _run(monkeypatch)

    create_sql, _ = _queries(mock_client)
    assert BASELINE_FEATURE_COLUMNS == ["TransactionAmt", *FEATURE_COLUMNS]
    for column in BASELINE_FEATURE_COLUMNS:
        assert f"m.{column}" in create_sql
    assert "V1" not in create_sql


def test_preprocessing_lives_in_transform_so_prediction_cannot_forget_it(monkeypatch):
    # TRANSFORM is stored with the model and replayed on every prediction.
    # The proof that it works is negative: ML.PREDICT passes raw columns and
    # never restates the scaling. If it had to, that restatement would be the
    # skew bug TRANSFORM exists to prevent.
    _result, mock_client, _ = _run(monkeypatch)

    create_sql, predict_sql = _queries(mock_client)
    assert "TRANSFORM(" in create_sql
    for column in BASELINE_FEATURE_COLUMNS:
        assert f"ML.STANDARD_SCALER({column}) OVER() AS {column}" in create_sql
    assert "ML.STANDARD_SCALER" not in predict_sql


def test_model_is_registered_into_vertex_from_bigquery(monkeypatch):
    # Exercises the data-layer/registry seam before anything important
    # depends on it.
    _result, mock_client, _ = _run(monkeypatch)

    create_sql, _ = _queries(mock_client)
    assert "model_registry = 'vertex_ai'" in create_sql
    assert f"vertex_ai_model_id = '{VERTEX_BASELINE_MODEL_ID}'" in create_sql


def test_metrics_are_computed_here_not_read_from_ml_evaluate(monkeypatch):
    # One metric implementation for every model in the project. Comparing a
    # BQML-reported number against a sklearn-computed one is not a comparison.
    result, mock_client, _ = _run(monkeypatch)

    _create_sql, predict_sql = _queries(mock_client)
    assert "ML.EVALUATE" not in predict_sql

    assert getattr(result.metadata["pr_auc"], "value", result.metadata["pr_auc"]) == pytest.approx(1.0)
    assert getattr(result.metadata["roc_auc"], "value", result.metadata["roc_auc"]) == pytest.approx(1.0)
    # The positive rate is the PR-AUC a random ranker would score.
    assert getattr(result.metadata["positive_rate"], "value", result.metadata["positive_rate"]) == pytest.approx(0.2)


def test_metrics_are_published_where_the_validation_gate_looks(monkeypatch):
    _result, _mock_client, mock_blob = _run(monkeypatch)

    upload = mock_blob.upload_from_string.call_args
    assert upload is not None
    payload = upload[0][0]
    assert '"pr_auc"' in payload
    assert BASELINE_METRICS_PATH == "baseline/metrics.json"


def test_run_is_tracked_from_the_first_model(monkeypatch):
    # Instrumentation added after a good model exists never gets added.
    tracker = _RecordingTracker()
    result, _mock_client, _ = _run(monkeypatch, tracker=tracker)

    # Unique per execution: Vertex will not reopen a finished run, so a fixed
    # name would make the second materialization of this asset fail.
    assert tracker.logged["run_name"].startswith("bqml-baseline-logreg-")
    assert tracker.logged["run_name"] != "bqml-baseline-logreg-"
    assert tracker.logged["params"]["backend"] == "bigquery_ml"
    assert tracker.logged["params"]["feature_count"] == len(BASELINE_FEATURE_COLUMNS)
    assert "pr_auc" in tracker.logged["metrics"]
    assert getattr(result.metadata["experiment_run"], "value", result.metadata["experiment_run"]).startswith("bqml-baseline-logreg-")


def test_no_scored_rows_is_a_failure(monkeypatch):
    with pytest.raises(Failure, match="scored zero rows"):
        _run(monkeypatch, rows=[])


def test_validation_split_without_positives_is_a_failure(monkeypatch):
    # PR-AUC is undefined with no positives, and silently returning 0.0 would
    # look like a bad model rather than a broken split.
    with pytest.raises(Failure, match="no positive labels"):
        _run(monkeypatch, rows=[{"y_true": 0, "y_score": 0.1}] * 10)


def test_training_failure_raises_dagster_failure(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    _mock_gcs(monkeypatch)
    mock_client.query.return_value.result.side_effect = RuntimeError("boom")

    with pytest.raises(Failure, match="BQML baseline training failed"):
        bqml_baseline(
            build_asset_context(),
            BigQueryResource(project="test-project"),
            _RecordingTracker(),
            ModelArtifactStore(bucket="test-bucket"),
            split_assignment="test-project.features.splits",
        )
