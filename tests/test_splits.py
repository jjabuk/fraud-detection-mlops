from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dagster import Failure, build_asset_context

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.core.config import get_training_params
from fraud_detection.core.schema import CARD_ENTITY_COLUMN, MODEL_INPUT_TABLE, SPLIT_TABLE
from fraud_detection.orchestration.assets.splits import split_assignment
from fraud_detection.orchestration.resources import BigQueryResource

# BigQuery is always mocked here.

_STATS_ROWS = [
    {"split": "train", "rows_in_split": 413_378, "rows_with_card_seen_in_train": 413_378},
    {"split": "val", "rows_in_split": 88_581, "rows_with_card_seen_in_train": 80_000},
    {"split": "test", "rows_in_split": 88_581, "rows_with_card_seen_in_train": 78_000},
]


def _mock_bigquery_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    import pyarrow as pa
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value.to_arrow.return_value = pa.Table.from_pylist(_STATS_ROWS)
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: mock_client)
    return mock_client


def _split_query(mock_client: MagicMock) -> str:
    (query_text,), _kwargs = mock_client.query.call_args_list[0]
    return query_text


def test_writes_a_split_table_from_the_model_input(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)

    result = split_assignment(build_asset_context(), BigQueryResource(project="test-project"))

    query_text = _split_query(mock_client)
    assert f"test-project.features.{MODEL_INPUT_TABLE}" in query_text
    assert f"test-project.features.{SPLIT_TABLE}" in query_text

    assert getattr(result.metadata["train_rows"], "value", result.metadata["train_rows"]) == 413_378
    assert getattr(result.metadata["val_rows"], "value", result.metadata["val_rows"]) == 88_581
    assert getattr(result.metadata["test_rows"], "value", result.metadata["test_rows"]) == 88_581


def test_split_is_on_the_time_axis_and_never_random(monkeypatch):
    # A K-fold shuffle would train on a card's later transactions and
    # evaluate on its earlier ones -- leakage above the feature layer, which
    # no window frame can undo. See docs/point-in-time.md section 4.
    mock_client = _mock_bigquery_client(monkeypatch)

    split_assignment(build_asset_context(), BigQueryResource(project="test-project"))

    training_params = get_training_params()
    query_text = _split_query(mock_client)
    assert "TransactionDT" in query_text
    assert f"PERCENTILE_CONT(TransactionDT, {training_params.get('train_fraction', 0.70)})" in query_text
    assert f"PERCENTILE_CONT(TransactionDT, {training_params.get('val_start_fraction', 0.70)})" in query_text
    assert f"PERCENTILE_CONT(TransactionDT, {training_params.get('val_end_fraction', 0.85)})" in query_text
    assert "RAND()" not in query_text
    assert "FARM_FINGERPRINT" not in query_text


def test_split_boundaries_are_exact_not_approximate(monkeypatch):
    # APPROX_QUANTILES moved 56 rows between validation and test across two
    # materializations of byte-identical input. This project compares runs
    # constantly, so the boundaries have to be reproducible.
    mock_client = _mock_bigquery_client(monkeypatch)

    split_assignment(build_asset_context(), BigQueryResource(project="test-project"))

    query_text = _split_query(mock_client)
    assert "APPROX_QUANTILES" not in query_text
    assert "PERCENTILE_CONT" in query_text


def test_boundaries_keep_peers_at_one_timestamp_in_the_same_split(monkeypatch):
    # Same reasoning as the RANGE frames in feature engineering: a timestamp
    # is a unit. Using < on one bound and <= on the other would cut a group
    # of simultaneous transactions across two splits.
    mock_client = _mock_bigquery_client(monkeypatch)

    split_assignment(build_asset_context(), BigQueryResource(project="test-project"))

    query_text = _split_query(mock_client)
    assert "s.TransactionDT <= b.train_end" in query_text
    assert "s.TransactionDT > b.val_start" in query_text


def test_card_overlap_is_labelled_rather_than_filtered_out(monkeypatch):
    # Returning cards are the production reality, so dropping them would
    # evaluate the model on a population it never meets. The column exists so
    # metrics can be segmented on it.
    mock_client = _mock_bigquery_client(monkeypatch)

    result = split_assignment(build_asset_context(), BigQueryResource(project="test-project"))

    query_text = _split_query(mock_client)
    assert "card_seen_in_train" in query_text
    assert f"s.{CARD_ENTITY_COLUMN} AS card_entity" in query_text
    assert getattr(result.metadata["test_rows_on_card_seen_in_train"], "value", result.metadata["test_rows_on_card_seen_in_train"]) == 78_000


def test_query_failure_raises_dagster_failure(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    mock_client.query.return_value.result.side_effect = RuntimeError("boom")

    with pytest.raises(Failure, match="Split assignment failed"):
        split_assignment(build_asset_context(), BigQueryResource(project="test-project"))
