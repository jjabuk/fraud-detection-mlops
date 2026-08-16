from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dagster import Failure, build_asset_context

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.core.schema import (
    CARD_ENTITY_COLUMN,
    DEVICE_ENTITY_COLUMN,
    FEATURE_TABLE,
    JOINED_TABLE,
)
from fraud_detection.orchestration.assets.feature_engineering import (
    CLIENT_ENTITY_COLUMN,
    WINDOW_1H_SECONDS,
    WINDOW_24H_SECONDS,
    transaction_features,
)
from fraud_detection.orchestration.resources import BigQueryResource

# BigQuery is always mocked here. A real run (against real data in the
# raw table) is a manual/integration check, not part of this suite.


def _mock_bigquery_client(monkeypatch: pytest.MonkeyPatch, *, num_rows: int = 590_540) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_table.return_value = MagicMock(
        num_rows=num_rows, full_table_id="test-project:features.transaction_features"
    )
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: mock_client)
    return mock_client


def test_runs_query_job_and_reports_row_count(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    context = build_asset_context()

    result = transaction_features(context, BigQueryResource(project="test-project"))

    mock_client.query.assert_called_once()
    (query_text,), _kwargs = mock_client.query.call_args

    assert f"test-project.raw.{JOINED_TABLE}" in query_text
    assert f"test-project.features.{FEATURE_TABLE}" in query_text
    assert "CREATE OR REPLACE TABLE" in query_text

    # Point-in-time guarantee: every windowed aggregate must stop at
    # "1 PRECEDING" (RANGE, not ROWS) so a row never sees itself, a peer
    # at the same TransactionDT, or the future. Six windows: client_prior,
    # client_24h, card_1h, card_24h, card_prior, device_24h.
    assert query_text.count("PRECEDING AND 1 PRECEDING") == 6
    assert f"{WINDOW_1H_SECONDS} PRECEDING AND 1 PRECEDING" in query_text
    assert f"{WINDOW_24H_SECONDS} PRECEDING AND 1 PRECEDING" in query_text
    assert "UNBOUNDED PRECEDING AND 1 PRECEDING" in query_text
    assert f"PARTITION BY {CARD_ENTITY_COLUMN}" in query_text
    assert f"PARTITION BY {DEVICE_ENTITY_COLUMN}" in query_text

    mock_client.get_table.assert_called_once_with(f"test-project.features.{FEATURE_TABLE}")
    assert getattr(result.metadata["rows_in_table"], "value", result.metadata["rows_in_table"]) == 590_540
    assert getattr(result.metadata["source_table"], "value", result.metadata["source_table"]) == f"test-project.raw.{JOINED_TABLE}"


def test_previous_transaction_gap_uses_an_unbounded_range_frame(monkeypatch):
    # The gap to the previous transaction needs every earlier row in scope,
    # not just a 1h/24h slice, so its frame starts at UNBOUNDED PRECEDING --
    # while still ending at 1 PRECEDING like all the others.
    mock_client = _mock_bigquery_client(monkeypatch)

    transaction_features(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args

    assert "MAX(TransactionDT) OVER card_prior" in query_text
    assert "RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING" in query_text
    assert "TransactionDT - prev_txn_dt_card AS seconds_since_prev_txn_card" in query_text


def test_card_aggregates_need_no_null_guard():
    # card1 is fully populated in this dataset -- 0 nulls across all 590,540
    # rows -- which is why it was chosen as the entity. The device guard is
    # not symmetrical carelessness; it is there because DeviceInfo is not.
    assert CARD_ENTITY_COLUMN == "card1"


def test_query_failure_raises_dagster_failure(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    mock_client.query.return_value.result.side_effect = RuntimeError("boom")
    context = build_asset_context()

    with pytest.raises(Failure, match="Feature engineering query failed"):
        transaction_features(context, BigQueryResource(project="test-project"))

    mock_client.get_table.assert_not_called()


def test_client_entity_is_null_rather_than_a_sentinel(monkeypatch):
    """The client uid must not fall back to a filled-in value.

    Filling a missing addr1 or D1 with -1 was measured: label purity on that
    subset falls to 79.4%, which is worse than card1 alone (84.8%), because
    the sentinel merges unrelated clients into one group. Well-formed uids
    reach 98.5%. Same lesson as the device guard -- a plausible wrong grouping
    is more dangerous than a missing one.
    """
    mock_client = _mock_bigquery_client(monkeypatch)

    transaction_features(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args

    assert "addr1 IS NULL OR D1 IS NULL" in query_text
    assert "IFNULL(addr1" not in query_text
    assert "COALESCE(addr1" not in query_text


def test_client_aggregates_are_all_null_guarded(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)

    transaction_features(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args

    # Every window function over the client partition sits inside a guard, so
    # rows without a reconstructable client get NULL rather than a count taken
    # across every other such row.
    for aggregate in (
        "COUNT(*) OVER client_prior",
        "COUNT(*) OVER client_24h",
        "AVG(TransactionAmt) OVER client_prior",
        "MAX(TransactionDT) OVER client_prior",
    ):
        assert f"IF({CLIENT_ENTITY_COLUMN} IS NULL, NULL, {aggregate})" in query_text


def test_client_uid_is_never_a_model_feature():
    # 217,735 levels. As a categorical the model would memorise clients, and
    # every client it memorised is one it can never meet again.
    from fraud_detection.training.data import EXCLUDED_COLUMNS

    assert CLIENT_ENTITY_COLUMN in EXCLUDED_COLUMNS
