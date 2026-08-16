from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dagster import Failure, build_asset_context

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.core.schema import (
    FEATURE_COLUMNS,
    FEATURE_TABLE,
    JOINED_TABLE,
    MODEL_INPUT_TABLE,
)
from fraud_detection.orchestration.assets.model_input import build_feature_column_list, model_input
from fraud_detection.orchestration.resources import BigQueryResource

# BigQuery is always mocked here.

# The six columns both source tables carry. If f.* were ever used instead of
# an explicit list, each of these would arrive twice and the query would fail
# -- or worse, succeed with suffixed duplicates.
SHARED_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "DeviceInfo",
    "isFraud",
]


def _mock_bigquery_client(
    monkeypatch: pytest.MonkeyPatch, *, num_rows: int = 590_540, num_columns: int = 442
) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_table.return_value = MagicMock(
        num_rows=num_rows,
        full_table_id=f"test-project:features.{MODEL_INPUT_TABLE}",
        schema=[MagicMock()] * num_columns,
    )
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: mock_client)
    return mock_client


def test_joins_the_raw_table_to_the_feature_table(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)

    result = model_input(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args
    assert f"test-project.raw.{JOINED_TABLE}" in query_text
    assert f"test-project.features.{FEATURE_TABLE}" in query_text
    assert f"test-project.features.{MODEL_INPUT_TABLE}" in query_text
    assert "ON j.TransactionID = f.TransactionID" in query_text

    assert getattr(result.metadata["rows_in_table"], "value", result.metadata["rows_in_table"]) == 590_540
    assert getattr(result.metadata["engineered_columns"], "value", result.metadata["engineered_columns"]) == len(FEATURE_COLUMNS)


def test_every_engineered_column_is_named_explicitly(monkeypatch):
    # A feature produced upstream but missing from FEATURE_COLUMNS would
    # never reach the model, silently. Naming them here is what makes that
    # a visible failure rather than a quiet one.
    mock_client = _mock_bigquery_client(monkeypatch)

    model_input(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args
    for column in FEATURE_COLUMNS:
        assert f"f.{column}" in query_text


def test_shared_columns_are_taken_from_the_raw_side_only(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)

    model_input(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args

    # Only the SELECT list -- f.TransactionID legitimately appears below it,
    # in the ON clause.
    select_list = query_text.split("FROM `")[0]

    assert "f.*" not in select_list
    for column in SHARED_COLUMNS:
        assert f"f.{column}" not in select_list


def test_join_is_inner_so_a_missing_row_surfaces_as_a_count_mismatch(monkeypatch):
    # Every transaction has exactly one row in each table. An outer join
    # would paper over an incomplete upstream asset with nulls in the
    # training matrix; an inner one shows up as a row count that does not
    # match 590,540.
    mock_client = _mock_bigquery_client(monkeypatch)

    model_input(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args
    assert "LEFT JOIN" not in query_text
    assert "FULL" not in query_text


def test_feature_column_list_is_prefixed_and_complete():
    rendered = build_feature_column_list(["a", "b"])

    assert rendered == "f.a,\n  f.b"


def test_query_failure_raises_dagster_failure(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    mock_client.query.return_value.result.side_effect = RuntimeError("boom")

    with pytest.raises(Failure, match="Model input assembly failed"):
        model_input(build_asset_context(), BigQueryResource(project="test-project"))

    mock_client.get_table.assert_not_called()
