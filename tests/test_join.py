from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dagster import Failure, build_asset_context

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.core.schema import JOINED_TABLE, RAW_IDENTITY_TABLE, RAW_TRANSACTION_TABLE
from fraud_detection.orchestration.assets.join import (
    NULL_COUNT_COLUMN,
    V_COLUMNS,
    build_null_count_expression,
    joined_transactions_identity,
)
from fraud_detection.orchestration.resources import BigQueryResource

# BigQuery is always mocked here. A real run against the raw tables is a
# manual/integration check, not part of this suite.


def _mock_bigquery_client(
    monkeypatch: pytest.MonkeyPatch, *, num_rows: int = 590_540, num_columns: int = 434
) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_table.return_value = MagicMock(
        num_rows=num_rows,
        full_table_id=f"test-project:raw.{JOINED_TABLE}",
        schema=[MagicMock()] * num_columns,
    )
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: mock_client)
    return mock_client


def test_joins_both_raw_tables_into_the_joined_table(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    context = build_asset_context()

    result = joined_transactions_identity(context, BigQueryResource(project="test-project"))

    mock_client.query.assert_called_once()
    (query_text,), _kwargs = mock_client.query.call_args

    assert f"test-project.raw.{RAW_TRANSACTION_TABLE}" in query_text
    assert f"test-project.raw.{RAW_IDENTITY_TABLE}" in query_text
    assert f"test-project.raw.{JOINED_TABLE}" in query_text
    assert "CREATE OR REPLACE TABLE" in query_text

    assert getattr(result.metadata["rows_in_table"], "value", result.metadata["rows_in_table"]) == 590_540
    assert getattr(result.metadata["columns_in_table"], "value", result.metadata["columns_in_table"]) == 434


def test_join_is_left_so_transactions_without_identity_survive(monkeypatch):
    # Identity covers only a subset of transactions. An inner join would
    # silently drop the rest of the training set.
    mock_client = _mock_bigquery_client(monkeypatch)

    joined_transactions_identity(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args
    assert "LEFT JOIN" in query_text
    assert "INNER JOIN" not in query_text


def test_transaction_id_is_not_duplicated_by_the_join(monkeypatch):
    # Both tables carry TransactionID; selecting i.* unguarded would produce
    # a duplicate column name and fail the query.
    mock_client = _mock_bigquery_client(monkeypatch)

    joined_transactions_identity(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args
    assert "i.* EXCEPT (TransactionID)" in query_text


def test_null_count_covers_the_whole_v_block_before_imputation(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)

    joined_transactions_identity(build_asset_context(), BigQueryResource(project="test-project"))

    (query_text,), _kwargs = mock_client.query.call_args

    # The missing-value pattern across V1-V339 is signal on this dataset, so
    # the count has to be taken on raw nulls -- nothing may impute them first.
    assert f"AS {NULL_COUNT_COLUMN}" in query_text
    assert "IFNULL" not in query_text
    assert "COALESCE" not in query_text
    for column in ("V1", "V170", "V339"):
        assert f"IF(t.{column} IS NULL, 1, 0)" in query_text


def test_null_count_expression_spans_every_v_column():
    expression = build_null_count_expression()

    assert expression.count("IS NULL") == len(V_COLUMNS) == 339


def test_null_count_expression_is_valid_sql_when_no_columns_given():
    assert build_null_count_expression([]) == "0"


def test_query_failure_raises_dagster_failure(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    mock_client.query.return_value.result.side_effect = RuntimeError("boom")

    with pytest.raises(Failure, match="Identity join failed"):
        joined_transactions_identity(
            build_asset_context(), BigQueryResource(project="test-project")
        )

    mock_client.get_table.assert_not_called()
