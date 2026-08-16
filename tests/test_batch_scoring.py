"""The scoring path's two structural claims, checked on the SQL it emits.

1. The velocity windows are computed over train ∪ test. A test row's `card_txn_count_prior`
   has to count the card's transactions in the *training* period, because that is what the
   same column counted when the model was fitted. Building the features from the test
   tables alone -- which this pipeline did -- gives the first test transaction of every
   card an empty window, on 98.6% of test rows.

2. Only the test rows are scored, and the selection happens *after* the windows, never
   before. A `WHERE origin = 'test'` applied first would be the original bug wearing a
   filter.

BigQuery is mocked, as everywhere else in this suite: what is under test is the statement,
not the warehouse.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dagster import build_asset_context
from google.cloud import bigquery

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.core.schema import (
    JOINED_TABLE,
    ORIGIN_COLUMN,
    SCORING_HISTORY_TABLE,
    TEST_JOINED_TABLE,
    TEST_MODEL_INPUT_TABLE,
)
from fraud_detection.feature_engineering.scoring_history import (
    align_to_training_schema,
    build_scoring_history_sql,
)
from fraud_detection.orchestration.assets.inference import (
    kaggle_test_model_input,
    scoring_history,
)
from fraud_detection.orchestration.resources import BigQueryResource

TRAIN_SCHEMA = [
    bigquery.SchemaField("TransactionID", "INT64"),
    bigquery.SchemaField("TransactionDT", "INT64"),
    bigquery.SchemaField("TransactionAmt", "FLOAT64"),
    bigquery.SchemaField("card1", "INT64"),
    bigquery.SchemaField("DeviceInfo", "STRING"),
    bigquery.SchemaField("isFraud", "FLOAT64"),
]


def _mock_client(monkeypatch, *, test_fields: list[str] | None = None) -> MagicMock:
    client = MagicMock()
    names = [f.name for f in TRAIN_SCHEMA] if test_fields is None else test_fields

    def get_table(table_id):
        table = MagicMock(num_rows=679_121, full_table_id=table_id)
        table.schema = (
            TRAIN_SCHEMA
            if JOINED_TABLE in table_id
            else [bigquery.SchemaField(name, "STRING") for name in names]
        )
        return table

    client.get_table.side_effect = get_table
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: client)
    return client


def test_the_union_carries_both_periods_and_labels_which_is_which(monkeypatch):
    client = _mock_client(monkeypatch)

    result = scoring_history(
        build_asset_context(),
        f"p.raw.{TEST_JOINED_TABLE}",
        BigQueryResource(project="p"),
    )

    (sql,), _ = client.query.call_args
    assert f"p.raw.{JOINED_TABLE}" in sql
    assert f"p.raw.{TEST_JOINED_TABLE}" in sql
    assert "UNION ALL" in sql
    assert f"'train' AS {ORIGIN_COLUMN}" in sql
    assert f"'test' AS {ORIGIN_COLUMN}" in sql
    assert result == f"p.raw.{SCORING_HISTORY_TABLE}"


def test_the_union_is_not_filtered_to_the_test_period():
    """The training rows are the entire reason the table exists.

    Filtering them out anywhere before the window functions run reinstates the empty-window
    bug, and it would be an easy "optimisation" to make while reading only this statement.
    """
    sql = build_scoring_history_sql(
        destination_table="p.raw.scoring_history",
        train_joined_table="p.raw.ieee_train_joined",
        test_joined_table="p.raw.test_joined",
        train_columns=["`TransactionID`"],
        test_columns=["`TransactionID`"],
    )

    assert "WHERE" not in sql.upper()


def test_a_column_the_test_tables_lack_arrives_as_a_typed_null():
    """isFraud exists in training and is unknown in the test period.

    `UNION ALL` matches by position, so a column present on one side and absent on the
    other cannot be dropped -- it has to be selected as a NULL of the training column's
    type, or every column after it shifts by one and the model reads the wrong values
    under the right names.
    """
    train_columns, test_columns = align_to_training_schema(
        TRAIN_SCHEMA, {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "DeviceInfo"}
    )

    assert len(train_columns) == len(test_columns) == len(TRAIN_SCHEMA)
    assert "CAST(NULL AS FLOAT64) AS `isFraud`" in test_columns
    assert test_columns[0] == "`TransactionID`"


def test_a_column_only_the_test_tables_have_is_dropped():
    _, test_columns = align_to_training_schema(
        TRAIN_SCHEMA, {f.name for f in TRAIN_SCHEMA} | {"some_new_column"}
    )

    assert not any("some_new_column" in column for column in test_columns)


def test_features_run_over_the_union_and_the_model_input_over_the_test_rows(monkeypatch):
    """The order that makes the fix a fix: aggregate over everything, select afterwards."""
    client = _mock_client(monkeypatch)

    result = kaggle_test_model_input(
        build_asset_context(),
        f"p.raw.{TEST_JOINED_TABLE}",
        f"p.raw.{SCORING_HISTORY_TABLE}",
        BigQueryResource(project="p"),
    )

    features_sql, model_input_sql = (call.args[0] for call in client.query.call_args_list)

    # The windows see the union ...
    assert f"p.raw.{SCORING_HISTORY_TABLE}" in features_sql
    assert "RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING" in features_sql
    assert f"p.raw.{TEST_JOINED_TABLE}" not in features_sql

    # ... and the rows scored are the test rows, selected by the join that comes after.
    assert f"p.raw.{TEST_JOINED_TABLE}" in model_input_sql
    assert result == f"p.features.{TEST_MODEL_INPUT_TABLE}"


@pytest.mark.parametrize("banned", ["LAG(", "LEAD(", "ROWS BETWEEN"])
def test_the_scoring_path_inherits_the_point_in_time_ban(monkeypatch, banned):
    """Same rule as training, checked on the statement the scoring path actually runs."""
    client = _mock_client(monkeypatch)
    kaggle_test_model_input(
        build_asset_context(),
        f"p.raw.{TEST_JOINED_TABLE}",
        f"p.raw.{SCORING_HISTORY_TABLE}",
        BigQueryResource(project="p"),
    )

    features_sql = client.query.call_args_list[0].args[0]

    assert banned.upper() not in features_sql.upper()
