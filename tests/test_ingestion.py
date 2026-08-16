from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from dagster import Failure, build_asset_context
from google.cloud import bigquery

import fraud_detection.orchestration.resources as resources_module
from fraud_detection.orchestration import raw_load
from fraud_detection.orchestration.assets import ingestion as ingestion_module
from fraud_detection.orchestration.assets.ingestion import (
    raw_identity_bigquery,
    raw_transactions_bigquery,
)
from fraud_detection.orchestration.resources import (
    BigQueryResource,
    IdentityRawCsvSourceResource,
    RawCsvSourceResource,
)

# ---------------------------------------------------------------------------
# raw_transactions_bigquery -- BigQuery client is always mocked. Real
# BigQuery calls are a manual/integration check (see README), not part of
# this suite.
# ---------------------------------------------------------------------------


def _mock_bigquery_client(monkeypatch: pytest.MonkeyPatch, *, num_rows: int = 5) -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_table.return_value = MagicMock(
        num_rows=num_rows, full_table_id="test-project:raw.ieee_train_transaction_raw"
    )
    monkeypatch.setattr(resources_module.bigquery, "Client", lambda *a, **k: mock_client)
    return mock_client


def test_bigquery_asset_local_source_uses_load_table_from_file(monkeypatch):
    mock_client = _mock_bigquery_client(monkeypatch)
    import contextlib
    import io
    @contextlib.contextmanager
    def _mock_open(*args, **kwargs):
        yield io.BytesIO(b"TransactionID,TransactionDT,TransactionAmt,isFraud,DeviceInfo,id_01\n1,100,10.0,0,Windows,0.5\n")
    monkeypatch.setattr(ingestion_module, "_open_source", _mock_open)
    context = build_asset_context()

    result = [r for r in raw_transactions_bigquery(
        context,
        raw_csv_source=RawCsvSourceResource(),  # default: committed sample
        bigquery_resource=BigQueryResource(project="test-project"),
    )][-1]

    mock_client.load_table_from_uri.assert_not_called()
    mock_client.load_table_from_file.assert_called_once()
    args, kwargs = mock_client.load_table_from_file.call_args
    assert args[1] == "test-project.raw.ieee_train_transaction_raw"
    job_config = kwargs["job_config"]
    assert job_config.autodetect is True
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE

    assert getattr(result.metadata["rows_in_table"], "value", result.metadata["rows_in_table"]) == 5


def test_bigquery_asset_gcs_source_without_pinned_schema_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion_module, "BQ_SCHEMA_PATH_TRANSACTIONS", tmp_path / "does_not_exist.json")
    mock_client = _mock_bigquery_client(monkeypatch)
    import contextlib
    import io
    @contextlib.contextmanager
    def _mock_open(*args, **kwargs):
        yield io.BytesIO(b"TransactionID,TransactionDT,TransactionAmt,isFraud,DeviceInfo,id_01\n1,100,10.0,0,Windows,0.5\n")
    monkeypatch.setattr(ingestion_module, "_open_source", _mock_open)
    monkeypatch.setattr(ingestion_module, "_get_gcs_etag", lambda uri, proj: "mock_etag")
    context = build_asset_context()

    with pytest.raises(Failure, match="Pinned BigQuery schema not found"):
        list(raw_transactions_bigquery(
            context,
            raw_csv_source=RawCsvSourceResource(uri="gs://fraud-bucket/train_transaction.csv"),
            bigquery_resource=BigQueryResource(project="test-project"),
        ))

    mock_client.load_table_from_uri.assert_not_called()
    mock_client.load_table_from_file.assert_not_called()


def test_bigquery_asset_gcs_source_with_pinned_schema_uses_load_table_from_uri(
    monkeypatch, tmp_path
):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps([{"name": "TransactionID", "type": "INTEGER", "mode": "REQUIRED"}]))
    monkeypatch.setattr(ingestion_module, "BQ_SCHEMA_PATH_TRANSACTIONS", schema_path)

    mock_client = _mock_bigquery_client(monkeypatch, num_rows=590_540)
    import contextlib
    import io
    @contextlib.contextmanager
    def _mock_open(*args, **kwargs):
        yield io.BytesIO(b"TransactionID,TransactionDT,TransactionAmt,isFraud,DeviceInfo,id_01\n1,100,10.0,0,Windows,0.5\n")
    monkeypatch.setattr(ingestion_module, "_open_source", _mock_open)
    monkeypatch.setattr(ingestion_module, "_get_gcs_etag", lambda uri, proj: "mock_etag")
    mock_client.schema_from_json.return_value = [bigquery.SchemaField("TransactionID", "INTEGER")]
    context = build_asset_context()

    result = [r for r in raw_transactions_bigquery(
        context,
        raw_csv_source=RawCsvSourceResource(uri="gs://fraud-bucket/train_transaction.csv"),
        bigquery_resource=BigQueryResource(project="test-project"),
    )][-1]

    mock_client.load_table_from_file.assert_not_called()
    mock_client.load_table_from_uri.assert_called_once()
    args, kwargs = mock_client.load_table_from_uri.call_args
    assert args[0] == "gs://fraud-bucket/train_transaction.csv"

    # The load lands in staging, not in the raw table: the file spells whole numbers as
    # "1.0" and BigQuery will not parse that into an INT64 column. The pinned types are
    # restored by the cast below.
    assert args[1] == "test-project.raw.ieee_train_transaction_raw_staging"
    job_config = kwargs["job_config"]
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert job_config.autodetect is not True  # pinned schema, not autodetected
    assert [f.field_type for f in job_config.schema] == ["FLOAT64"]
    # Read twice, and both reads matter: once by the skip-guard, to compare the live
    # table's types against the pin, and once by the load itself. The guard's read is the
    # one added on 2026-08-16 — without it a retype changes neither row count nor source
    # etag, the guard reports "already current", and the load that would apply the new
    # types never runs.
    assert mock_client.schema_from_json.call_count == 2
    assert {call.args for call in mock_client.schema_from_json.call_args_list} == {
        (str(schema_path),)
    }

    # ... and one query writes the real table, under the pinned INT64 type.
    cast_sql, cast_kwargs = mock_client.query.call_args
    assert "CAST(`TransactionID` AS INT64)" in cast_sql[0]
    destination = cast_kwargs["job_config"].destination
    assert f"{destination.project}.{destination.dataset_id}.{destination.table_id}" == (
        "test-project.raw.ieee_train_transaction_raw"
    )

    # Staging is the same bytes again; it must not survive the run.
    mock_client.delete_table.assert_called_once()
    assert mock_client.delete_table.call_args.args[0].endswith("_staging")

    assert getattr(result.metadata["rows_in_table"], "value", result.metadata["rows_in_table"]) == 590_540


# ---------------------------------------------------------------------------
# raw_identity_bigquery
# ---------------------------------------------------------------------------


def test_identity_bigquery_fails_instead_of_skipping_when_local_file_absent(
    monkeypatch, tmp_path
):
    mock_client = _mock_bigquery_client(monkeypatch)
    import contextlib
    import io
    @contextlib.contextmanager
    def _mock_open(*args, **kwargs):
        yield io.BytesIO(b"TransactionID,TransactionDT,TransactionAmt,isFraud,DeviceInfo,id_01\n1,100,10.0,0,Windows,0.5\n")
    monkeypatch.setattr(ingestion_module, "_open_source", _mock_open)

    with pytest.raises(Failure, match="Local identity CSV not found"):
        list(raw_identity_bigquery(
            build_asset_context(),
            identity_raw_csv_source=IdentityRawCsvSourceResource(uri=str(tmp_path / "nope.csv")),
            bigquery_resource=BigQueryResource(project="test-project"),
        ))

    mock_client.load_table_from_file.assert_not_called()
    mock_client.load_table_from_uri.assert_not_called()


def test_identity_bigquery_gcs_source_without_pinned_schema_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ingestion_module, "BQ_SCHEMA_PATH_IDENTITY", tmp_path / "does_not_exist.json"
    )
    mock_client = _mock_bigquery_client(monkeypatch)
    import contextlib
    import io
    @contextlib.contextmanager
    def _mock_open(*args, **kwargs):
        yield io.BytesIO(b"TransactionID,TransactionDT,TransactionAmt,isFraud,DeviceInfo,id_01\n1,100,10.0,0,Windows,0.5\n")
    monkeypatch.setattr(ingestion_module, "_open_source", _mock_open)
    monkeypatch.setattr(ingestion_module, "_get_gcs_etag", lambda uri, proj: "mock_etag")

    with pytest.raises(Failure, match="Pinned BigQuery schema not found"):
        list(raw_identity_bigquery(
            build_asset_context(),
            identity_raw_csv_source=IdentityRawCsvSourceResource(
                uri="gs://fraud-bucket/train_identity.csv"
            ),
            bigquery_resource=BigQueryResource(project="test-project"),
        ))

    mock_client.load_table_from_uri.assert_not_called()


def test_identity_bigquery_gcs_source_with_pinned_schema_truncates_the_table(
    monkeypatch, tmp_path
):
    schema_path = tmp_path / "identity_schema.json"
    schema_path.write_text(
        json.dumps([{"name": "TransactionID", "type": "INTEGER", "mode": "REQUIRED"}])
    )
    monkeypatch.setattr(ingestion_module, "BQ_SCHEMA_PATH_IDENTITY", schema_path)

    mock_client = _mock_bigquery_client(monkeypatch, num_rows=144_233)
    import contextlib
    import io
    @contextlib.contextmanager
    def _mock_open(*args, **kwargs):
        yield io.BytesIO(b"TransactionID,TransactionDT,TransactionAmt,isFraud,DeviceInfo,id_01\n1,100,10.0,0,Windows,0.5\n")
    monkeypatch.setattr(ingestion_module, "_open_source", _mock_open)
    monkeypatch.setattr(ingestion_module, "_get_gcs_etag", lambda uri, proj: "mock_etag")
    mock_client.schema_from_json.return_value = [bigquery.SchemaField("TransactionID", "INTEGER")]

    result = [r for r in raw_identity_bigquery(
        build_asset_context(),
        identity_raw_csv_source=IdentityRawCsvSourceResource(
            uri="gs://fraud-bucket/train_identity.csv"
        ),
        bigquery_resource=BigQueryResource(project="test-project"),
    )][-1]

    mock_client.load_table_from_file.assert_not_called()
    mock_client.load_table_from_uri.assert_called_once()
    args, kwargs = mock_client.load_table_from_uri.call_args
    assert args[0] == "gs://fraud-bucket/train_identity.csv"
    assert args[1] == "test-project.raw.ieee_train_identity_raw"
    job_config = kwargs["job_config"]
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert job_config.autodetect is not True  # pinned schema, not autodetected

    assert getattr(result.metadata["rows_in_table"], "value", result.metadata["rows_in_table"]) == 144_233


# ---- INTEGER columns that the source file writes as floats ----------------------
#
# `C1` is `1.0` in train_transaction.csv, and BigQuery's CSV loader fails the whole job
# rather than parsing it into INT64. The load therefore widens those columns for the
# staging table and casts them back. These tests pin the two statements that does, and the
# guard that makes the cast trustworthy rather than merely convenient.

PINNED_SAMPLE = [
    bigquery.SchemaField("TransactionID", "INTEGER"),
    bigquery.SchemaField("TransactionAmt", "FLOAT"),
    bigquery.SchemaField("C1", "INTEGER"),
    bigquery.SchemaField("ProductCD", "STRING"),
    bigquery.SchemaField("M1", "BOOLEAN"),
]


def test_the_staging_schema_widens_only_the_integer_columns():
    widened = {f.name: f.field_type for f in raw_load.relaxed_schema(PINNED_SAMPLE)}

    assert widened["TransactionID"] == "FLOAT64"
    assert widened["C1"] == "FLOAT64"
    # Everything else survives untouched, or the staging table stops being a faithful copy
    # of the file and the cast is restoring the wrong thing.
    assert widened["TransactionAmt"] == "FLOAT"
    assert widened["ProductCD"] == "STRING"
    assert widened["M1"] == "BOOLEAN"


def test_the_cast_restores_exactly_the_pinned_integer_types():
    sql = raw_load.build_cast_sql("p.raw.t_staging", PINNED_SAMPLE)

    assert "CAST(`C1` AS INT64) AS `C1`" in sql
    assert "CAST(`TransactionID` AS INT64) AS `TransactionID`" in sql
    # A column the schema does not call INTEGER must pass through uncast.
    assert "`TransactionAmt`" in sql and "CAST(`TransactionAmt`" not in sql
    assert "FROM `p.raw.t_staging`" in sql


def test_every_pinned_column_survives_the_cast_projection():
    """A column dropped from the projection would vanish from the raw table silently."""
    sql = raw_load.build_cast_sql("p.raw.t_staging", PINNED_SAMPLE)

    for field in PINNED_SAMPLE:
        assert f"`{field.name}`" in sql


def test_the_integrality_guard_counts_fractional_values_per_column():
    """The check that catches a column the pinned schema is wrong about.

    D9 is the hour of the day as a fraction of one, so `CAST(... AS INT64)` collapses its
    24 values onto 0 and 1 — with no error raised anywhere. This is what makes that a
    failed run instead of a quietly damaged column.
    """
    sql = raw_load.build_integrality_sql(
        "p.raw.t_staging", raw_load.integer_columns(PINNED_SAMPLE)
    )

    assert "COUNTIF(`C1` IS NOT NULL AND `C1` != TRUNC(`C1`)) AS `C1`" in sql
    assert "TransactionID" in sql
    assert "TransactionAmt" not in sql  # not an INTEGER column, not this check's business
    assert sql.count("FROM") == 1  # one scan for every column, not one scan per column


def test_a_fractional_value_fails_the_run(monkeypatch):
    client = MagicMock()
    client.query.return_value.result.return_value = [{"C1": 0, "D9": 70_736}]

    with pytest.raises(Failure, match="D9"):
        ingestion_module._assert_integral(
            build_asset_context(), client, "p.raw.t_staging", ["C1", "D9"]
        )


def test_integral_columns_pass_the_guard():
    client = MagicMock()
    client.query.return_value.result.return_value = [{"C1": 0, "TransactionID": 0}]

    ingestion_module._assert_integral(
        build_asset_context(), client, "p.raw.t_staging", ["C1", "TransactionID"]
    )


def test_no_integer_columns_means_no_query_at_all():
    client = MagicMock()

    ingestion_module._assert_integral(build_asset_context(), client, "p.raw.t_staging", [])

    client.query.assert_not_called()


# ---- the skip-guard must notice a retype, 2026-08-16 ------------------------------

def test_a_retyped_column_is_not_current(monkeypatch):
    """Row count and source etag both unchanged; only the pinned type moved.

    This is the case that shipped silently: seven columns went FLOAT -> INTEGER in
    `schemas/bq_schema_train_transaction.json`, the guard compared rows and etag, found
    both unchanged, logged "already current" and skipped the load. The run was green and
    the table kept the old types.
    """
    from google.cloud import bigquery as bq

    from fraud_detection.orchestration.raw_load import schema_matches_pinned

    pinned = [
        bq.SchemaField("TransactionID", "INTEGER"),
        bq.SchemaField("addr1", "INTEGER"),
    ]

    class FakeTable:
        def __init__(self, schema):
            self.schema = schema

    stale = FakeTable([bq.SchemaField("TransactionID", "INTEGER"), bq.SchemaField("addr1", "FLOAT")])
    assert schema_matches_pinned(stale, pinned) is False

    current = FakeTable([bq.SchemaField("TransactionID", "INT64"), bq.SchemaField("addr1", "INT64")])
    # INT64 and INTEGER are the same type under two names; a guard that called this a
    # mismatch would reload the table on every single run.
    assert schema_matches_pinned(current, pinned) is True

    missing = FakeTable([bq.SchemaField("TransactionID", "INTEGER")])
    assert schema_matches_pinned(missing, pinned) is False
