from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from dagster import AssetCheckResult, AssetCheckSpec, AssetKey, Failure, Output, asset
from google.cloud import bigquery, storage

from fraud_detection.core.schema import (
    RAW_DATASET,
    RAW_IDENTITY_TABLE,
    RAW_TEST_IDENTITY_TABLE,
    RAW_TEST_TRANSACTION_TABLE,
    RAW_TRANSACTION_TABLE,
)
from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    FEATURE_PLATFORM,
)
from fraud_detection.orchestration.raw_load import (
    STAGING_SUFFIX,
    build_cast_sql,
    build_integrality_sql,
    integer_columns,
    relaxed_schema,
    schema_matches_pinned,
)
from fraud_detection.orchestration.resources import (
    BQ_SCHEMA_PATH_IDENTITY,
    BQ_SCHEMA_PATH_TRANSACTIONS,
    TEST_DUMP_GCS_URI,
    TEST_IDENTITY_DUMP_GCS_URI,
    BigQueryResource,
    IdentityRawCsvSourceResource,
    RawCsvSourceResource,
)

REQUIRED_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt"]
FRAUD_LABEL_COLUMN = "isFraud"

REQUIRED_IDENTITY_COLUMNS = ["TransactionID", "DeviceInfo"]
VALIDATION_CHUNK_SIZE = 100_000

def _assert_integral(context, client, staging_table: str, columns: list[str]) -> None:
    """Fails the run if casting any INTEGER-typed column would round a value away."""
    if not columns:
        return

    rows = list(client.query(build_integrality_sql(staging_table, columns)).result())
    if not rows:
        return

    offenders = {name: count for name, count in dict(rows[0]).items() if count}
    if offenders:
        raise Failure(
            "The pinned schema types these columns INTEGER, but the source holds "
            "fractional values in them, so loading them as INT64 would silently round: "
            + ", ".join(f"{name} ({count:,} rows)" for name, count in sorted(offenders.items()))
            + ". Either the schema is wrong about the column or the source has changed."
        )

    context.log.info(
        "%s INTEGER-typed columns checked; every value is integral, so the cast is exact.",
        len(columns),
    )


def _load_via_staging(context, client, *, source_uri: str, table_id: str, pinned):
    """Loads the CSV through a widened staging table, then casts into the pinned types.

    Returns the completed query job so the caller's `.result()` is a no-op on a job that
    has already finished.
    """
    staging_table = f"{table_id}{STAGING_SUFFIX}"

    context.log.info(
        "Loading %s into %s with INTEGER columns widened to FLOAT64.", source_uri, staging_table
    )
    load_job = client.load_table_from_uri(
        source_uri,
        staging_table,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            schema=relaxed_schema(pinned),
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    load_job.result(timeout=1800)

    try:
        _assert_integral(context, client, staging_table, integer_columns(pinned))

        context.log.info("Casting %s into %s under the pinned schema.", staging_table, table_id)
        query_job = client.query(
            build_cast_sql(staging_table, pinned),
            job_config=bigquery.QueryJobConfig(
                destination=table_id,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            ),
        )
        query_job.result(timeout=1800)
    finally:
        # The staging copy is the same 600 MB again; leaving it behind doubles the storage
        # bill for the largest table in the project, and it is regenerable by definition.
        client.delete_table(staging_table, not_found_ok=True)

    return query_job

# Names live in schema.py, which both layers agree on -- the join now derives them
# from there rather than receiving them as values from these assets.
RAW_TABLE = RAW_TRANSACTION_TABLE


def _open_source(uri: str, project: str):
    if uri.startswith("gs://"):
        bucket_name, _, blob_path = uri.removeprefix("gs://").partition("/")
        return storage.Client(project=project).bucket(bucket_name).blob(blob_path).open("rb")
    return Path(uri).open("rb")


def _get_gcs_etag(uri: str, project: str) -> str | None:
    """Returns the GCS object's etag (checksum) if the URI is a GCS path."""
    if not uri.startswith("gs://"):
        return None
    bucket_name, _, blob_path = uri.removeprefix("gs://").partition("/")
    blob = storage.Client(project=project).bucket(bucket_name).blob(blob_path)
    blob.reload()  # Fetch metadata
    return blob.etag


def _table_is_current(
    client, table_id: str, expected_rows: int, source_etag: str | None, pinned=None
) -> bool:
    """Checks the table exists, has the expected rows, matches the source, and is typed
    as the pinned schema says it should be."""
    try:
        table = client.get_table(table_id)
    except Exception:  # noqa: BLE001
        return False  # Table doesn't exist

    # Check row count
    if table.num_rows != expected_rows:
        return False

    # Check the pinned types. `pinned=None` means the caller has no pinned schema for this
    # table, not "any schema will do".
    if pinned is not None and not schema_matches_pinned(table, pinned):
        return False

    # Check source etag (stored in table labels, sanitized) - only for BigQuery
    if source_etag and hasattr(table, 'labels'):
        sanitized_source_etag = source_etag.replace('"', '').replace('/', '_').replace('+', '_').replace('=', '_').lower()[:63]
        table_etag = table.labels.get("source_etag") if table.labels else None
        if table_etag != sanitized_source_etag:
            return False
    
    return True


@asset(
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="raw_ingestion",
    deps=[AssetKey(["fraud_detection", "raw_transaction_kaggle_to_gcs"])],
    check_specs=[AssetCheckSpec(name="schema_valid", asset=AssetKey(["fraud_detection", "raw_transactions_bigquery"]))],
    description="Validates and loads the raw CSV into BigQuery.",
)
def raw_transactions_bigquery(
    context,
    raw_csv_source: RawCsvSourceResource,
    bigquery_resource: BigQueryResource,
) -> Any:
    """Validates and loads the raw CSV into BigQuery."""
    # 1. Validation Phase
    cols = [*REQUIRED_COLUMNS, FRAUD_LABEL_COLUMN]
    total_rows = 0
    missing: set[str] = set(cols)
    bad_label_values: set[Any] = set()

    with _open_source(raw_csv_source.uri, raw_csv_source.project) as source_file:
        for chunk in pl.scan_csv(source_file).collect_batches(chunk_size=VALIDATION_CHUNK_SIZE):
            missing -= set(chunk.columns)
            total_rows += len(chunk)
            if FRAUD_LABEL_COLUMN in chunk.columns:
                labels = chunk.get_column(FRAUD_LABEL_COLUMN).drop_nulls().unique().to_list()
                bad_label_values |= set(labels) - {0, 1}

    if missing or bad_label_values:
        yield AssetCheckResult(
            passed=False,
            description="Missing columns or bad label values",
            metadata={
                "missing_columns": sorted(missing),
                "bad_label_values": sorted(map(str, bad_label_values)),
            },
        )
        raise Failure("Validation failed, aborting load.")

    yield AssetCheckResult(passed=True, metadata={"rows_validated": total_rows})

    # 2. Check if table is already current
    client = bigquery_resource.get_client()
    table_id = f"{bigquery_resource.project}.raw.{RAW_TABLE}"
    
    source_etag = _get_gcs_etag(raw_csv_source.uri, raw_csv_source.project) if raw_csv_source.is_gcs else None

    if _table_is_current(
        client, table_id, total_rows, source_etag,
        pinned=client.schema_from_json(str(BQ_SCHEMA_PATH_TRANSACTIONS)),
    ):
        table = client.get_table(table_id)
        context.log.info(
            "Table %s is already current (%s rows, etag=%s, types match the pinned "
            "schema). Skipping load.",
            table_id, table.num_rows, source_etag or "N/A"
        )
        yield Output(
            table.full_table_id,
            metadata={
                "rows_in_table": table.num_rows,
                "status": "skipped_already_current",
                "source_etag": source_etag or "N/A",
            },
        )
        return

    # 3. Load Phase (only if table is stale or missing)

    if raw_csv_source.is_gcs:
        if not BQ_SCHEMA_PATH_TRANSACTIONS.exists():
            raise Failure("Pinned BigQuery schema not found.")
        load_job = _load_via_staging(
            context,
            client,
            source_uri=raw_csv_source.uri,
            table_id=table_id,
            pinned=client.schema_from_json(str(BQ_SCHEMA_PATH_TRANSACTIONS)),
        )
    else:
        source_path = Path(raw_csv_source.uri)
        if not source_path.exists():
            raise Failure(f"Local CSV not found at {source_path}.")
        job_config = bigquery.LoadJobConfig(
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with source_path.open("rb") as source_file:
            load_job = client.load_table_from_file(source_file, table_id, job_config=job_config)

    load_job.result(timeout=1800)
    table = client.get_table(table_id)
    
    # Store source etag in table labels for future idempotency checks (BigQuery only)
    if source_etag and hasattr(table, 'labels'):
        table.labels = table.labels or {}
        # BigQuery labels must match [a-z0-9_-]+ so we sanitize the etag
        table.labels["source_etag"] = source_etag.replace('"', '').replace('/', '_').replace('+', '_').replace('=', '_').lower()[:63]
        client.update_table(table, ["labels"])

    context.log.info("Loaded %s rows into %s (etag=%s).", table.num_rows, table_id, source_etag or "N/A")
    yield Output(
        table.full_table_id,
        metadata={
            "rows_in_table": table.num_rows,
            "load_path": (
                "staging + cast to the pinned schema" if raw_csv_source.is_gcs else "autodetect"
            ),
            "source_etag": source_etag or "N/A",
            "status": "loaded",
        },
    )


@asset(
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="raw_ingestion",
    deps=[AssetKey(["fraud_detection", "raw_identity_kaggle_to_gcs"])],
    check_specs=[AssetCheckSpec(name="schema_valid", asset=AssetKey(["fraud_detection", "raw_identity_bigquery"]))],
    description="Validates and loads the raw identity CSV into BigQuery.",
)
def raw_identity_bigquery(
    context,
    identity_raw_csv_source: IdentityRawCsvSourceResource,
    bigquery_resource: BigQueryResource,
) -> Any:
    """Validates and loads the raw identity CSV into BigQuery."""
    # 1. Validation Phase
    cols = REQUIRED_IDENTITY_COLUMNS
    total_rows = 0
    missing: set[str] = set(cols)

    try:
        source = _open_source(identity_raw_csv_source.uri, identity_raw_csv_source.project)
    except FileNotFoundError as exc:
        raise Failure(f"Identity CSV not found at {identity_raw_csv_source.uri}.") from exc

    with source as source_file:
        for chunk in pl.scan_csv(source_file).collect_batches(chunk_size=VALIDATION_CHUNK_SIZE):
            missing -= set(chunk.columns)
            total_rows += len(chunk)

    if missing:
        yield AssetCheckResult(
            passed=False,
            description="Missing columns",
            metadata={"missing_columns": sorted(missing)},
        )
        raise Failure("Validation failed, aborting load.")

    yield AssetCheckResult(passed=True, metadata={"rows_validated": total_rows})

    # 2. Check if table is already current
    client = bigquery_resource.get_client()
    table_id = f"{bigquery_resource.project}.raw.{RAW_IDENTITY_TABLE}"
    
    source_etag = _get_gcs_etag(identity_raw_csv_source.uri, identity_raw_csv_source.project) if identity_raw_csv_source.is_gcs else None

    if _table_is_current(
        client, table_id, total_rows, source_etag,
        pinned=client.schema_from_json(str(BQ_SCHEMA_PATH_IDENTITY)),
    ):
        table = client.get_table(table_id)
        context.log.info(
            "Table %s is already current (%s rows, etag=%s, types match the pinned "
            "schema). Skipping load.",
            table_id, table.num_rows, source_etag or "N/A"
        )
        yield Output(
            table.full_table_id,
            metadata={
                "rows_in_table": table.num_rows,
                "status": "skipped_already_current",
                "source_etag": source_etag or "N/A",
            },
        )
        return

    # 3. Load Phase (only if table is stale or missing)

    if identity_raw_csv_source.is_gcs:
        if not BQ_SCHEMA_PATH_IDENTITY.exists():
            raise Failure(f"Pinned BigQuery schema not found at {BQ_SCHEMA_PATH_IDENTITY}.")
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            schema=client.schema_from_json(str(BQ_SCHEMA_PATH_IDENTITY)),
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        load_job = client.load_table_from_uri(
            identity_raw_csv_source.uri, table_id, job_config=job_config
        )
    else:
        source_path = Path(identity_raw_csv_source.uri)
        if not source_path.exists():
            raise Failure(f"Local identity CSV not found at {source_path}.")
        job_config = bigquery.LoadJobConfig(
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with source_path.open("rb") as source_file:
            load_job = client.load_table_from_file(source_file, table_id, job_config=job_config)

    load_job.result(timeout=1800)
    table = client.get_table(table_id)
    
    # Store source etag in table labels for future idempotency checks (BigQuery only)
    if source_etag and hasattr(table, 'labels'):
        table.labels = table.labels or {}
        table.labels["source_etag"] = source_etag.replace('"', '').replace('/', '_').replace('+', '_').replace('=', '_').lower()[:63]
        client.update_table(table, ["labels"])

    context.log.info("Loaded %s rows into %s (etag=%s).", table.num_rows, table_id, source_etag or "N/A")
    yield Output(
        table.full_table_id,
        metadata={
            "rows_in_table": table.num_rows,
            "source_etag": source_etag or "N/A",
            "status": "loaded",
        },
    )


# ---- the Kaggle test period -----------------------------------------------------
#
# Scored, never trained on. Two things make loading it different from the training dump,
# and both are properties of the files Kaggle ships rather than choices made here.
#
# `test_transaction.csv` has no `isFraud` column -- the labels are what the competition
# withholds -- so the pinned training schema does not describe it. Every other column is
# present, in the same order.
#
# `test_identity.csv` names its columns `id-01`, with a hyphen, where the training file
# writes `id_01`. Loading it under its own header would produce 41 columns whose names
# match nothing in the training table, and `scoring_history`'s union would fill all of
# them with NULL -- a model scored on empty identity data, with no error anywhere.
#
# Both are handled by the same mechanism and it needs no renaming code: a CSV load with an
# explicit schema skips the header and assigns names **by position**. Hand it the training
# schema and the columns come back named the way training named them. The column orders are
# asserted below rather than trusted, because that is the assumption the whole thing rests
# on.


def schema_for_test_transaction(pinned: list[bigquery.SchemaField]) -> list[bigquery.SchemaField]:
    """The training schema with the label removed, which is what the test file contains."""
    return [field for field in pinned if field.name != FRAUD_LABEL_COLUMN]


def _assert_positional_match(context, source_uri: str, project: str, schema, label: str) -> None:
    """The header we are about to discard has to line up with the schema replacing it.

    Positional loading is what renames `id-01` to `id_01` for free, and it is also what
    would silently shift every column by one if Kaggle ever inserted a field. Comparing the
    header once, at load time, is the difference between the two.
    """
    with _open_source(source_uri, project) as handle:
        header = handle.readline().decode("utf-8-sig").strip().split(",")

    expected = [field.name for field in schema]
    if len(header) != len(expected):
        raise Failure(
            f"{label}: the file has {len(header)} columns and the schema names "
            f"{len(expected)}. Loading by position would shift every column after the "
            "first difference."
        )

    normalised = [name.replace("-", "_") for name in header]
    if normalised != expected:
        mismatched = [
            f"position {i}: file {have!r} vs schema {want!r}"
            for i, (have, want) in enumerate(zip(normalised, expected))
            if have != want
        ]
        raise Failure(f"{label}: header and schema disagree — {'; '.join(mismatched[:5])}")

    context.log.info("%s: %s columns line up with the pinned schema by position.", label, len(header))


@asset(
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="raw_ingestion",
    description="Loads the Kaggle test transactions, typed by the training schema minus isFraud.",
)
def raw_test_transaction_bigquery(context, bigquery_resource: BigQueryResource) -> Output:
    """Loads the Kaggle test transactions, typed by the training schema minus isFraud."""
    if not BQ_SCHEMA_PATH_TRANSACTIONS.exists():
        raise Failure("Pinned BigQuery schema not found.")

    client = bigquery_resource.get_client()
    table_id = f"{bigquery_resource.project}.{RAW_DATASET}.{RAW_TEST_TRANSACTION_TABLE}"
    schema = schema_for_test_transaction(
        client.schema_from_json(str(BQ_SCHEMA_PATH_TRANSACTIONS))
    )

    _assert_positional_match(
        context, TEST_DUMP_GCS_URI, bigquery_resource.project, schema, "test_transaction"
    )
    _load_via_staging(
        context, client, source_uri=TEST_DUMP_GCS_URI, table_id=table_id, pinned=schema
    )

    table = client.get_table(table_id)
    context.log.info("Loaded %s test transactions into %s.", table.num_rows, table_id)
    return Output(table.full_table_id, metadata={"rows_in_table": table.num_rows})


@asset(
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="raw_ingestion",
    description="Loads the Kaggle test identity rows, renamed to the training spelling.",
)
def raw_test_identity_bigquery(context, bigquery_resource: BigQueryResource) -> Output:
    """Loads the Kaggle test identity rows, renamed to the training spelling."""
    if not BQ_SCHEMA_PATH_IDENTITY.exists():
        raise Failure(f"Pinned BigQuery schema not found at {BQ_SCHEMA_PATH_IDENTITY}.")

    client = bigquery_resource.get_client()
    table_id = f"{bigquery_resource.project}.{RAW_DATASET}.{RAW_TEST_IDENTITY_TABLE}"
    schema = client.schema_from_json(str(BQ_SCHEMA_PATH_IDENTITY))

    _assert_positional_match(
        context, TEST_IDENTITY_DUMP_GCS_URI, bigquery_resource.project, schema, "test_identity"
    )
    _load_via_staging(
        context, client, source_uri=TEST_IDENTITY_DUMP_GCS_URI, table_id=table_id, pinned=schema
    )

    table = client.get_table(table_id)
    context.log.info("Loaded %s test identity rows into %s.", table.num_rows, table_id)
    return Output(table.full_table_id, metadata={"rows_in_table": table.num_rows})
