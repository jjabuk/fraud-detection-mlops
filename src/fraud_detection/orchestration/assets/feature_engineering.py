from __future__ import annotations

from dagster import AssetKey, AutomationCondition, Failure, Output, asset

from fraud_detection.core.feature_contract import load_admission_rules
from fraud_detection.core.schema import (
    CARD_ENTITY_COLUMN,
    CLIENT_ENTITY_COLUMN,
    DEVICE_ENTITY_COLUMN,
    FEATURE_TABLE,
    FEATURES_DATASET,
    JOINED_TABLE,
    RAW_DATASET,
    qualified,
)
from fraud_detection.feature_engineering.features import (
    WINDOW_1H_SECONDS,
    WINDOW_24H_SECONDS,
    build_sql,
)
from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    FEATURE_PLATFORM,
)
from fraud_detection.orchestration.resources import BigQueryResource


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="feature_store",
    deps=[AssetKey(["fraud_detection", "joined_transactions_identity"])],
    description="Builds point-in-time card-velocity features from the raw table and writes them to the Feature Store (`features` dataset in BigQuery).",
)
def transaction_features(
    context,
    bigquery_resource: BigQueryResource,
):
    """Builds point-in-time card-velocity features from the raw table and
    writes them to the Feature Store (`features` dataset in BigQuery).

    Runs entirely as a single BigQuery query job (CREATE OR REPLACE TABLE
    AS SELECT) -- like raw_transaction_bq_schema, this never pulls the
    (~590k-row, 394-column) raw table's bytes through this process; only
    the query text and the resulting job metadata do.

    Every aggregate is computed with a RANGE window frame bounded at
    "1 PRECEDING" on TransactionDT (see FEATURE_ENGINEERING_SQL), so each
    row's features are built only from transactions strictly earlier in
    time for the same card -- never from itself or from the future. This
    is what makes the resulting table safe to use for both training and
    point-in-time serving without leakage.
    """
    rules = load_admission_rules()
    client = bigquery_resource.get_client()
    project = bigquery_resource.project
    # By key, not by value. Taking the upstream's return value means the run depends on the
    # orchestrator's local storage as well as on the data.
    source_table = qualified(project, RAW_DATASET, JOINED_TABLE)
    destination_table = qualified(project, FEATURES_DATASET, FEATURE_TABLE)

    
    query = build_sql(
        source_table=f"`{source_table}`",
        destination_table=destination_table,
        card_entity_column=CARD_ENTITY_COLUMN,
        device_entity_column=DEVICE_ENTITY_COLUMN,
        client_entity_column=CLIENT_ENTITY_COLUMN,
        window_1h=WINDOW_1H_SECONDS,
        window_24h=WINDOW_24H_SECONDS,
        # The published solutions' aggregation family, under this project's point-in-time
        # rule. Provenance: ATTRIBUTION.md.
        # The derivations come from config/feature-admission.toml, so adding a D column to
        # aggregate is a config diff.
        derivations=[d for d in rules.derivations if d.name in set(rules.uid_std_of_derived)],
        c_columns=rules.uid_c_columns,
        m_columns=rules.uid_m_columns,
    )

    query_job = client.query(query)
    try:
        query_job.result(timeout=1800)
    except Exception as exc:
        raise Failure(
            f"Feature engineering query failed writing {destination_table} "
            f"from {source_table}: {exc}"
        ) from exc

    table = client.get_table(destination_table)
    context.log.info(
        "Materialized %s rows of point-in-time features into %s.",
        table.num_rows,
        destination_table,
    )
    return Output(
        destination_table,
        metadata={
            "table_id": table.full_table_id,
            "rows_in_table": table.num_rows,
            "source_table": source_table,
        },
    )
