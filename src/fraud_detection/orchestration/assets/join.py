from __future__ import annotations

from dagster import AssetKey, AutomationCondition, Failure, Output, asset

from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    FEATURE_PLATFORM,
)
from fraud_detection.orchestration.resources import BigQueryResource
from fraud_detection.schema import (
    JOINED_TABLE,
    RAW_DATASET,
    RAW_IDENTITY_TABLE,
    RAW_TRANSACTION_TABLE,
    qualified,
)

# The anonymized block. Every one of these lives in the transaction table,
# whose schema is pinned in Terraform -- so if one ever goes missing, this
# query fails loudly instead of quietly computing a different feature.
V_COLUMNS = [f"V{i}" for i in range(1, 340)]

NULL_COUNT_COLUMN = "null_count_V_block"

# Identity is present for only a subset of transactions, so the join is LEFT
# and rows without a match must survive it.
#
# null_count_V_block is computed here, on the raw nulls, because the
# missing-value pattern across V1-V339 is itself signal on this dataset --
# it is what defines the natural groups within the block. Anything that
# imputes those nulls has to run after this column exists.
JOIN_SQL = """
CREATE OR REPLACE TABLE `{destination_table}` AS
SELECT
  t.*,
  i.* EXCEPT (TransactionID),
  {null_count_expression} AS {null_count_column}
FROM `{transaction_table}` AS t
LEFT JOIN `{identity_table}` AS i
  ON t.TransactionID = i.TransactionID
"""


def build_null_count_expression(columns: list[str] = V_COLUMNS) -> str:
    """Sum of null indicators across the V-block, as a SQL expression."""
    if not columns:
        return "0"
    return " + ".join(f"IF(t.{column} IS NULL, 1, 0)" for column in columns)


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="raw_ingestion",
    deps=[AssetKey(["fraud_detection", "raw_transactions_bigquery"]), AssetKey(["fraud_detection", "raw_identity_bigquery"])],
    description="Joins the raw transaction and identity tables into `raw.ieee_train_joined`.",
)
def joined_transactions_identity(
    context,
    bigquery_resource: BigQueryResource,
):
    """Joins the raw transaction and identity tables into `raw.ieee_train_joined`.

    Runs as a single BigQuery query job: the ~590k-row, ~430-column result
    never passes through this process, only the query text and the job
    metadata do.
    """
    client = bigquery_resource.get_client()
    project = bigquery_resource.project
    transaction_table = qualified(project, RAW_DATASET, RAW_TRANSACTION_TABLE)
    identity_table = qualified(project, RAW_DATASET, RAW_IDENTITY_TABLE)
    destination_table = qualified(project, RAW_DATASET, JOINED_TABLE)

    query = JOIN_SQL.format(
        destination_table=destination_table,
        transaction_table=transaction_table,
        identity_table=identity_table,
        null_count_expression=build_null_count_expression(),
        null_count_column=NULL_COUNT_COLUMN,
    )

    query_job = client.query(query)
    try:
        query_job.result(timeout=1800)
    except Exception as exc:
        raise Failure(
            f"Identity join failed writing {destination_table} from "
            f"{transaction_table} and {identity_table}: {exc}"
        ) from exc

    table = client.get_table(destination_table)
    context.log.info(
        "Joined %s rows into %s (%s columns).",
        table.num_rows,
        destination_table,
        len(table.schema),
    )
    return Output(
        destination_table,
        metadata={
            "table_id": table.full_table_id,
            "rows_in_table": table.num_rows,
            "columns_in_table": len(table.schema),
            "transaction_table": transaction_table,
            "identity_table": identity_table,
        },
    )
