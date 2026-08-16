from __future__ import annotations

from dagster import AssetKey, AutomationCondition, Failure, Output, asset

from fraud_detection.core.schema import (
    FEATURE_COLUMNS,
    FEATURE_TABLE,
    FEATURES_DATASET,
    JOINED_TABLE,
    MODEL_INPUT_TABLE,
    RAW_DATASET,
    qualified,
    uid_aggregate_feature_columns,
)
from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    FEATURE_PLATFORM,
)
from fraud_detection.orchestration.resources import BigQueryResource

# Assembled from two tables on purpose. A transaction arriving at /predict has
# never been seen before, so its own V*/id_* values cannot be retrieved from
# anywhere -- they are in the request. Only the card's and device's prior
# history can be looked up. Serving input is therefore
# "request fields + retrieved entity state", and training has to be assembled
# the same way or the two paths compute different things.
# See docs/feature-engineering.md.
#
# The engineered columns are named from FEATURE_COLUMNS rather than taken as
# f.*: the two tables share six pass-through columns (TransactionID,
# TransactionDT, TransactionAmt, card1, DeviceInfo, isFraud), and f.* would
# duplicate every one of them. Naming them also means a feature added upstream
# without updating FEATURE_COLUMNS fails loudly here instead of silently
# never reaching the model.
MODEL_INPUT_SQL = """
CREATE OR REPLACE TABLE `{destination_table}` AS
SELECT
  j.*,
  {feature_columns}
FROM `{joined_table}` AS j
JOIN `{feature_table}` AS f
  ON j.TransactionID = f.TransactionID
"""


def build_feature_column_list(columns: list[str] | None = None) -> str:
    """Every engineered column, named explicitly rather than star-selected.

    `None` means "whatever the pipeline currently produces": the velocity aggregates plus
    the uid aggregates the policy's derivations generate. Naming them explicitly is what
    makes a missing feature a query error instead of a silently narrower training matrix.
    """
    if columns is None:
        columns = [*FEATURE_COLUMNS, *uid_aggregate_feature_columns()]
    return ",\n  ".join(f"f.{column}" for column in columns)


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=BIGQUERY,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="feature_store",
    deps=[AssetKey(["fraud_detection", "joined_transactions_identity"]), AssetKey(["fraud_detection", "transaction_features"])],
    description="Joins the raw joined table to the engineered features, per transaction.",
)
def model_input(
    context,
    bigquery_resource: BigQueryResource,
):
    """Joins the raw joined table to the engineered features, per transaction.

    An INNER join is correct here and an outer one would not be: every
    transaction has exactly one row in each table, so a row missing on either
    side means an upstream asset produced something incomplete, and that
    should surface as a row-count mismatch rather than as nulls in the
    training matrix.
    """
    client = bigquery_resource.get_client()
    project = bigquery_resource.project
    joined_table = qualified(project, RAW_DATASET, JOINED_TABLE)
    feature_table = qualified(project, FEATURES_DATASET, FEATURE_TABLE)
    destination_table = qualified(project, FEATURES_DATASET, MODEL_INPUT_TABLE)

    query = MODEL_INPUT_SQL.format(
        destination_table=destination_table,
        joined_table=joined_table,
        feature_table=feature_table,
        feature_columns=build_feature_column_list(),
    )

    query_job = client.query(query)
    try:
        query_job.result(timeout=1800)
    except Exception as exc:
        raise Failure(
            f"Model input assembly failed writing {destination_table} from "
            f"{joined_table} and {feature_table}: {exc}"
        ) from exc

    table = client.get_table(destination_table)
    context.log.info(
        "Assembled %s rows and %s columns into %s.",
        table.num_rows,
        len(table.schema),
        destination_table,
    )
    return Output(
        destination_table,
        metadata={
            "table_id": table.full_table_id,
            "rows_in_table": table.num_rows,
            "columns_in_table": len(table.schema),
            "engineered_columns": len(FEATURE_COLUMNS),
        },
    )
