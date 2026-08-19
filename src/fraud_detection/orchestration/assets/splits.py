from __future__ import annotations

from dagster import AssetKey, AutomationCondition, Failure, Output, asset

from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    MODEL_FACTORY,
)
from fraud_detection.orchestration.resources import BigQueryResource
from fraud_detection.schema import (
    FEATURES_DATASET,
    MODEL_INPUT_TABLE,
    SPLIT_TABLE,
    qualified,
)

# Percentiles are now loaded dynamically from config/feature-admission.toml, 
# so the split fractions are not hard-coded here.

# A narrow table (TransactionID, split, card_seen_in_train) rather than a
# second copy of the ~440-column model input. Downstream joins on
# TransactionID; nothing needs the wide table duplicated per split.
#
# The split is on the TIME AXIS, not random. In payments you predict the
# future from the past, so a K-fold shuffle would train on a card's later
# transactions and evaluate on its earlier ones -- leakage that no window
# frame can undo, because it happens above the feature layer.
# See docs/point-in-time.md, section 4.
#
# Boundaries use <=, so every row sharing a TransactionDT lands in the same
# split. That is the same peer-grouping logic the feature windows use: a
# timestamp is never cut in half by a split boundary.
SPLIT_SQL = """
CREATE OR REPLACE TABLE `{destination_table}` AS
WITH bounds AS (
  SELECT DISTINCT
    PERCENTILE_CONT(TransactionDT, {train_fraction}) OVER () AS train_end,
    PERCENTILE_CONT(TransactionDT, {val_start_fraction}) OVER () AS val_start,
    PERCENTILE_CONT(TransactionDT, {val_end_fraction}) OVER () AS val_end
  FROM `{source_table}`
),
assigned AS (
  SELECT
    s.TransactionID,
    s.{card_entity_column} AS card_entity,
    CASE
      WHEN s.TransactionDT <= b.train_end THEN 'train'
      WHEN s.TransactionDT <= b.val_end THEN 'val'
      ELSE 'test'
    END AS split
  FROM `{source_table}` AS s
  CROSS JOIN bounds AS b
  WHERE s.TransactionDT <= b.train_end 
     OR s.TransactionDT > b.val_start
),
train_cards AS (
  SELECT DISTINCT card_entity FROM assigned WHERE split = 'train'
)
SELECT
  a.TransactionID,
  a.split,
  -- Not a filter, a label. Returning cards are the production reality, so
  -- dropping them would evaluate the model on a population it will never
  -- meet. What this column buys is the ability to report metrics separately
  -- for cards the model has seen and cards it has not -- and to notice if a
  -- headline number is carried entirely by the former.
  a.card_entity IN (SELECT card_entity FROM train_cards) AS card_seen_in_train
FROM assigned AS a
"""


# `model_input` is depended on by key, not consumed as a value. It is the one edge that
# crosses from the feature platform into the model factory, and the two live in separate
# code locations -- so this asset cannot receive the upstream's return value, only the
# guarantee that it ran. The table name comes from the constant both sides share.
@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=BIGQUERY,
    owners=MODEL_FACTORY,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="dataset_preparation", 
    deps=[AssetKey(["fraud_detection", "model_input"])],
    description="Assigns every transaction to train/val/test on the time axis.",
)
def split_assignment(
    context,
    bigquery_resource: BigQueryResource,
):
    """Assigns every transaction to train/val/test on the time axis."""
    from fraud_detection.config import get_training_params
    from fraud_detection.orchestration.assets.feature_engineering import CARD_ENTITY_COLUMN

    client = bigquery_resource.get_client()
    source_table = qualified(bigquery_resource.project, FEATURES_DATASET, MODEL_INPUT_TABLE)
    destination_table = qualified(bigquery_resource.project, FEATURES_DATASET, SPLIT_TABLE)
    
    training_params = get_training_params()

    query = SPLIT_SQL.format(
        destination_table=destination_table,
        source_table=source_table,
        train_fraction=training_params.get("train_fraction"),
        val_start_fraction=training_params.get("val_start_fraction"),
        val_end_fraction=training_params.get("val_end_fraction"),
        card_entity_column=CARD_ENTITY_COLUMN,
    )

    query_job = client.query(query)
    try:
        query_job.result(timeout=1800)
    except Exception as exc:
        raise Failure(
            f"Split assignment failed writing {destination_table} from {source_table}: {exc}"
        ) from exc

    stats_query = f"""
    SELECT
      split,
      COUNT(*) AS rows_in_split,
      COUNTIF(card_seen_in_train) AS rows_with_card_seen_in_train
    FROM `{destination_table}`
    GROUP BY split
    """
    import polars as pl
    stats_df = pl.from_arrow(client.query(stats_query).result().to_arrow())
    stats = {
        row["split"]: {
            "rows": int(row["rows_in_split"]),
            "card_seen_in_train": int(row["rows_with_card_seen_in_train"]),
        }
        for row in stats_df.iter_rows(named=True)
    }

    for split, values in sorted(stats.items()):
        context.log.info(
            "Split %s: %s rows, %s of them on a card also present in train.",
            split,
            values["rows"],
            values["card_seen_in_train"],
        )

    return Output(
        destination_table,
        metadata={
            "table_id": destination_table,
            "train_rows": stats.get("train", {}).get("rows", 0),
            "val_rows": stats.get("val", {}).get("rows", 0),
            "test_rows": stats.get("test", {}).get("rows", 0),
            "train_card_overlap": float(stats["train"]["card_seen_in_train"] / stats["train"]["rows"]) if stats.get("train", {}).get("rows", 0) > 0 else 0.0,
            "val_card_overlap": float(stats["val"]["card_seen_in_train"] / stats["val"]["rows"]) if stats.get("val", {}).get("rows", 0) > 0 else 0.0,
            "test_card_overlap": float(stats["test"]["card_seen_in_train"] / stats["test"]["rows"]) if stats.get("test", {}).get("rows", 0) > 0 else 0.0,
            "test_rows_on_card_seen_in_train": stats.get("test", {}).get("card_seen_in_train", 0),
        },
    )
