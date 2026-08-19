"""Assembling the frame the test period is scored on.

Pure SQL construction: the union that lets a test row see its entity's earlier history, and
the projection that keeps both sides of that union on one schema. The assets in
`orchestration/assets/inference.py` execute it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class SchemaField(Protocol):
    """What this module needs of a BigQuery schema field, without importing the SDK.

    A structural type rather than `bigquery.SchemaField`: this module is on the pure side
    of the layering rule, so a notebook can import it without pulling in a cloud client.
    """

    name: str
    field_type: str


from fraud_detection.schema import ORIGIN_COLUMN, ORIGIN_TEST, ORIGIN_TRAIN

__all__ = ["SCORING_HISTORY_SQL", "align_to_training_schema", "build_scoring_history_sql"]

# The union that makes the test period scoreable.
#
# Every velocity aggregate is `RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`
# partitioned by card1 / DeviceInfo / client_uid. Run over the test tables alone -- which
# is what this pipeline did -- the first test transaction of a card sees an empty window
# and gets `card_txn_count_prior = 0` and a NULL `seconds_since_prev_txn_card`, even
# though 98.6% of test rows sit on a card with history in the training period. The model
# was fitted on rows where those columns carried that history. Feeding it rows where they
# do not is training-serving skew, and no amount of code sharing prevents it: the layering
# rule stops the *transformation* being reimplemented, it says nothing about the *state*
# the transformation is applied to.
#
# Point-in-time correctness is untouched, and that is the reason this is a fix rather than
# a trade. The frame is still strictly earlier than the current row, so a test row sees
# every training row and every earlier test row, and never its own future. That is also
# precisely what a production system does -- entity state accumulates over all history to
# date -- which makes this the same semantics as the entity-keyed snapshot in
# architecture.md §3.2, differing only in materialization.
SCORING_HISTORY_SQL = """
CREATE OR REPLACE TABLE `{destination_table}` AS
SELECT {train_columns}, '{origin_train}' AS {origin_column} FROM `{train_joined_table}`
UNION ALL
SELECT {test_columns}, '{origin_test}' AS {origin_column} FROM `{test_joined_table}`
"""


def build_scoring_history_sql(
    *,
    destination_table: str,
    train_joined_table: str,
    test_joined_table: str,
    train_columns: list[str],
    test_columns: list[str],
) -> str:
    """The union statement, with both sides projected onto the training column list.

    `UNION ALL` matches by position, so both sides are named explicitly and in the same
    order rather than star-selected. A column the test table does not carry is selected as
    a typed NULL -- caller's job to pass it that way -- because a missing column has to
    show up as a missing value, not as a silently shifted column list.
    """
    return SCORING_HISTORY_SQL.format(
        destination_table=destination_table,
        train_joined_table=train_joined_table,
        test_joined_table=test_joined_table,
        train_columns=",\n  ".join(train_columns),
        test_columns=",\n  ".join(test_columns),
        origin_column=ORIGIN_COLUMN,
        origin_train=ORIGIN_TRAIN,
        origin_test=ORIGIN_TEST,
    )


def align_to_training_schema(
    train_schema: Sequence[SchemaField], test_field_names: set[str]
) -> tuple[list[str], list[str]]:
    """Both SELECT lists for the union, derived from the training table's schema.

    The training table is the reference because it defines what the model was fitted on.
    Anything it has that the test table lacks arrives as a typed NULL; anything the test
    table has that training does not is dropped, because the model has no column for it.
    """
    train_columns = [f"`{field.name}`" for field in train_schema]
    test_columns = [
        f"`{field.name}`"
        if field.name in test_field_names
        else f"CAST(NULL AS {field.field_type}) AS `{field.name}`"
        for field in train_schema
    ]
    return train_columns, test_columns
