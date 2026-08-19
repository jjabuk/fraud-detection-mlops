"""The entity the data does not name, as the pipeline uses it.

Anonymised transaction data rarely names the customer. This one is reconstructed
from the card, the address and the day the card's history starts -- and that
reconstruction happens **once, in SQL**, in `features.CLIENT_UID_EXPRESSION`. It is
carried into `features.model_input` as `client_uid` and read from there.

Nothing here rebuilds it. An identifier assembled in two places agrees only for as
long as nobody edits either, and the two places would be a BigQuery expression and
a polars one -- different null semantics, different string formatting, no test that
compares them.

What is left is what the pipeline actually does with the key: split so the halves
share no entity, and flag whether a scored row belongs to an entity training has
already seen. Both are set operations on a column.

Whether this is the *right* key is a question about the data, and it is answered in
``analysis/`` -- where the candidates are compared against a permuted null and the
chosen one is checked against this column.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fraud_detection.schema import CLIENT_ENTITY_COLUMN

__all__ = ["entity_ids", "entity_split", "seen_entity_flag"]


def entity_ids(df: pl.DataFrame, column: str = CLIENT_ENTITY_COLUMN) -> pl.Series:
    """The entity column, with a clear error when it is not there.

    A missing column here means the frame came from somewhere upstream of
    `model_input` -- and silently falling back to rebuilding the key would be the
    duplication this module exists to remove, so it raises instead.
    """
    if column not in df.columns:
        raise KeyError(
            f"{column!r} is not in the frame. It is produced by the feature-engineering "
            "statement and carried into features.model_input; a frame without it has not "
            "been through that step."
        )
    return df.get_column(column)


def entity_split(
    df: pl.DataFrame,
    uid: pl.Series | None = None,
    *,
    frac: float = 0.7,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split so the two halves share **no entity**.

    A time split does not do this: most entities appear on both sides of it, so a
    model scored that way is being asked about customers it has already met. This
    split asks the question that matters for a new customer -- and a fraudster is,
    by construction, always a new customer.

    Rows with no id go to the holdout: unknown at scoring time is exactly what they
    are.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be strictly between 0 and 1, got {frac}")
    if uid is None:
        uid = entity_ids(df)

    known = uid.is_not_null()
    # Sorted, and that is the whole guarantee. `unique()` is hash-based and returns
    # its result in an order that is not stable between calls, so permuting it under
    # a fixed seed still produced a different split every time -- the model was
    # trained on different rows on every run while the seed said otherwise.
    #
    # `maintain_order=True` would fix the symptom and inherit a different one: the
    # order would then depend on the frame's row order, which a multi-threaded scan
    # does not promise either. Sorting makes the split depend on the *set* of
    # entities and the seed, and on nothing else.
    entities = uid.filter(known).unique().sort().to_numpy()
    rng = np.random.default_rng(seed)
    train_entities = entities[rng.permutation(len(entities))[: int(len(entities) * frac)]]

    in_train = known & uid.is_in(pl.Series(train_entities).implode())
    return df.filter(in_train), df.filter(~in_train)


def seen_entity_flag(
    train: pl.DataFrame, holdout: pl.DataFrame, column: str = CLIENT_ENTITY_COLUMN
) -> pl.Series:
    """For each holdout row: was its entity present in train? Null when it has no id.

    Segment a metric by this rather than filtering on it. Dropping the unseen rows
    makes an aggregate score look better while hiding the population the model is
    worst on, which is the population a fraud model exists for.
    """
    seen = entity_ids(train, column).drop_nulls().unique().implode()
    # `is_in` already returns null for a null id rather than False, which is the
    # distinction the gate reads: "this entity is new" and "this row has no entity"
    # are different populations and the segment must not merge them.
    return entity_ids(holdout, column).is_in(seen).alias("seen_in_train")
