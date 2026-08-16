"""Rebuild an entity nobody labelled, then prove it is real.

Anonymized transaction data rarely names the customer. You can often reconstruct one by
combining columns — but a reconstruction that looks plausible and a reconstruction that is
correct are different things, and guessing wrong fragments real customers into unrelated
groups.

The test used here is **label purity**. When a label propagates within an entity — one
fraudulent card marks that card's later transactions too — a correct reconstruction
produces groups that are label-homogeneous, and a wrong one does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = [
    "Anchor",
    "Coverage",
    "EntityKey",
    "Purity",
    "compare",
    "coverage",
    "entity_split",
    "purity",
    "seen_entity_flag",
]

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Anchor:
    """A "days since something began" column, turned back into the day it began.

    ``D1`` counts days since the card started. Subtract it from the transaction's own day
    and you recover the start day, which is a constant for that card's whole history —
    and therefore usable as part of an identifier.
    """

    days_since: str
    time: str = "TransactionDT"
    unit_seconds: int = SECONDS_PER_DAY

    def assign(self, df: pl.DataFrame) -> pl.Series:
        day = (df.get_column(self.time) / self.unit_seconds).floor()
        return (day - df.get_column(self.days_since)).round()


@dataclass(frozen=True)
class EntityKey:
    """Which columns, combined, are believed to identify one entity.

    Any component missing means the entity is **unknown**, and the id is NA. Filling the
    gap with a sentinel instead looks harmless and is not: every row with a missing
    component collapses into one enormous fake entity.
    """

    columns: tuple[str, ...] = ()
    anchors: tuple[Anchor, ...] = ()
    name: str = ""

    def assign(self, df: pl.DataFrame) -> pl.Series:
        parts = [df.get_column(c) for c in self.columns] + [a.assign(df) for a in self.anchors]
        if not parts:
            raise ValueError("an EntityKey needs at least one column or anchor")

        # String conversion and concatenation using expressions
        # Because we need a Series out, we construct a temp df
        temp_df = pl.DataFrame(parts)
        exprs = [pl.col(p.name).cast(pl.String) for p in parts]
        known = pl.all_horizontal([pl.col(p.name).is_not_null() for p in parts])
        
        uid_expr = pl.when(known).then(pl.concat_str(exprs, separator="|")).otherwise(None)
        return temp_df.select(uid_expr.alias("uid")).get_column("uid")

    @property
    def label(self) -> str:
        return self.name or "+".join([*self.columns, *(f"day-{a.days_since}" for a in self.anchors)])


@dataclass
class Purity:
    """How label-homogeneous the reconstructed groups are.

    ``pure_share_multi`` is the number that means something. Singleton entities are pure
    by definition — one transaction cannot disagree with itself — so a figure computed
    over all entities mostly measures how much the reconstruction fragmented, and rises
    as the reconstruction gets *worse*. Both are reported so the two can never be quietly
    swapped for each other.
    """

    entity: str
    entities: int
    entities_multi: int
    singleton_share: float
    pure_share_multi: float
    pure_share_all: float
    fraud_share_in_touched_groups: float
    txns_per_entity: float
    rows_covered: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Coverage:
    """What a serving path would actually have at prediction time."""

    entity: str
    rows: int
    share_without_uid: float
    share_first_txn: float
    share_with_history: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def purity(df: pl.DataFrame, uid: pl.Series, label: str) -> Purity:
    """Label homogeneity of the groups ``uid`` defines. Rows with no id are excluded."""
    known = uid.is_not_null()
    
    g = df.select([pl.Series("uid", uid), pl.col(label).alias("y")]).filter(known).group_by("uid").agg([
        pl.len().alias("size"),
        pl.col("y").mean().alias("mean")
    ])
    
    size = g.get_column("size")
    mean = g.get_column("mean")
    
    multi = size >= 2
    is_pure = (mean == 0.0) | (mean == 1.0)
    touched = mean > 0.0

    entities = len(size)
    
    return Purity(
        entity="",
        entities=entities,
        entities_multi=int(multi.sum()),
        singleton_share=round(float((~multi).mean()), 4) if entities > 0 else 0.0,
        pure_share_multi=round(float(is_pure.filter(multi).mean()), 4) if multi.any() else float("nan"),
        pure_share_all=round(float(is_pure.mean()), 4) if entities > 0 else 0.0,
        fraud_share_in_touched_groups=round(float(mean.filter(touched).mean()), 4) if touched.any() else 0.0,
        txns_per_entity=round(float(size.mean()), 2) if entities > 0 else 0.0,
        rows_covered=round(float(known.mean()), 4) if len(uid) > 0 else 0.0,
    )


def coverage(df: pl.DataFrame, uid: pl.Series, time_col: str = "TransactionDT") -> Coverage:
    """How often an entity aggregate would have nothing to aggregate.

    A row is unusable for entity history in two ways: no id at all, or an id whose first
    transaction this is. Both are ordinary at serving time and both must be null-safe
    downstream.
    """
    known = uid.is_not_null()
    
    first = df.with_columns(pl.Series("uid", uid)).with_columns(
        pl.when(pl.col("uid").is_not_null())
        .then(pl.col(time_col).rank("ordinal").over("uid") == 1)
        .otherwise(False)
        .alias("is_first")
    ).get_column("is_first")

    return Coverage(
        entity="",
        rows=len(df),
        share_without_uid=round(float((~known).mean()), 4),
        share_first_txn=round(float(first.mean()), 4),
        share_with_history=round(float((known & ~first).mean()), 4),
    )


def compare(df: pl.DataFrame, keys: Sequence[EntityKey], label: str, time_col: str = "TransactionDT") -> pl.DataFrame:
    """Score several candidate reconstructions side by side, best purity last.

    Read it as a ladder: a reconstruction that is genuinely tighter should raise
    ``pure_share_multi``. One that merely fragments raises ``pure_share_all`` and
    ``singleton_share`` while leaving ``pure_share_multi`` flat or lower.
    """
    rows = []
    for key in keys:
        uid = key.assign(df)
        p, c = purity(df, uid, label), coverage(df, uid, time_col)
        rows.append({"entity": key.label, **{k: v for k, v in p.as_dict().items() if k != "entity"},
                     "share_without_uid": c.share_without_uid, "share_first_txn": c.share_first_txn})
    return pl.DataFrame(rows).sort("pure_share_multi")


def entity_split(
    df: pl.DataFrame, uid: pl.Series, *, frac: float = 0.7, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split so the two halves share **no entity**.

    A time split does not do this: most entities appear on both sides of it, so a model
    scored that way is being asked about customers it has already met. This split asks the
    question that matters for a new customer — and a fraudster is, by construction, always
    a new customer.

    Rows with no id go to the holdout: unknown at scoring time is exactly what they are.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be strictly between 0 and 1, got {frac}")

    known = uid.is_not_null()
    entities = uid.filter(known).unique().to_numpy()
    rng = np.random.default_rng(seed)
    train_entities = set(entities[rng.permutation(len(entities))[: int(len(entities) * frac)]])

    in_train = known & uid.is_in(list(train_entities))
    return df.filter(in_train), df.filter(~in_train)


def seen_entity_flag(train: pl.DataFrame, holdout: pl.DataFrame, key: EntityKey) -> pl.Series:
    """For each holdout row: was its entity present in train? NA when it has no id.

    Segment a metric by this rather than filtering on it. Dropping the unseen rows makes
    an aggregate score look better while hiding the population the model is worst on.
    """
    seen = set(key.assign(train).drop_nulls().to_list())
    uid = key.assign(holdout)
    return pl.Series([v in seen if v is not None else None for v in uid.to_list()])
