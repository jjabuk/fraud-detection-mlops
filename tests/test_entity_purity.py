"""Synthetic ground truth: we know who the real entities are, so we know the right answer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fraud_detection.evaluation.entity_purity import (
    Anchor,
    EntityKey,
    compare,
    coverage,
    entity_split,
    purity,
    seen_entity_flag,
)

DAY = 86_400
N_ENTITIES = 400
TXNS = 5


def make(seed: int = 0, missing_addr: float = 0.0) -> pl.DataFrame:
    """`truth` is the real entity. `card`+`addr`+`start_day` reconstruct it exactly.

    Fraud is a property of the entity, not of the transaction — as it is whenever a
    chargeback marks a customer's whole history.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(N_ENTITIES):
        start_day = int(rng.integers(0, 60))
        fraud = int(rng.random() < 0.2)
        card = e % 50  # cards are reused, so card alone is a coarse entity
        addr = e % 17
        for t in range(TXNS):
            day = start_day + 10 + t * 3
            rows.append({
                "truth": e, "card": card, "addr": addr,
                "TransactionDT": day * DAY + t, "D1": day - start_day,
                "isFraud": fraud,
            })

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.Series("txn_id", np.arange(len(df))))
    if missing_addr:
        df = df.with_columns(
            pl.when(pl.Series(rng.random(len(df))) < missing_addr)
            .then(None)
            .otherwise(pl.col("addr"))
            .alias("addr")
        )
    return df


CARD = EntityKey(columns=("card",), name="card")
CARD_ADDR = EntityKey(columns=("card", "addr"), name="card+addr")
FULL = EntityKey(columns=("card", "addr"), anchors=(Anchor("D1"),), name="card+addr+day-D1")


def test_the_full_key_recovers_the_true_entity():
    df = make()
    uid = FULL.assign(df)

    assert uid.n_unique() == df.get_column("truth").n_unique()
    assert df.select([pl.Series("uid", uid), pl.col("truth")]).group_by("uid").agg(pl.col("truth").n_unique()).get_column("truth").max() == 1


def test_a_correct_reconstruction_is_label_pure():
    df = make()
    assert purity(df, FULL.assign(df), "isFraud").pure_share_multi == 1.0


def test_a_coarse_entity_is_less_pure():
    df = make()
    coarse = purity(df, CARD.assign(df), "isFraud").pure_share_multi
    exact = purity(df, FULL.assign(df), "isFraud").pure_share_multi

    assert coarse < exact


def test_singletons_are_reported_separately_from_multi_transaction_entities():
    """The trap this class exists to close: singletons are pure by definition.

    A key so fine that every row is its own entity scores 100% over all entities and is
    worthless. `pure_share_multi` must not be fooled by it.
    """
    df = make()
    per_row = EntityKey(columns=("txn_id",), name="one entity per row")

    p = purity(df, per_row.assign(df), "isFraud")
    assert p.pure_share_all == 1.0
    assert p.singleton_share == 1.0
    assert np.isnan(p.pure_share_multi)


def test_a_missing_component_yields_no_id_rather_than_a_fake_one():
    df = make(missing_addr=0.3)
    uid = FULL.assign(df)

    assert uid.null_count() > 0
    df_filtered = df.with_columns(pl.Series("uid", uid)).filter(uid.is_not_null())
    assert df_filtered.group_by("uid").agg(pl.col("truth").n_unique()).get_column("truth").max() == 1


def test_purity_ignores_rows_without_an_id():
    df = make(missing_addr=0.3)
    p = purity(df, FULL.assign(df), "isFraud")

    assert p.pure_share_multi == 1.0
    assert p.rows_covered < 1.0


def test_coverage_counts_first_transactions_as_having_no_history():
    df = make()
    c = coverage(df, FULL.assign(df))

    assert c.share_without_uid == 0.0
    assert c.share_first_txn == pytest.approx(1 / TXNS, abs=0.01)
    assert c.share_with_history == pytest.approx(1 - 1 / TXNS, abs=0.01)


def test_coverage_reports_rows_with_no_id():
    df = make(missing_addr=0.3)
    c = coverage(df, FULL.assign(df))

    assert 0.2 < c.share_without_uid < 0.4
    assert c.share_without_uid + c.share_first_txn + c.share_with_history == pytest.approx(1.0)


def test_compare_ranks_the_true_reconstruction_last():
    df = make()
    table = compare(df, [CARD, CARD_ADDR, FULL], "isFraud")

    assert table.get_column("entity").to_list()[-1] == "card+addr+day-D1"
    assert table.get_column("pure_share_multi").is_sorted()


def test_entity_split_shares_no_entity_between_the_halves():
    df = make()
    uid = FULL.assign(df)
    train, holdout = entity_split(df, uid, frac=0.7)

    assert set(FULL.assign(train).drop_nulls().to_list()) & set(FULL.assign(holdout).drop_nulls().to_list()) == set()
    assert len(train) + len(holdout) == len(df)
    assert 0.6 < len(train) / len(df) < 0.8


def test_rows_without_an_id_land_in_the_holdout():
    df = make(missing_addr=0.3)
    uid = FULL.assign(df)
    _, holdout = entity_split(df, uid, frac=0.7)

    assert FULL.assign(holdout).null_count() == uid.null_count()


def test_entity_split_rejects_a_degenerate_fraction():
    with pytest.raises(ValueError, match="strictly between"):
        entity_split(make(), FULL.assign(make()), frac=1.0)


def test_seen_entity_flag_marks_unseen_entities():
    df = make()
    train, holdout = entity_split(df, FULL.assign(df), frac=0.7)

    flag = seen_entity_flag(train, holdout, FULL)
    assert set(flag.drop_nulls().to_list()) == {False}  # an entity split leaves nothing seen


def test_seen_entity_flag_on_a_time_split_finds_mostly_seen_entities():
    """Contrast: when entities are long-lived, a time split hands the model customers
    it has already met — which is why an entity split is a different question."""
    rows = [
        {"card": e, "addr": 0, "D1": day, "TransactionDT": day * DAY, "isFraud": 0}
        for e in range(50)
        for day in range(0, 100, 5)  # every entity spans the whole period
    ]
    df = pl.DataFrame(rows).sort("TransactionDT")
    train = df.head(len(df) // 2)
    holdout = df.tail(len(df) - len(df) // 2)

    flag = seen_entity_flag(train, holdout, EntityKey(columns=("card", "addr"))).drop_nulls()
    assert flag.mean() > 0.9  # a time split leaves almost every entity already seen
