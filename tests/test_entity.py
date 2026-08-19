"""The entity key is a column now, so these test set operations, not construction.

What used to be here — that pasting card, address and an anchor together recovers a
customer — is not a Python question any more. The identifier is built once by the
feature-engineering statement and carried into `model_input`; whether it is the *right*
identifier is measured in `analysis/`, against a permuted null.
"""

from __future__ import annotations

import polars as pl
import pytest

from fraud_detection.features.entity import entity_ids, entity_split, seen_entity_flag


def frame(uids: list[str | None], **extra) -> pl.DataFrame:
    return pl.DataFrame({"client_uid": uids, "x": list(range(len(uids))), **extra})


def test_a_frame_without_the_column_raises_rather_than_rebuilding_the_key():
    # Falling back to reconstruction is exactly the duplication this module removed:
    # it would agree with the warehouse for as long as nobody edited either.
    with pytest.raises(KeyError, match="model_input"):
        entity_ids(pl.DataFrame({"card1": [1, 2]}))


def test_the_split_halves_share_no_entity():
    df = frame([str(i // 4) for i in range(200)])
    train, holdout = entity_split(df, frac=0.7)
    assert train.height + holdout.height == df.height
    assert not set(train["client_uid"]) & set(holdout["client_uid"])


def test_rows_with_no_entity_go_to_the_holdout():
    # Unknown at scoring time is exactly what they are, and putting them in training
    # would teach the model a customer that does not exist.
    df = frame(["a", "a", None, None, "b", "b"])
    train, holdout = entity_split(df, frac=0.5)
    assert train["client_uid"].null_count() == 0
    assert holdout["client_uid"].null_count() == 2


def test_the_split_is_deterministic_on_the_seed():
    # Enough entities that two seeds picking the same 70% is not a coincidence
    # waiting to happen: at thirty entities it is common enough to make the test
    # flaky, which is a worse failure than no test.
    df = frame([str(i // 3) for i in range(900)])
    a, _ = entity_split(df, seed=7)
    b, _ = entity_split(df, seed=7)
    c, _ = entity_split(df, seed=8)
    assert a["x"].to_list() == b["x"].to_list()
    assert a["x"].to_list() != c["x"].to_list()


def test_an_impossible_fraction_is_refused():
    with pytest.raises(ValueError, match="strictly between"):
        entity_split(frame(["a"]), frac=1.0)


def test_seen_flag_separates_new_entities_from_rows_that_have_none():
    # Three populations, and the gate reads them differently: an entity training saw,
    # one it did not, and a row with no entity at all. Collapsing the last two into
    # False would merge a cold customer with a missing address.
    train = frame(["a", "a", "b"])
    holdout = frame(["a", "c", None])
    assert seen_entity_flag(train, holdout).to_list() == [True, False, None]


def test_seen_flag_ignores_entities_that_are_null_in_train():
    train = frame(["a", None, None])
    holdout = frame(["a", None])
    assert seen_entity_flag(train, holdout).to_list() == [True, None]


def test_the_split_is_the_same_on_every_call():
    # `unique()` is hash-based and its order is not stable between calls, so a seeded
    # permutation of it still produced a different split each time. Nothing in a
    # report would show that: the metrics move by less than seed noise, and the
    # split silently is not the one the previous run measured.
    df = frame([str(i // 3) for i in range(600)])
    runs = [entity_split(df, seed=3)[0]["x"].to_list() for _ in range(5)]
    assert all(run == runs[0] for run in runs)


def test_the_split_does_not_depend_on_the_order_rows_arrived_in():
    # The same rows in a different sequence must give the same split, because a
    # filtered Arrow scan does not promise to return them in file order.
    df = frame([str(i // 3) for i in range(600)])
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=11)
    a, _ = entity_split(df, seed=3)
    b, _ = entity_split(shuffled, seed=3)
    assert sorted(a["x"].to_list()) == sorted(b["x"].to_list())
