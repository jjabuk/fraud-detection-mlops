from __future__ import annotations

import polars as pl
import pytest

from fraud_detection.schema import EXCLUDED_COLUMNS
from fraud_detection.training.data import (
    align_categories,
    feature_columns,
    prepare_features,
    to_lightgbm,
)


def _frame(**overrides) -> pl.DataFrame:
    base = {
        "TransactionID": [1, 2, 3],
        "TransactionDT": [100, 200, 300],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "isFraud": [0, 1, 0],
        "split": ["train", "train", "train"],
        "card_seen_in_train": [True, True, False],
        "ProductCD": ["W", "C", "W"],
        "V1": [1.0, 2.0, 3.0],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_identifiers_labels_and_split_metadata_are_never_features():
    columns = feature_columns(_frame())

    assert set(columns).isdisjoint(EXCLUDED_COLUMNS)
    assert set(columns) == {"TransactionAmt", "ProductCD", "V1"}


def test_transaction_dt_is_excluded():
    # Specific to a time-based split: every test value of TransactionDT lies
    # after every training value, so a split on it can only send the whole
    # test set down one branch. Time reaches the model through the velocity
    # features, which are relative to the transaction and therefore transfer.
    assert "TransactionDT" in EXCLUDED_COLUMNS
    assert "TransactionDT" not in feature_columns(_frame())


def test_strings_become_categoricals_and_floats_shrink():
    prepared = prepare_features(_frame())

    assert prepared.schema["ProductCD"] == pl.Categorical
    assert prepared.schema["V1"] == pl.Float32


def test_alignment_gives_validation_the_training_vocabulary():
    # The bug this prevents is silent: pandas assigns category codes per
    # frame, so a value present in training but absent from validation gets a
    # different integer in each. LightGBM then splits on the wrong branch and
    # still returns predictions -- they are just wrong.
    train = prepare_features(_frame(ProductCD=["W", "C", "H"]))
    val = prepare_features(_frame(ProductCD=["C", "H", "C"]))

    aligned = align_categories(val, train)

    # In Polars using Enum, the categories are explicitly the same
    train_cats = train.get_column("ProductCD").unique().to_list()
    assert list(aligned.get_column("ProductCD").dtype.categories) == train_cats
    # Polars Enum encodes values based on order of categories
    assert aligned.get_column("ProductCD").to_physical().to_list() == [
        train_cats.index(value) for value in ["C", "H", "C"]
    ]


def test_alignment_survives_a_categorical_that_has_missing_values():
    """Nearly every categorical in this dataset has nulls, and this raised on all of them.

    A null is a value an Enum column may hold, never one of its categories, but the
    vocabulary was built with `unique()` — which includes None — so `pl.Enum` raised
    "Enum categories must not contain null values" and no model could be fitted at all.
    `card4`, `card6`, `M4`, `DeviceInfo` and both email domains are all affected.
    """
    train = prepare_features(_frame(ProductCD=["W", None, "C"]))
    val = prepare_features(_frame(ProductCD=["C", None, "W"]))

    aligned = align_categories(val, train)

    assert None not in list(aligned.get_column("ProductCD").dtype.categories)
    # The missing value survives as missing rather than becoming a category of its own:
    # LightGBM routes it down its own branch.
    assert aligned.get_column("ProductCD").null_count() == 1
    assert aligned.get_column("ProductCD").drop_nulls().to_list() == ["C", "W"]


def test_alignment_puts_columns_in_training_order():
    train = prepare_features(_frame())
    shuffled = prepare_features(_frame())[["V1", "ProductCD", "TransactionAmt"]]

    assert list(align_categories(shuffled, train).columns) == list(train.columns)


def test_a_category_unseen_in_training_becomes_missing_rather_than_shifting_codes():
    # LightGBM treats an unknown categorical as missing, which is the honest
    # answer. What must never happen is the unseen value quietly taking the
    # code of some other category.
    train = prepare_features(_frame(ProductCD=["W", "C", "W"]))
    val = prepare_features(_frame(ProductCD=["W", "S", "C"]))

    aligned = align_categories(val, train)

    assert aligned.get_column("ProductCD").null_count() == 1
    assert set(aligned.get_column("ProductCD").drop_nulls()) <= set(train.get_column("ProductCD").unique().to_list())


def test_declared_categorical_columns_override_the_inferred_dtype():
    """The dataset author's contract wins over BigQuery autodetect.

    Vesta lists id_12-id_38 as categorical. Autodetect typed twelve of them as
    floats because they contain digits, so the model was imposing an
    arithmetic ordering on codes -- and id_14 is a timezone, where that
    ordering is meaningless rather than merely unhelpful.
    """
    frame = _frame(id_14=[1.0, 2.0, 3.0], id_19=[100.0, 200.0, 100.0])

    prepared = prepare_features(frame)

    assert prepared.schema["id_14"] == pl.Categorical
    assert prepared.schema["id_19"] == pl.Categorical


def test_numeric_columns_outside_the_declared_set_stay_numeric():
    # The override is a list, not a heuristic: V-columns are numeric by the
    # author's explicit instruction and must not be swept up.
    frame = _frame(V1=[1.0, 2.0, 3.0], id_13=[7.0, 8.0, 9.0])

    prepared = prepare_features(frame)

    assert prepared.schema["V1"] == pl.Float32
    assert prepared.schema["id_13"] == pl.Categorical


# ---- the contract decides what the model sees ----------------------------------


class _Contract:
    """Just enough contract for the projection under test."""

    def __init__(self, admitted):
        self._admitted = list(admitted)

    def training_features(self):
        return list(self._admitted)


def test_the_model_sees_the_admitted_columns_in_contract_order():
    from fraud_detection.training.data import split_with_contract

    frame = _frame()
    seen = pl.Series([True, True, False])

    split = split_with_contract(frame, _Contract(["V1", "ProductCD"]), seen_in_train=seen)

    # TransactionAmt is a feature the contract did not admit, so it must not reach the
    # model -- even though SplitFrame still carries the amounts for the cost curve.
    assert list(split.features.columns) == ["V1", "ProductCD"]
    assert split.amounts.to_list() == [10.0, 20.0, 30.0]
    assert split.labels.to_list() == [0, 1, 0]
    assert split.seen_in_train.to_list() == [True, True, False]


def test_an_admitted_column_missing_from_the_table_fails_loudly():
    # Quietly dropping it would train on a smaller set than the one whose metrics get
    # published, and the published number would describe a model that was never fitted.
    from fraud_detection.training.data import split_with_contract

    with pytest.raises(KeyError, match="drifted apart"):
        split_with_contract(
            _frame(),
            _Contract(["V1", "V999"]),
            seen_in_train=pl.Series([True, True, True]),
        )


def test_lightgbm_accepts_what_to_lightgbm_produces():
    """The crossing LightGBM cannot make on its own.

    LightGBM reads a polars frame through Arrow, where a categorical is a *dictionary*
    type it does not support: it aborts the process with "Unsupported Arrow type:
    dictionary" from libc++abi, so no caller can catch it and every fit died there. pandas
    `category` is the supported spelling, and LightGBM splits on the categories rather than
    their codes — the property `align_categories` exists to keep stable across splits.

    A real fit, not a dtype assertion: the failure was in LightGBM's C++, which a check on
    the Python types would have sailed straight past.
    """
    import lightgbm as lgb

    frame = pl.DataFrame(
        {
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "ProductCD": ["W", "C", None, "W"],
            "isFraud": [0, 1, 0, 1],
        }
    )
    features = prepare_features(frame)
    features = align_categories(features, features)

    converted = to_lightgbm(features)
    assert str(converted["ProductCD"].dtype) == "category"
    # The vocabulary and its order have to survive, or the codes mean something else.
    assert list(converted["ProductCD"].cat.categories) == list(
        features.get_column("ProductCD").dtype.categories
    )

    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 2},
        lgb.Dataset(converted, label=frame["isFraud"].to_numpy()),
        num_boost_round=2,
    )
    assert len(booster.predict(to_lightgbm(features))) == len(frame)


def test_a_boolean_with_missing_values_reaches_lightgbm_as_a_number():
    """numpy bool cannot hold missing, so a nullable boolean lands in pandas as `object`.

    LightGBM rejects that outright: "pandas dtypes must be int, float or bool". Five
    columns in this dataset are exactly that shape — `M5`, `M6`, `id_35`, `id_36`,
    `id_37` — each missing on a quarter to three quarters of rows, so the whole fit
    failed on data that looked perfectly ordinary in polars.
    """
    import lightgbm as lgb

    frame = pl.DataFrame(
        {
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "M5": [True, None, False, True],
            "isFraud": [0, 1, 0, 1],
        }
    )
    features = prepare_features(frame)
    assert features.schema["M5"] == pl.Boolean, "the case under test needs a real boolean"

    converted = to_lightgbm(features)

    assert str(converted["M5"].dtype) == "float32"
    assert converted["M5"].isna().sum() == 1  # missing stays missing, not False
    assert converted["M5"].dropna().tolist() == [1.0, 0.0, 1.0]

    lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 2},
        lgb.Dataset(converted, label=frame["isFraud"].to_numpy()),
        num_boost_round=2,
    )
