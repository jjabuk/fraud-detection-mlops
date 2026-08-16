"""Synthetic ground truth: each column knows what shift it is supposed to show."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fraud_detection.evaluation.distribution_shift import (
    NULL_BUCKET,
    OTHER_BUCKET,
    Reference,
    adversarial_auc,
    adversarial_per_feature,
)

N = 5_000


def reference_frame(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "stable": rng.normal(size=N),
            "shifted": rng.normal(size=N),
            "nulls": [None if r < 0.2 else v for r, v in zip(rng.random(N), rng.normal(size=N))],
            "cat": rng.choice(["a", "b", "c"], size=N, p=[0.6, 0.3, 0.1]),
            "constant": np.ones(N),
        }
    )


def current_frame(seed: int = 1) -> pl.DataFrame:
    """`shifted` moves by two sigma, `nulls` triples its missing rate, the rest holds."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "stable": rng.normal(size=N),
            "shifted": rng.normal(size=N) + 2.0,
            "nulls": [None if r < 0.6 else v for r, v in zip(rng.random(N), rng.normal(size=N))],
            "cat": rng.choice(["a", "b", "c"], size=N, p=[0.6, 0.3, 0.1]),
            "constant": np.ones(N),
        }
    )


COLUMNS = ["stable", "shifted", "nulls", "cat", "constant"]


@pytest.fixture(scope="module")
def ref() -> Reference:
    return Reference.fit(reference_frame(), COLUMNS)


@pytest.fixture(scope="module")
def psi(ref) -> pl.DataFrame:
    return ref.psi(current_frame())


def test_a_shifted_column_outranks_a_stable_one(psi):
    psi_dict = dict(zip(psi.get_column("column"), psi.get_column("psi")))
    assert psi_dict["shifted"] > psi_dict["stable"]
    assert psi_dict["shifted"] > 0.25  # "significant" by the usual convention
    assert psi_dict["stable"] < 0.01


def test_reference_against_itself_is_zero(ref):
    same = ref.psi(reference_frame())
    same_dict = dict(zip(same.get_column("column"), same.get_column("psi")))
    assert max(same_dict[col] for col in ["stable", "shifted", "nulls", "cat"]) < 1e-9


def test_missingness_is_a_bucket_not_a_dropped_row(psi):
    # `nulls` has the same distribution where present; only its null rate moved.
    nulls_row = psi.filter(pl.col("column") == "nulls").row(0, named=True)
    assert nulls_row["null_share_ref"] == pytest.approx(0.2, abs=0.02)
    assert nulls_row["null_share_cur"] == pytest.approx(0.6, abs=0.02)
    assert nulls_row["psi"] > 0.25
    assert nulls_row["top_bucket"] == NULL_BUCKET


def test_constant_column_is_degenerate_not_zero(psi):
    const_row = psi.filter(pl.col("column") == "constant").row(0, named=True)
    assert const_row["psi"] is None or np.isnan(const_row["psi"])
    assert "distinct" in const_row["note"]


def test_unseen_categorical_level_lands_in_other(ref):
    current = current_frame()
    current = current.with_row_index().with_columns(
        pl.when(pl.col("index") < N // 2).then(pl.lit("brand_new")).otherwise(pl.col("cat")).alias("cat")
    ).drop("index")

    row = ref.psi(current).filter(pl.col("column") == "cat").row(0, named=True)
    assert row["psi"] > 0.25
    assert row["top_bucket"] == OTHER_BUCKET


def test_binning_survives_a_json_round_trip(ref):
    import json

    restored = Reference.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert_frame_equal(ref.psi(current_frame()), restored.psi(current_frame()))


def test_edges_are_pinned_not_recomputed(ref):
    """The point of the whole class: a shifted sample must not re-bin itself to 'no shift'.

    Refitting on the current sample makes every column look stable, because each bucket
    is defined to hold a tenth of it. Same data, opposite conclusion.
    """
    refit = Reference.fit(current_frame(), COLUMNS)

    pinned = ref.psi(current_frame()).filter(pl.col("column") == "shifted").get_column("psi")[0]
    recomputed = refit.psi(current_frame()).filter(pl.col("column") == "shifted").get_column("psi")[0]

    assert pinned > 0.25
    assert recomputed < 1e-9


def test_adversarial_auc_separates_the_samples_and_names_the_column():
    auc, importance = adversarial_auc(reference_frame(), current_frame(), COLUMNS)

    assert auc > 0.9
    assert importance.row(0, named=True)["column"] in {"shifted", "nulls"}
    assert importance.get_column("gain_share").sum() == pytest.approx(1.0)


def test_adversarial_auc_is_chance_when_nothing_moved():
    auc, _ = adversarial_auc(reference_frame(0), reference_frame(1), ["stable", "cat"])
    assert auc == pytest.approx(0.5, abs=0.05)


def test_per_feature_adversarial_ranks_the_moved_columns_first():
    report = adversarial_per_feature(reference_frame(), current_frame(), COLUMNS)

    assert set(report.head(2).get_column("column")) == {"shifted", "nulls"}
    report_dict = dict(zip(report.get_column("column"), report.get_column("adversarial_auc")))
    assert report_dict["stable"] == pytest.approx(0.5, abs=0.05)
    assert report_dict["constant"] is None


def test_per_feature_parallel_matches_sequential():
    args = (reference_frame(), current_frame(), COLUMNS)
    assert_frame_equal(
        adversarial_per_feature(*args),
        adversarial_per_feature(*args, n_jobs=2),
    )
