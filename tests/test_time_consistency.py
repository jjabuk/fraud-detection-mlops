"""Synthetic ground truth: each fixture knows what verdict it deserves."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fraud_detection.evaluation.time_consistency import (
    Verdict,
    scan,
    time_consistency,
    time_windows,
)

N = 4_000
RATE = 0.2


def make(seed: int = 0) -> pl.DataFrame:
    """Two windows of equal size. `t` is the time axis, `y` the label.

    - `stable`   predicts y the same way in both windows
    - `inverted` predicts y in one direction early and the opposite direction late
    - `noise`    is independent of y
    - `constant` never varies
    - `empty`    is null everywhere
    """
    rng = np.random.default_rng(seed)
    y = rng.random(N) < RATE
    late = np.arange(N) >= N // 2

    stable = rng.normal(size=N) + y * 1.5
    flip = np.where(late, -1.0, 1.0)
    inverted = rng.normal(size=N) + y * 1.5 * flip

    return pl.DataFrame(
        {
            "t": np.arange(N),
            "y": y.astype(int),
            "stable": stable,
            "inverted": inverted,
            "noise": rng.normal(size=N),
            "constant": np.ones(N),
            "empty": [None] * N,
        },
        schema_overrides={"empty": pl.Float64}
    )


@pytest.fixture(scope="module")
def windows() -> tuple[pl.DataFrame, pl.DataFrame]:
    return time_windows(make(), "t", train=(0.0, 0.5), holdout=(0.5, 1.0))


def check(windows, feature: str, **kw):
    tr, ho = windows
    return time_consistency(tr, ho, feature, "y", **kw)


def test_stable_feature_passes(windows):
    r = check(windows, "stable")
    assert r.verdict is Verdict.PASS
    assert r.auc_train > 0.7 and r.auc_holdout > 0.7
    assert abs(r.delta) < 0.1


def test_inverted_feature_is_caught(windows):
    r = check(windows, "inverted")
    assert r.verdict is Verdict.INVERTED
    assert r.auc_train > 0.5 > r.auc_holdout
    assert r.delta < -0.2


def test_pure_noise_is_weak_not_inverted(windows):
    r = check(windows, "noise")
    assert r.verdict in (Verdict.WEAK, Verdict.PASS)
    assert r.verdict is not Verdict.INVERTED


@pytest.mark.parametrize("feature,fragment", [("constant", "constant"), ("empty", "all-null")])
def test_unevaluable_features_are_degenerate(windows, feature, fragment):
    r = check(windows, feature)
    assert r.verdict is Verdict.DEGENERATE
    assert fragment in r.note
    assert r.auc_train is None and r.delta is None


def test_single_class_window_is_degenerate(windows):
    tr, ho = windows
    r = time_consistency(tr.with_columns(pl.lit(0).alias("y")), ho, "stable", "y")
    assert r.verdict is Verdict.DEGENERATE
    assert "one class" in r.note


def test_too_few_positives_is_degenerate(windows):
    tr, ho = windows
    sparse = tr.with_row_index().with_columns(
        pl.when(pl.col("index") >= 5).then(0).otherwise(pl.col("y")).alias("y")
    ).drop("index")
    r = time_consistency(sparse, ho, "stable", "y")
    assert r.verdict is Verdict.DEGENERATE
    assert "min_positives" in r.note


def test_categorical_feature_is_accepted(windows):
    tr, ho = windows
    to_cat = lambda d: d.with_columns(pl.col("stable").round(0).cast(pl.String).cast(pl.Categorical).alias("cat"))
    r = time_consistency(to_cat(tr), to_cat(ho), "cat", "y")
    assert r.verdict is Verdict.PASS
    assert r.auc_train > 0.6


def test_result_is_deterministic(windows):
    assert check(windows, "stable") == check(windows, "stable")


def test_scan_ranks_the_inverted_feature_first(windows):
    tr, ho = windows
    report = scan(tr, ho, ["stable", "noise", "constant", "inverted", "empty"], "y")

    assert len(report) == 5
    assert report.row(0, named=True)["feature"] == "inverted"
    assert report.row(-1, named=True)["verdict"] == Verdict.DEGENERATE.value
    assert set(report.get_column("feature")) == {"stable", "noise", "constant", "inverted", "empty"}


def test_parallel_scan_is_identical_to_sequential(windows):
    tr, ho = windows
    features = ["stable", "noise", "constant", "inverted", "empty"]

    assert_frame_equal(
        scan(tr, ho, features, "y"),
        scan(tr, ho, features, "y", n_jobs=2),
    )


def test_parallel_scan_reports_every_feature_once(windows):
    tr, ho = windows
    features = ["stable", "noise", "inverted"]

    seen: list[str] = []
    report = scan(tr, ho, features, "y", n_jobs=2, on_result=lambda r: seen.append(r.feature))

    assert seen == features
    assert len(report) == len(features)


def test_windows_do_not_overlap_and_are_ordered():
    df = make()
    tr, ho = time_windows(df, "t", train=(0.0, 0.25), holdout=(0.75, 1.0))
    assert tr.get_column("t").max() < ho.get_column("t").min()
    assert len(tr) > 0 and len(ho) > 0
    # The skipped middle is real, not an off-by-one.
    assert len(tr) + len(ho) < len(df)


def test_window_last_row_is_never_dropped():
    df = make()
    _, ho = time_windows(df, "t", train=(0.0, 0.25), holdout=(0.75, 1.0))
    assert ho.get_column("t").max() == df.get_column("t").max()


def test_overlapping_windows_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        time_windows(make(), "t", train=(0.0, 0.6), holdout=(0.5, 1.0))


def test_backwards_window_is_rejected():
    with pytest.raises(ValueError, match="increasing range"):
        time_windows(make(), "t", train=(0.5, 0.2), holdout=(0.75, 1.0))
