"""Synthetic ground truth: two columns carry the label, the rest are decoration."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fraud_detection.evaluation.selection import (
    forward_selection,
    from_forward_selection,
    pca_groups,
)

N = 4_000


def make(seed: int = 0) -> pl.DataFrame:
    """`signal_a` and `signal_b` predict `y`. `noise_*` do not. `echo_a` copies `signal_a`."""
    rng = np.random.default_rng(seed)
    y = rng.random(N) < 0.3

    return pl.DataFrame({
        "y": y.astype(int),
        "signal_a": rng.normal(size=N) + y * 1.6,
        "signal_b": rng.normal(size=N) + y * 1.2,
        "echo_a": rng.normal(size=N) + y * 1.6,
        "noise_1": rng.normal(size=N),
        "noise_2": rng.normal(size=N),
        "noise_3": rng.normal(size=N),
    })


CANDIDATES = ["noise_1", "signal_a", "noise_2", "signal_b", "noise_3", "echo_a"]


@pytest.fixture(scope="module")
def split():
    df = make()
    return df.head(N // 2), df.tail(N - N // 2)


# ---- forward selection ---------------------------------------------------------


def test_the_predictive_columns_are_picked_first(split):
    train, holdout = split
    result = forward_selection(train, holdout, CANDIDATES, "y", max_features=3, min_gain=0.0)

    assert set(result.selected[:2]) <= {"signal_a", "signal_b", "echo_a"}
    assert "noise_1" not in result.selected[:2]


def test_the_search_stops_when_the_gain_stops_paying(split):
    train, holdout = split
    result = forward_selection(train, holdout, CANDIDATES, "y", max_features=6, min_gain=0.01)

    assert len(result.selected) < len(CANDIDATES)
    assert "below min_gain" in result.stopped_because


def test_the_cap_is_respected_and_reported(split):
    train, holdout = split
    result = forward_selection(train, holdout, CANDIDATES, "y", max_features=2, min_gain=0.0)

    assert len(result.selected) == 2
    assert result.stopped_because == "reached max_features"


def test_every_candidate_is_either_selected_or_rejected(split):
    train, holdout = split
    result = forward_selection(train, holdout, CANDIDATES, "y", max_features=3, min_gain=0.0)

    assert set(result.selected) | set(result.rejected) == set(CANDIDATES)
    assert not set(result.selected) & set(result.rejected)


def test_the_path_is_recorded_not_just_the_destination(split):
    train, holdout = split
    result = forward_selection(train, holdout, CANDIDATES, "y", max_features=3, min_gain=0.0)

    assert list(result.steps.get_column("column")) == result.selected
    assert result.steps.get_column("auc").is_sorted()
    assert result.auc > 0.7


def test_candidates_absent_from_the_frame_are_ignored(split):
    train, holdout = split
    result = forward_selection(train, holdout, [*CANDIDATES, "not_a_column"], "y", max_features=2, min_gain=0.0)

    assert "not_a_column" not in result.selected + result.rejected


def test_parallel_selection_matches_sequential(split):
    train, holdout = split
    kw = {"max_features": 3, "min_gain": 0.0}

    a = forward_selection(train, holdout, CANDIDATES, "y", **kw)
    b = forward_selection(train, holdout, CANDIDATES, "y", n_jobs=2, **kw)

    assert a.selected == b.selected
    from polars.testing import assert_frame_equal
    assert_frame_equal(a.steps, b.steps)


def test_the_fragment_rejects_everything_the_search_did_not_reach(split):
    train, holdout = split
    result = forward_selection(train, holdout, CANDIDATES, "y", max_features=2, min_gain=0.0)
    frag = from_forward_selection(result)

    assert {r.column for r in frag.rejections} == set(result.rejected)
    assert frag.qualification["selected"] == 2
    assert "max_features" in frag.params["stopped_because"]


# ---- PCA -----------------------------------------------------------------------


def test_pca_replaces_a_group_with_one_component():
    df = make()
    out = pca_groups(df, [["signal_a", "echo_a"], ["noise_1", "noise_2"]])

    assert out.n_inputs == 4
    assert out.n_outputs == 2
    assert len(out.components) == len(df)


def test_a_group_that_is_really_one_signal_keeps_most_of_its_variance():
    df = make()
    out = pca_groups(df, [["signal_a", "echo_a"], ["noise_1", "noise_2"]]).explained
    
    correlated = out.filter(pl.col("group") == "signal_a+echo_a").row(0, named=True)["explained_variance"]
    unrelated = out.filter(pl.col("group") == "noise_1+noise_2").row(0, named=True)["explained_variance"]

    assert correlated > unrelated
    assert unrelated < 0.65  # two independent columns split their variance evenly


def test_more_components_retain_more_variance():
    df = make()
    one = pca_groups(df, [["signal_a", "echo_a", "signal_b"]], n_components=1)
    two = pca_groups(df, [["signal_a", "echo_a", "signal_b"]], n_components=2)

    assert two.n_outputs == 2
    assert two.explained.get_column("explained_variance")[0] > one.explained.get_column("explained_variance")[0]


def test_nulls_are_imputed_and_reported_rather_than_dropping_rows():
    df = make()
    df = df.with_columns(
        pl.when(pl.Series(np.arange(len(df))) < 1000)
        .then(None)
        .otherwise(pl.col("signal_a"))
        .alias("signal_a")
    )
    out = pca_groups(df, [["signal_a", "echo_a"]])

    assert len(out.components) == len(df)  # no row lost
    assert out.components.null_count().sum_horizontal()[0] == 0
    assert out.explained.get_column("null_share")[0] > 0.1  # the cost is visible in the report


def test_a_column_that_is_null_everywhere_does_not_break_the_fit():
    df = make()
    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("all_null"))
    out = pca_groups(df, [["signal_a", "all_null"]])

    assert out.n_outputs == 1
    assert out.components.null_count().sum_horizontal()[0] == 0


def test_groups_with_no_present_columns_are_skipped():
    out = pca_groups(make(), [["nowhere"], ["signal_a", "echo_a"]])
    assert out.n_outputs == 1
