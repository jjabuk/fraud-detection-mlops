"""The sixth audit: signal within a segment, against signal between segments.

The property under test is the one the other five audits cannot see. A column that only
separates *segments* earns a pooled AUC from the segments' differing base rates, and every
pooled check credits it as signal. These tests build that situation on purpose and assert
the module reports it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fraud_detection.evaluation.segment_qualification import (
    MIN_POSITIVES,
    MIN_ROWS,
    collapsed,
    column_auc,
    qualify,
    summarise,
)

RNG = np.random.default_rng(0)


def frame_with_two_segments(n: int = 4000) -> pl.DataFrame:
    """Two segments with very different base rates, and three kinds of column.

    * `product_proxy` is a constant per segment — it separates segments perfectly and
      carries nothing about fraud inside either. Pooled it looks strong; within, it is
      undefined. This is `card3` and `M4`.
    * `real_signal` predicts fraud identically in both segments.
    * `noise` predicts nothing anywhere.
    """
    segment = np.array(["W"] * (n // 2) + ["C"] * (n // 2))
    # W is the big, low-base-rate segment; C the small, high-base-rate one — the shape
    # ProductCD actually has (0.0198 against 0.1467).
    rate = np.where(segment == "W", 0.05, 0.40)
    labels = (RNG.random(n) < rate).astype(np.int8)

    return pl.DataFrame(
        {
            "isFraud": labels,
            "ProductCD": segment,
            "product_proxy": np.where(segment == "W", 0.0, 1.0),
            "real_signal": labels * 2.0 + RNG.normal(0, 1.0, n),
            "noise": RNG.normal(0, 1, n),
        }
    )


COLUMNS = ["product_proxy", "real_signal", "noise"]


def test_a_segment_proxy_scores_pooled_and_is_undefined_within():
    """The whole point. `product_proxy` is constant inside each segment, so there is
    nothing to rank there — and pooled it inherits the base-rate gap."""
    scored = qualify(frame_with_two_segments(), COLUMNS, "isFraud", "ProductCD")

    pooled = scored.filter((pl.col("column") == "product_proxy") & (pl.col("segment") == "__pooled__"))
    inside = scored.filter((pl.col("column") == "product_proxy") & (pl.col("segment") == "W"))

    assert pooled["auc"][0] > 0.60, "the proxy should look strong on pooled data"
    assert inside["measurable"][0] is False
    assert "constant" in inside["note"][0]


def test_real_signal_survives_the_split():
    scored = qualify(frame_with_two_segments(), COLUMNS, "isFraud", "ProductCD")
    inside = scored.filter((pl.col("column") == "real_signal") & (pl.col("segment") == "W"))

    assert inside["auc"][0] > 0.70


def test_noise_is_near_a_coin_flip_everywhere():
    scored = qualify(frame_with_two_segments(), COLUMNS, "isFraud", "ProductCD")
    for segment in ("__pooled__", "W", "C"):
        row = scored.filter((pl.col("column") == "noise") & (pl.col("segment") == segment))
        assert row["auc"][0] < 0.58, segment


def test_the_score_is_direction_free():
    """A feature that ranks the other way is still a feature. An inversion *between time
    windows* is `time_consistency`'s business and a different question entirely."""
    labels = np.array([0] * 1000 + [1] * 1000, dtype=np.int8)
    rising = np.concatenate([RNG.normal(0, 1, 1000), RNG.normal(3, 1, 1000)])
    falling = -rising
    mask = np.ones(2000, dtype=bool)

    up, _, _ = column_auc(rising, labels, mask)
    down, _, _ = column_auc(falling, labels, mask)

    assert up is not None and down is not None
    assert abs(up - down) < 1e-9


def test_a_segment_too_small_to_judge_says_so_rather_than_scoring_it():
    """Absent is not zero — the rule the promotion gate had to learn the hard way."""
    values = np.arange(100, dtype=float)
    labels = np.zeros(100, dtype=np.int8)
    labels[:30] = 1

    auc, rows, positives = column_auc(values, labels, np.ones(100, dtype=bool))

    assert auc is None
    assert rows < MIN_ROWS
    assert positives >= MIN_POSITIVES  # the row count is what disqualified it


def test_positives_floor_is_enforced_independently_of_rows():
    values = RNG.normal(0, 1, 4000)
    labels = np.zeros(4000, dtype=np.int8)
    labels[:5] = 1  # plenty of rows, far too few positives

    auc, rows, positives = column_auc(values, labels, np.ones(4000, dtype=bool))

    assert auc is None
    assert rows >= MIN_ROWS
    assert positives < MIN_POSITIVES


def test_summarise_counts_the_tail_not_the_median():
    """The median barely moved on real data (0.5407 -> 0.5350) while the strong tail
    halved. A summary that reported the median would have hidden the finding."""
    scored = qualify(frame_with_two_segments(), COLUMNS, "isFraud", "ProductCD")
    table = summarise(scored, "W")

    assert set(table.columns) == {"threshold", "pooled_above", "segment_above"}
    assert table.height == 3


def test_collapsed_lists_the_columns_a_contract_review_should_read():
    scored = qualify(frame_with_two_segments(), COLUMNS, "isFraud", "ProductCD")
    worst = collapsed(scored, "W", pooled_floor=0.55)

    # `product_proxy` is undefined within W, so it drops out of the join rather than
    # appearing with a fabricated delta — a column nobody can score is not a column that
    # scored badly, and the two must not look alike.
    assert "product_proxy" not in worst["column"].to_list()
    assert "real_signal" in worst["column"].to_list()
    assert worst["delta"].to_list() == sorted(worst["delta"].to_list())
