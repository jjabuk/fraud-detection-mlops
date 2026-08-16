"""Does a column carry signal *within* each segment, or only between segments?

The other five audits judge a column on pooled data, and pooling is where this particular
failure hides. `ProductCD` splits the population into groups with very different fraud
rates — `W` runs at 0.0198 over 77% of rows, `C` at 0.1467 over 11% — so a column that
merely correlates with the product earns a pooled AUC for separating *products*, and a
pooled check records it as fraud signal.

**What this reports, and what it does not claim.** A rank statistic per column per segment,
not a verdict. A lower within-segment score is not automatically a defect: a segment with a
seven-times lower base rate is a harder problem, and between-segment signal is genuinely
predictive when the segment predicts the outcome. Acting on these verdicts was measured and
made the model worse, which is why the fragment built from them runs in report-only mode.
What the numbers do support is narrower and useful: a column at 0.50 inside the segment
holding most of the traffic is not doing there what its pooled score implies.

Direction-free (`max(auc, 1 - auc)`): a feature that ranks the other way is still a feature.
An inversion *between time windows* is `time_consistency`'s question, not this one.

No model is fitted — 225 columns across 5 segments would be over a thousand fits under
`time_consistency`'s design, and a rank statistic answers "is there monotone signal here"
without any of them. A weaker instrument that runs in seconds, which is the right trade for
a screen.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

__all__ = [
    "MIN_POSITIVES",
    "MIN_ROWS",
    "SegmentScore",
    "column_auc",
    "qualify",
    "summarise",
]

#: Below these a segment cannot support an estimate, and the score is `None`.
#: Absent is not zero.
MIN_ROWS = 500
MIN_POSITIVES = 20


@dataclass(frozen=True)
class SegmentScore:
    """One column, one segment."""

    column: str
    segment: str
    auc: float | None
    rows: int
    positives: int
    measurable: bool

    @property
    def note(self) -> str:
        if self.measurable:
            return ""
        if self.rows < MIN_ROWS:
            return f"fewer than {MIN_ROWS} rows"
        if self.positives < MIN_POSITIVES:
            return f"fewer than {MIN_POSITIVES} positives"
        return "constant or all-null in this segment"


def _numeric(series: pl.Series) -> np.ndarray:
    """Anything comparable, as floats. Categoricals go through their physical codes.

    The codes are arbitrary integers, so a monotone statistic over them means little for an
    unordered categorical: a low score there is not evidence against the column.
    """
    if series.dtype in (pl.Categorical, pl.Enum):
        series = series.to_physical()
    elif series.dtype == pl.String:
        series = series.cast(pl.Categorical).to_physical()
    elif series.dtype == pl.Boolean:
        series = series.cast(pl.Int8)
    return series.cast(pl.Float64, strict=False).to_numpy()


def column_auc(values: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[float | None, int, int]:
    """Direction-free single-feature AUC over the masked rows. Returns (auc, rows, positives)."""
    usable = mask & ~np.isnan(values)
    rows = int(usable.sum())
    positives = int(labels[usable].sum()) if rows else 0
    if rows < MIN_ROWS or positives < MIN_POSITIVES or np.unique(values[usable]).size < 2:
        return None, rows, positives
    auc = float(roc_auc_score(labels[usable], values[usable]))
    return max(auc, 1.0 - auc), rows, positives


def qualify(
    frame: pl.DataFrame,
    columns: Sequence[str],
    label: str,
    segment_column: str,
    *,
    include_pooled: bool = True,
) -> pl.DataFrame:
    """Score every column in every segment, plus the pooled view for comparison.

    The pooled row is what the other five audits see. Keeping it in the same table is the
    point: the finding is never a within-segment number on its own, it is the *gap*.
    """
    labels = frame.get_column(label).cast(pl.Int8).to_numpy()
    segments = frame.get_column(segment_column)
    levels = [lvl for lvl in segments.unique().drop_nulls().to_list()]

    masks = {str(level): (segments == level).to_numpy() for level in levels}
    if include_pooled:
        masks["__pooled__"] = np.ones(len(frame), dtype=bool)

    rows: list[SegmentScore] = []
    for column in columns:
        values = _numeric(frame.get_column(column))
        for segment, mask in masks.items():
            auc, n, positives = column_auc(values, labels, mask)
            rows.append(
                SegmentScore(column, segment, auc, n, positives, measurable=auc is not None)
            )

    return pl.DataFrame(
        [
            {
                "column": s.column,
                "segment": s.segment,
                "auc": None if s.auc is None else round(s.auc, 4),
                "rows": s.rows,
                "positives": s.positives,
                "measurable": s.measurable,
                "note": s.note,
            }
            for s in rows
        ]
    )


def summarise(scored: pl.DataFrame, segment: str, *, thresholds=(0.55, 0.60, 0.65)) -> pl.DataFrame:
    """Pooled against one segment: how much of the strong tail survives.

    The *tail*, not the median. Most columns are weak in both views, so the median barely
    moves while the columns the model leans on can halve.
    """
    pooled = scored.filter(pl.col("segment") == "__pooled__").select(["column", "auc"])
    inside = scored.filter(pl.col("segment") == segment).select(
        ["column", pl.col("auc").alias("auc_segment")]
    )
    joined = pooled.join(inside, on="column", how="inner").drop_nulls()

    return pl.DataFrame(
        [
            {
                "threshold": t,
                "pooled_above": int((joined["auc"] > t).sum()),
                "segment_above": int((joined["auc_segment"] > t).sum()),
            }
            for t in thresholds
        ]
    )


def collapsed(scored: pl.DataFrame, segment: str, *, pooled_floor: float = 0.60) -> pl.DataFrame:
    """Columns that look strong pooled and are a coin flip inside the segment, worst first.

    This is the list a contract review should read: every row is a column admitted on
    evidence that does not hold where most of the traffic is.
    """
    pooled = scored.filter(pl.col("segment") == "__pooled__").select(["column", "auc"])
    inside = scored.filter(pl.col("segment") == segment).select(
        ["column", pl.col("auc").alias("auc_segment")]
    )
    return (
        pooled.join(inside, on="column", how="inner")
        .drop_nulls()
        .filter(pl.col("auc") > pooled_floor)
        .with_columns(delta=(pl.col("auc_segment") - pl.col("auc")).round(4))
        .sort("delta")
    )
