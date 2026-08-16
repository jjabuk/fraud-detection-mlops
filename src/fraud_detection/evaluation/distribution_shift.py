"""Two questions about distribution shift that are usually confused for one.

**PSI** asks whether a column's *marginal distribution* moved. **Adversarial validation**
asks whether a model can tell the two periods apart at all. They are not the same
question, and a column can score high on one and low on the other.

The binning is pinned to the reference and travels with it, so PSI computed today and PSI
computed next month put the same values in the same buckets. Recomputing bucket edges per
call silently compares two different histograms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

__all__ = [
    "ADVERSARIAL_PARAMS",
    "NULL_BUCKET",
    "OTHER_BUCKET",
    "Binning",
    "Reference",
    "adversarial_auc",
    "adversarial_per_feature",
]

NULL_BUCKET = "__null__"
"""Missingness is a bucket, not a row to drop. On columns that are 86% null, a change in
how often the column is populated *is* the shift."""

OTHER_BUCKET = "__other__"
"""Categorical levels absent from the reference collapse here rather than being ignored."""

EPS = 1e-6
"""Floor on a bucket share. PSI divides by the reference share, so an empty reference
bucket would otherwise return infinity for a single stray row."""


@dataclass(frozen=True)
class Binning:
    """How one column is bucketed, and what the reference looked like inside it."""

    column: str
    kind: str  # "numeric" | "categorical"
    shares: dict[str, float]
    edges: tuple[float, ...] = ()
    levels: tuple[str, ...] = ()
    degenerate: str = ""

    def assign(self, s: pl.Series) -> pl.Series:
        """Map raw values onto reference buckets. Never invents a bucket."""
        if self.kind == "numeric":
            arr = s.to_numpy()
            # If all null, digitize fails on nan, wait digitize works on nan if nan is not in bins
            # Actually np.digitize places NaN at the end. We just need to handle nulls explicitly.
            indices = np.digitize(arr, self.edges)
            # Clip indices to max bin index
            indices = np.clip(indices, 1, len(self.edges) - 1)
            labels = np.array([f"b{i-1}" for i in indices], dtype=object)
            # Override nulls
            is_null = s.is_null().to_numpy()
            labels[is_null] = NULL_BUCKET
            return pl.Series(labels)
        else:
            known = set(self.levels)
            labels = s.cast(pl.String).map_elements(
                lambda v: v if (v is None or v in known) else OTHER_BUCKET,
                return_dtype=pl.String
            )
            return labels.fill_null(NULL_BUCKET)


@dataclass
class Reference:
    """A pinned baseline: bucket edges plus the reference distribution inside them.

    Serialize it with :meth:`to_dict` and commit the result. That file is what makes a
    PSI computed months apart comparable.
    """

    binnings: dict[str, Binning]
    n_rows: int
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        df: pl.DataFrame,
        columns: Sequence[str],
        *,
        bins: int = 10,
        max_levels: int = 20,
        meta: dict | None = None,
    ) -> Reference:
        binnings = {}
        for col in columns:
            s = df.get_column(col)
            null_share = float(s.null_count() / len(s))

            if s.dtype.is_numeric() and s.dtype != pl.Boolean:
                qs = np.linspace(0, 1, bins + 1)
                arr = s.to_numpy().astype("float64")
                edges = np.unique(np.nanquantile(arr, qs))
                if len(edges) < 3:
                    binnings[col] = Binning(
                        col, "numeric", {NULL_BUCKET: null_share}, degenerate="fewer than two distinct buckets"
                    )
                    continue
                edges[0], edges[-1] = -np.inf, np.inf
                b = Binning(col, "numeric", {}, edges=tuple(edges))
            else:
                s_no_nulls = s.drop_nulls()
                if len(s_no_nulls) == 0:
                    binnings[col] = Binning(
                        col, "categorical", {NULL_BUCKET: null_share}, degenerate="no non-null values"
                    )
                    continue
                
                counts = s_no_nulls.cast(pl.String).value_counts(sort=True)
                # value_counts returns struct with value and count. In Polars 1.0, it returns a df.
                levels = tuple(counts.get_column(col).head(max_levels).to_list())
                b = Binning(col, "categorical", {}, levels=levels)

            assigned = b.assign(s)
            counts_dict = assigned.value_counts(sort=True)
            # normalize
            shares = {k: v / len(s) for k, v in zip(counts_dict.get_column(assigned.name).to_list(), counts_dict.get_column("count").to_list())}
            binnings[col] = Binning(b.column, b.kind, shares, b.edges, b.levels)

        return cls(binnings=binnings, n_rows=len(df), meta=dict(meta or {}))

    def psi(self, df: pl.DataFrame) -> pl.DataFrame:
        """Per-column PSI of ``df`` against this reference. Highest shift first."""
        rows = []
        for col, b in self.binnings.items():
            if b.degenerate:
                rows.append(
                    {"column": col, "psi": np.nan, "null_share_ref": b.shares.get(NULL_BUCKET, np.nan),
                     "null_share_cur": float(df.get_column(col).null_count() / len(df)), "top_bucket": "",
                     "note": b.degenerate}
                )
                continue

            assigned = b.assign(df.get_column(col))
            counts_dict = assigned.value_counts(sort=True)
            cur = {k: v / len(df) for k, v in zip(counts_dict.get_column(assigned.name).to_list(), counts_dict.get_column("count").to_list())}
            keys = set(b.shares) | set(cur)
            terms = {
                k: (max(cur.get(k, 0.0), EPS) - max(b.shares.get(k, 0.0), EPS))
                * np.log(max(cur.get(k, 0.0), EPS) / max(b.shares.get(k, 0.0), EPS))
                for k in keys
            }
            top = max(terms, key=terms.get)
            rows.append(
                {"column": col, "psi": float(sum(terms.values())),
                 "null_share_ref": float(b.shares.get(NULL_BUCKET, 0.0)),
                 "null_share_cur": float(cur.get(NULL_BUCKET, 0.0)),
                 "top_bucket": top, "note": ""}
            )

        return pl.DataFrame(rows).sort("psi", descending=True, nulls_last=True)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "meta": self.meta,
            "binnings": {
                c: {"kind": b.kind, "shares": b.shares, "edges": list(b.edges),
                    "levels": list(b.levels), "degenerate": b.degenerate}
                for c, b in self.binnings.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Reference:
        return cls(
            binnings={
                c: Binning(c, v["kind"], v["shares"], tuple(v["edges"]), tuple(v["levels"]), v["degenerate"])
                for c, v in d["binnings"].items()
            },
            n_rows=d["n_rows"],
            meta=d.get("meta", {}),
        )


ADVERSARIAL_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": -1,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
}


def _oof_auc(x: pl.DataFrame, y: np.ndarray, params: dict | None, seed: int, folds: int) -> float:
    """Out-of-fold AUC. Scoring in-sample is the trap this function exists to avoid."""
    oof = np.zeros(len(y))
    
    cat_cols = [i for i, dtype in enumerate(x.dtypes) if dtype == pl.Categorical]
    
    # Cast categoricals to their physical integer representation to avoid pyarrow crash
    x_encoded = x.with_columns([
        pl.col(c).to_physical() for c in x.columns if x.schema[c] == pl.Categorical
    ]).to_numpy()

    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x_encoded, y):
        model = lgb.LGBMClassifier(**{**ADVERSARIAL_PARAMS, **(params or {}), "random_state": seed})
        model.fit(x_encoded[tr], y[tr], categorical_feature=cat_cols)
        oof[te] = model.predict_proba(x_encoded[te])[:, 1]
    return float(roc_auc_score(y, oof))


def _one_feature(
    ref_col: pl.DataFrame, cur_col: pl.DataFrame, col: str, params, seed, folds
) -> dict:
    """One column's discriminator. Module level so joblib can pickle it."""
    x, y = _stack(ref_col, cur_col, [col])
    if x.get_column(col).drop_nulls().n_unique() < 2:
        return {"column": col, "adversarial_auc": None, "note": "constant"}
    return {
        "column": col,
        "adversarial_auc": round(_oof_auc(x, y, params, seed, folds), 4),
        "note": "",
    }


def _stack(reference: pl.DataFrame, current: pl.DataFrame, columns: Sequence[str]):
    x = pl.concat([reference.select(columns), current.select(columns)])
    exprs = []
    for c in columns:
        if not x.schema[c].is_numeric():
            exprs.append(pl.col(c).cast(pl.Categorical))
    if exprs:
        x = x.with_columns(exprs)
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    return x, y

def adversarial_auc(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: Sequence[str],
    *,
    params: dict | None = None,
    seed: int = 0,
    folds: int = 3,
) -> tuple[float, pl.DataFrame]:
    """Can a model tell the two samples apart, and on what.

    Returns the **out-of-fold** AUC of that discriminator and its feature importances by
    gain, highest first. An AUC near 1 does not by itself mean the *relationship to the
    label* changed — only that the rows are distinguishable. The importances say why, and
    that is usually the more actionable half.
    """
    x, y = _stack(reference, current, columns)
    auc = _oof_auc(x, y, params, seed, folds)

    cat_cols = [i for i, dtype in enumerate(x.dtypes) if dtype == pl.Categorical]
    x_encoded = x.with_columns([
        pl.col(c).to_physical() for c in x.columns if x.schema[c] == pl.Categorical
    ]).to_numpy()

    model = lgb.LGBMClassifier(**{**ADVERSARIAL_PARAMS, **(params or {}), "random_state": seed})
    model.fit(x_encoded, y, categorical_feature=cat_cols)
    imp = pl.DataFrame({"column": list(columns), "gain": model.booster_.feature_importance("gain")})
    imp = imp.sort("gain", descending=True)
    imp = imp.with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_share"))
    return auc, imp


def adversarial_per_feature(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: Sequence[str],
    *,
    n_jobs: int = 1,
    params: dict | None = None,
    seed: int = 0,
    folds: int = 3,
) -> pl.DataFrame:
    """One discriminator per column: how well does this column alone date a row?

    0.5 means the column cannot tell the samples apart. 1.0 means it is a timestamp in
    disguise. Out-of-fold, so 0.5 really is the floor. Comparable across columns, and to
    a per-column PSI.
    """
    args = [(reference.select([c]), current.select([c]), c, params, seed, folds) for c in columns]

    if n_jobs == 1:
        rows = [_one_feature(*a) for a in args]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs)(delayed(_one_feature)(*a) for a in args)

    return (
        pl.DataFrame(rows)
        .sort("adversarial_auc", descending=True, nulls_last=True)
    )
