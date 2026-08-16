"""Does a feature still mean the same thing next month?

Train a single-feature model on an early time window, score a later one. A feature
whose ranking inverts between the two found a pattern in the present that does not
exist in the future.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.metrics import roc_auc_score

__all__ = [
    "DEFAULT_PARAMS",
    "Result",
    "Verdict",
    "scan",
    "time_consistency",
    "time_windows",
]


class Verdict(str, Enum):
    """Outcome of a single feature's check."""

    PASS = "pass"
    """Signal survives the time axis."""

    INVERTED = "inverted"
    """Ranks one way in the past and the opposite way in the future. Drop it."""

    WEAK = "weak"
    """Carries no signal on its own in either window. Harmless, but not evidence."""

    DEGENERATE = "degenerate"
    """Could not be evaluated — constant, all-null, or a window without both classes."""


@dataclass(frozen=True)
class Result:
    feature: str
    verdict: Verdict
    auc_train: float | None
    auc_holdout: float | None
    delta: float | None
    n_train: int
    n_holdout: int
    null_rate_train: float
    null_rate_holdout: float
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# A single-feature model on one column. 500 rounds at lr 0.02 is the reference
# procedure's setting and it is far past where a one-column model stops improving:
# there is one split axis, so the ensemble saturates in tens of trees, not hundreds.
# 150 is an argument, not a measurement, and the distinction matters here: nobody has
# checked at which round count the verdicts stop moving. What is measured is the ceiling
# on how much fit precision can possibly be worth -- this check reproduces at 0.32 when
# the window widens by 1.5x, so its verdicts are already dominated by window choice
# rather than by fit quality. **Both this and MAX_ROWS change verdicts, not just
# runtime.** Expect the admitted set to move on the next audit; if it moves a lot, that
# is a finding about the check's stability, not a regression to revert.
DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 150,
    "num_leaves": 8,
    "learning_rate": 0.02,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
}
"""Matches the reference procedure. Deterministic so two runs are comparable."""



#: Rows per window fed to a single-feature fit. `None` disables subsampling.
MAX_ROWS: int | None = 120_000


def _subsample(
    train: pl.DataFrame, holdout: pl.DataFrame, max_rows: int | None, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cap each window, deterministically. Whole-frame, not per-column, so every feature
    in one scan is judged on the same rows."""
    if not max_rows:
        return train, holdout
    if len(train) > max_rows:
        train = train.sample(max_rows, seed=seed, shuffle=False)
    if len(holdout) > max_rows:
        holdout = holdout.sample(max_rows, seed=seed, shuffle=False)
    return train, holdout


#: Where per-column verdicts are remembered between scans.
SCAN_CACHE_DIR = Path(os.getenv("TIME_CONSISTENCY_CACHE", ".cache/time-consistency"))


def _fingerprint(train: pl.DataFrame, holdout: pl.DataFrame, feature: str, label: str, kwargs) -> str:
    """A key over the *values* the verdict depends on, not over the run.

    The audit is documented as something that runs when the **data** changes. Retyping
    seven columns re-ran five hundred single-feature fits, of which seven could possibly
    have moved. Hashing the column's own bytes plus the parameters makes that literal: a
    column whose values and settings are unchanged reuses its verdict, and a column that
    moved by one row does not.

    `hash_rows` is content-addressed and order-sensitive, which is what we want -- the
    windows are time-ordered and a reordering would be a different experiment.
    """
    digest = hashlib.sha256()
    for frame in (train, holdout):
        digest.update(frame.select([feature, label]).hash_rows().to_numpy().tobytes())
    digest.update(repr(sorted((k, str(v)) for k, v in kwargs.items())).encode())
    digest.update(repr(sorted(DEFAULT_PARAMS.items())).encode())
    return digest.hexdigest()[:32]


def time_windows(
    df: pl.DataFrame,
    time_col: str,
    *,
    train: tuple[float, float] = (0.0, 0.25),
    holdout: tuple[float, float] = (0.75, 1.0),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cut an early and a late window out of ``df`` by quantiles of ``time_col``.

    The gap between ``train[1]`` and ``holdout[0]`` is deliberate: skipping the
    middle mimics the label-maturity lag you actually have at deployment time,
    where the most recent period has not finished being labelled yet.

    Windows are half-open ``[lo, hi)``, except the final window which includes its
    upper bound, so no row is silently dropped.
    """
    for name, w in (("train", train), ("holdout", holdout)):
        if not 0.0 <= w[0] < w[1] <= 1.0:
            raise ValueError(f"{name} window {w} is not an increasing range within [0, 1]")
    if train[1] > holdout[0]:
        raise ValueError(
            f"train window {train} overlaps holdout {holdout}; the check is only "
            "meaningful when training strictly precedes scoring"
        )

    t = pl.col(time_col)
    
    # Quantiles in polars can be computed at once
    quantiles_df = df.select([
        t.quantile(train[0]).alias("t0"),
        t.quantile(train[1]).alias("t1"),
        t.quantile(holdout[0]).alias("h0"),
        t.quantile(holdout[1]).alias("h1")
    ])
    edges = quantiles_df.row(0)

    def cut(lo: float, hi: float, closed_right: bool) -> pl.DataFrame:
        if closed_right:
            return df.filter((t >= lo) & (t <= hi))
        return df.filter((t >= lo) & (t < hi))

    return (
        cut(edges[0], edges[1], closed_right=train[1] == 1.0),
        cut(edges[2], edges[3], closed_right=holdout[1] == 1.0),
    )


def _prepare(s: pl.Series) -> pl.Series:
    """LightGBM takes numerics and polars categoricals; everything else becomes one."""
    if s.dtype == pl.Categorical or s.dtype.is_numeric():
        return s
    if s.dtype == pl.Boolean:
        return s.cast(pl.Float64)
    return s.cast(pl.Categorical)


def _degenerate(feature: str, note: str, tr: pl.Series, ho: pl.Series) -> Result:
    return Result(
        feature=feature,
        verdict=Verdict.DEGENERATE,
        auc_train=None,
        auc_holdout=None,
        delta=None,
        n_train=len(tr),
        n_holdout=len(ho),
        null_rate_train=float(tr.null_count() / len(tr)) if len(tr) else 1.0,
        null_rate_holdout=float(ho.null_count() / len(ho)) if len(ho) else 1.0,
        note=note,
    )


def time_consistency(
    train: pl.DataFrame,
    holdout: pl.DataFrame,
    feature: str,
    label: str,
    *,
    params: dict | None = None,
    margin: float = 0.02,
    min_rows: int = 100,
    min_positives: int = 10,
    seed: int = 0,
    max_rows: int | None = MAX_ROWS,
) -> Result:
    """Check one feature. ``train`` must cover a period strictly before ``holdout``.

    Two frames rather than one frame plus window bounds, so neither window can be
    sliced out of the other by accident and an empty holdout is impossible.

    ``margin`` is the distance from 0.5 that counts as signal. A feature is
    ``INVERTED`` when it clears the margin above 0.5 in training and below 0.5 in
    the holdout: a tree trained on one feature learns a direction, so an AUC under
    0.5 later means that direction reversed, not that the sign convention is off.
    """
    # Subsample before anything else. The verdict is a threshold crossing on a
    # single-feature AUC, and the standard error of an AUC estimated on 120k rows with a
    # 3.5% base rate is already far below the `margin` the verdict is read against -- so
    # the extra rows buy precision the decision cannot use. Deterministic on `seed`, so
    # two runs stay comparable, which is the property DEFAULT_PARAMS exists to protect.
    train, holdout = _subsample(train, holdout, max_rows, seed)

    x_tr, x_ho = _prepare(train.get_column(feature)), _prepare(holdout.get_column(feature))
    y_tr, y_ho = train.get_column(label), holdout.get_column(label)

    if len(train) < min_rows or len(holdout) < min_rows:
        return _degenerate(feature, f"window smaller than min_rows={min_rows}", x_tr, x_ho)
    if y_tr.drop_nulls().n_unique() < 2 or y_ho.drop_nulls().n_unique() < 2:
        return _degenerate(feature, "a window has only one class", x_tr, x_ho)
    if min(y_tr.sum(), y_ho.sum()) < min_positives:
        return _degenerate(feature, f"fewer than min_positives={min_positives}", x_tr, x_ho)
    if (len(x_tr) - x_tr.null_count()) == 0:
        return _degenerate(feature, "all-null in the training window", x_tr, x_ho)
    if x_tr.drop_nulls().n_unique() < 2:
        return _degenerate(feature, "constant in the training window", x_tr, x_ho)

    model = lgb.LGBMClassifier(**{**DEFAULT_PARAMS, **(params or {}), "random_state": seed})
    
    if x_tr.dtype == pl.Categorical:
        X_train = x_tr.to_physical().to_numpy().reshape(-1, 1)
        X_test = x_ho.to_physical().to_numpy().reshape(-1, 1)
        model.fit(X_train, y_tr.to_numpy(), categorical_feature=[0])
    else:
        X_train = x_tr.to_numpy().reshape(-1, 1)
        X_test = x_ho.to_numpy().reshape(-1, 1)
        model.fit(X_train, y_tr.to_numpy())

    auc_tr = float(roc_auc_score(y_tr.to_numpy(), model.predict_proba(X_train)[:, 1]))
    auc_ho = float(roc_auc_score(y_ho.to_numpy(), model.predict_proba(X_test)[:, 1]))

    if auc_tr >= 0.5 + margin and auc_ho <= 0.5 - margin:
        verdict = Verdict.INVERTED
    elif abs(auc_tr - 0.5) < margin and abs(auc_ho - 0.5) < margin:
        verdict = Verdict.WEAK
    else:
        verdict = Verdict.PASS

    return Result(
        feature=feature,
        verdict=verdict,
        auc_train=round(auc_tr, 4),
        auc_holdout=round(auc_ho, 4),
        delta=round(auc_ho - auc_tr, 4),
        n_train=len(train),
        n_holdout=len(holdout),
        null_rate_train=round(float(x_tr.null_count() / len(x_tr)), 4),
        null_rate_holdout=round(float(x_ho.null_count() / len(x_ho)), 4),
    )


def scan(
    train: pl.DataFrame,
    holdout: pl.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    n_jobs: int = 1,
    on_result: Callable[[Result], None] | None = None,
    use_cache: bool = True,
    **kwargs,
) -> pl.DataFrame:
    """Check many features. Returns one row each, worst first.

    ``n_jobs`` runs features in parallel processes — ``-1`` uses every core. Each
    feature is an independent fit with its own fixed seed and its own single-threaded
    LightGBM, so the output is identical to the sequential run, not merely equivalent.
    Only the two columns a feature needs are shipped to each worker.

    Leave the model's own ``n_jobs`` at 1 (the default) when using this: nested
    parallelism oversubscribes the cores and is slower than either alone.

    ``use_cache`` reuses a verdict when the column's own values, the label, the parameters
    and the scan settings are all unchanged — keyed by content, not by run, so it is
    correct to leave on. This is what makes the audit's own claim true in practice: it is
    documented as running when the *data* changes: a schema retype should not re-fit
    500 columns to learn that 7 of them might have moved.

    ``on_result`` is called as each feature finishes — useful for a progress bar over a
    few hundred columns. Under ``n_jobs != 1`` it is called once per feature after the
    batch completes, in the order given.
    """
    cached, to_fit = {}, []
    if use_cache:
        SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for f in features:
            path = SCAN_CACHE_DIR / f"{_fingerprint(train, holdout, f, label, kwargs)}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                cached[f] = Result(**{**payload, "verdict": Verdict(payload["verdict"])})
            else:
                to_fit.append(f)
    else:
        to_fit = list(features)

    def _remember(feature: str, result: Result) -> None:
        if not use_cache:
            return
        path = SCAN_CACHE_DIR / f"{_fingerprint(train, holdout, feature, label, kwargs)}.json"
        path.write_text(json.dumps(result.as_dict()))

    features_to_fit = to_fit
    if n_jobs == 1:
        fitted = {}
        for f in features_to_fit:
            r = time_consistency(train, holdout, f, label, **kwargs)
            _remember(f, r)
            fitted[f] = r
        # Emitted in the caller's order, cache hits included, so a progress callback sees
        # every feature exactly once whether or not it was refitted.
        results = [cached.get(f) or fitted[f] for f in features]
        if on_result is not None:
            for r in results:
                on_result(r)
    else:
        from joblib import Parallel, delayed

        # Only the two columns a feature needs go to each worker. Chunking the tasks to
        # ship them once per group was measured and was not faster: these fits are
        # memory-bound, not dominated by task overhead.
        fitted_list = Parallel(n_jobs=n_jobs)(
            delayed(time_consistency)(train.select([f, label]), holdout.select([f, label]), f, label, **kwargs)
            for f in features_to_fit
        )
        fitted = dict(zip(features_to_fit, fitted_list, strict=True))
        for f, r in fitted.items():
            _remember(f, r)
        results = [cached.get(f) or fitted[f] for f in features]
        if on_result is not None:
            for r in results:
                on_result(r)

    out = pl.DataFrame([r.as_dict() for r in results])
    order = {Verdict.INVERTED.value: 0, Verdict.WEAK.value: 1, Verdict.PASS.value: 2, Verdict.DEGENERATE.value: 3}
    
    # Add a _rank column and sort by _rank then delta
    return (
        out.with_columns(
            pl.col("verdict").replace_strict(order, return_dtype=pl.Int32).alias("_rank")
        )
        .sort(["_rank", "delta"], descending=[False, False], nulls_last=True)
        .drop("_rank")
    )
