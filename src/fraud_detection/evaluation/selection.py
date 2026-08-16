"""Two more ways to shrink a feature set, and what each one costs.

The published solutions list several selection techniques. Two are implemented here
because they answer questions the other modules do not:

* **PCA per family** — the alternative to picking a representative. Instead of keeping one
  column out of eleven, keep a component built from all eleven. Fewer inputs, no column
  thrown away, and an output nobody can interpret.
* **Forward selection** — grow a feature set one column at a time, keeping whichever
  column improves a holdout score most, and stop when the improvement stops paying.

Both are **score-chasing** rather than production safety, and the distinction matters.
:mod:`~fraud_detection.evaluation.time_consistency` rejects a feature because it will
break; these two drop a feature because it is not pulling its weight *today*, on *this*
holdout. That is a weaker claim, it does not transfer to next month's data, and a
selection produced this way should be re-derived when the data changes rather than pinned
forever.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from fraud_detection.core.feature_contract import Fragment, Rejection

__all__ = [
    "SELECTION_PARAMS",
    "ForwardSelection",
    "PcaGroups",
    "forward_selection",
    "from_forward_selection",
    "pca_groups",
]


@dataclass
class PcaGroups:
    """Components fitted per group, and how much of each group they kept."""

    components: pl.DataFrame
    explained: pl.DataFrame

    @property
    def n_inputs(self) -> int:
        return int(self.explained.get_column("columns").sum())

    @property
    def n_outputs(self) -> int:
        return self.components.shape[1]


def pca_groups(
    df: pl.DataFrame,
    groups: Sequence[Sequence[str]],
    *,
    n_components: int = 1,
    seed: int = 0,
) -> PcaGroups:
    """Replace each group with its leading principal component(s).

    Nulls are filled with the column's **training median** before fitting, because PCA has
    no notion of missing. That is a real cost: in this dataset missingness is itself
    signal, and imputing it away discards information that a tree would have used. Where
    a whole group is absent for a row, the component is the median row rather than a gap
    the model can branch on.

    ``explained`` reports, per group, how much variance the components retained. A group
    whose first component holds most of its variance was genuinely one signal; a group
    where it does not was never a group.
    """
    comps: dict[str, pl.Series] = {}
    rows = []
    
    cols_set = set(df.columns)

    for i, group in enumerate(groups):
        members = [c for c in group if c in cols_set]
        if not members:
            continue

        block = df.select(members).cast(pl.Float64)
        
        exprs = []
        for c in members:
            exprs.append(pl.col(c).fill_null(pl.col(c).median()).fill_null(0.0).alias(c))
        filled = block.select(exprs)

        scaled_exprs = []
        for c in members:
            mean = pl.col(c).mean()
            std = pl.when(pl.col(c).std() == 0.0).then(1.0).otherwise(pl.col(c).std())
            scaled_exprs.append(((pl.col(c) - mean) / std).alias(c))
        scaled = filled.select(scaled_exprs)

        k = min(n_components, len(members))
        pca = PCA(n_components=k, random_state=seed)
        transformed = pca.fit_transform(scaled.to_numpy())

        name = f"pca_{i:03d}_{members[0]}"
        for j in range(k):
            comps[f"{name}_{j}" if k > 1 else name] = pl.Series(transformed[:, j])
            
        null_share = block.select(pl.all().null_count() / pl.len()).mean_horizontal()[0]

        rows.append({
            "group": "+".join(members),
            "columns": len(members),
            "components": k,
            "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "null_share": round(float(null_share), 4),
        })

    return PcaGroups(pl.DataFrame(comps), pl.DataFrame(rows))


SELECTION_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 120,
    "num_leaves": 31,
    "learning_rate": 0.08,
    "verbose": -1,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
}


@dataclass
class ForwardSelection:
    """The path a greedy search took, not just where it stopped."""

    steps: pl.DataFrame
    selected: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    stopped_because: str = ""

    @property
    def auc(self) -> float:
        return float(self.steps.get_column("auc").to_list()[-1]) if len(self.steps) else float("nan")


def _fit_auc(train, holdout, features, label, params, seed) -> float:
    model = lgb.LGBMClassifier(**{**SELECTION_PARAMS, **(params or {}), "random_state": seed})
    # Polars -> numpy to avoid Arrow segfaults on Mac during LightGBM fit
    X_train = train.select(features).to_numpy()
    y_train = train.get_column(label).to_numpy()
    X_holdout = holdout.select(features).to_numpy()
    y_holdout = holdout.get_column(label).to_numpy()
    
    model.fit(X_train, y_train)
    return float(roc_auc_score(y_holdout, model.predict_proba(X_holdout)[:, 1]))


def forward_selection(
    train: pl.DataFrame,
    holdout: pl.DataFrame,
    candidates: Sequence[str],
    label: str,
    *,
    max_features: int = 20,
    min_gain: float = 0.0005,
    params: dict | None = None,
    n_jobs: int = 1,
    seed: int = 0,
) -> ForwardSelection:
    """Grow a feature set greedily, stopping when the next column stops paying.

    Cost is quadratic: selecting *k* features from *n* candidates fits roughly ``k·n``
    models. That is why ``max_features`` and ``min_gain`` are required rather than
    optional — an unbounded forward selection over a few hundred columns is not a check
    you run, it is a weekend.

    ``min_gain`` is the honest part of the procedure. Greedy search will always find
    *some* column that nudges the holdout score, so without a floor it selects noise and
    reports it as improvement. The stopping reason is recorded so a reader can tell a
    search that converged from one that simply hit its cap.
    """
    remaining = [c for c in candidates if c in train.columns]
    selected: list[str] = []
    rows = []
    best_auc = 0.5
    stopped = "reached max_features"

    while remaining and len(selected) < max_features:
        trials = [(c, selected + [c]) for c in remaining]

        if n_jobs == 1:
            scores = [_fit_auc(train, holdout, f, label, params, seed) for _, f in trials]
        else:
            from joblib import Parallel, delayed

            scores = Parallel(n_jobs=n_jobs)(
                delayed(_fit_auc)(train.select([*f, label]), holdout.select([*f, label]), f, label, params, seed)
                for _, f in trials
            )

        best_i = int(np.argmax(scores))
        column, auc = trials[best_i][0], scores[best_i]
        gain = auc - best_auc

        if gain < min_gain:
            stopped = f"gain {gain:+.5f} below min_gain {min_gain}"
            break

        selected.append(column)
        remaining.remove(column)
        best_auc = auc
        rows.append({"step": len(selected), "column": column, "auc": round(auc, 5), "gain": round(gain, 5)})

    return ForwardSelection(
        steps=pl.DataFrame(rows),
        selected=selected,
        rejected=[c for c in candidates if c in train.columns and c not in selected],
        stopped_because=stopped,
    )


def from_forward_selection(
    result: ForwardSelection, *, params: dict | None = None, qualification: dict | None = None
) -> Fragment:
    """Turn a selection into a contract fragment.

    Everything the search did not reach is rejected. Note what this claims and what it
    does not: a rejected column failed to add measurable value *on one holdout, given the
    columns already chosen*. It is not evidence the column is harmful, and a different
    starting order could have kept it.
    """
    return Fragment(
        check="forward_selection",
        rejections=tuple(
            Rejection(column=c, by="forward_selection", value=None, unit="column")
            for c in result.rejected
        ),
        params={"stopped_because": result.stopped_because, **(params or {})},
        qualification={"final_auc": result.auc, "selected": len(result.selected), **(qualification or {})},
        tool="fraud_detection.evaluation.selection",
    )
