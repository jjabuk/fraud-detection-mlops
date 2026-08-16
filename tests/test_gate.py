"""The promotion gate reads the metrics training actually writes.

This file exists because of a specific failure, and the failure was silent. The
cold-entity check read `test_pr_auc_card_unseen`; `training.model._metrics` writes
`test_pr_auc_unseen`. `dict.get(..., 0.0)` turned that mismatch into the number 0.0, the
floor was also 0.0, and the check reported PASS on every run without ever looking at a
model. It would have passed a model that scored nothing at all on new customers — the
exact population the check exists to protect.

The lesson is not "compare the strings". It is that a check reading a key nobody writes
looks identical, in a log, to a check that passed on merit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from dagster import Failure

from fraud_detection.orchestration.gate_checks import (
    DOMINANT_SEGMENT_SHARE,
    UNSEEN_SEGMENT_METRIC,
    dominant_segment,
    unseen_segment_pr_auc,
)
from fraud_detection.training.data import SplitFrame


def _split(n: int = 200, *, seed: int = 0, unseen_share: float = 0.5) -> SplitFrame:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    return SplitFrame(
        features=pl.DataFrame({"V1": rng.normal(size=n).astype("float32")}),
        labels=pl.Series(labels).cast(pl.Int8),
        amounts=pl.Series(rng.uniform(10, 100, n)),
        seen_in_train=pl.Series(rng.random(n) > unseen_share),
    )


def test_training_writes_the_key_the_gate_reads():
    """The two halves, checked against each other rather than against a literal.

    `_metrics` is the writer and `UNSEEN_SEGMENT_METRIC` is the reader, so this fails the
    moment either side is renamed on its own.
    """
    from fraud_detection.training.model import _metrics

    class _Model:
        test_probabilities = None
        val_probabilities = None
        booster = None
        budget_choice = None
        cost_choice = None
        test_at_threshold = None

    val, test = _split(seed=1), _split(seed=2)

    # A minimal stand-in: only the fields _metrics reads.
    from types import SimpleNamespace

    probabilities = np.clip(np.asarray(test.labels) * 0.6 + 0.2, 0, 1)
    model = SimpleNamespace(
        val_probabilities=np.clip(np.asarray(val.labels) * 0.6 + 0.2, 0, 1),
        test_probabilities=probabilities,
        # Raw booster output, deliberately a different array from the calibrated one:
        # `uncalibrated_test_roc_auc` reads this and `test_roc_auc` reads the other, and a
        # stand-in that used one array for both could not tell the two apart.
        test_scores=np.clip(np.asarray(test.labels) * 0.55 + 0.25, 0, 1),
        booster=SimpleNamespace(best_iteration=10),
        budget_choice=SimpleNamespace(
            threshold=0.5, false_positive_rate=0.01, true_positive_rate=0.5,
            precision=0.6, total_cost=1.0,
        ),
        cost_choice=SimpleNamespace(threshold=0.4, total_cost=0.9),
        test_at_threshold=SimpleNamespace(
            false_positive_rate=0.011, true_positive_rate=0.5, precision=0.6,
        ),
    )

    metrics = _metrics(model, val, test)

    assert UNSEEN_SEGMENT_METRIC in metrics, (
        f"the gate reads {UNSEEN_SEGMENT_METRIC!r}, which training no longer writes; "
        f"available segment keys: {sorted(k for k in metrics if 'seen' in k)}"
    )
    assert unseen_segment_pr_auc(metrics) == metrics[UNSEEN_SEGMENT_METRIC]


def test_a_missing_segment_metric_fails_instead_of_reading_as_zero():
    """Absent is not zero, and the difference is the whole check.

    Defaulting to 0.0 made a model that was never evaluated on cold entities
    indistinguishable, in the gate's own log, from one that had been.
    """
    with pytest.raises(Failure, match=UNSEEN_SEGMENT_METRIC):
        unseen_segment_pr_auc({"test_pr_auc": 0.5})


def test_the_configured_thresholds_are_not_unconditional():
    """A floor of 0.0 is not a floor: every model clears `>= baseline × 0.0`.

    Both knobs sat at 0.0 while MEASUREMENTS.md documented 1.10 and 1.0, so two of the
    gate's four checks passed by construction.
    """
    from fraud_detection.orchestration.assets.gate import (
        MIN_PR_AUC_LIFT,
        MIN_UNSEEN_SEGMENT_RATIO,
    )

    assert MIN_PR_AUC_LIFT > 0.0
    assert MIN_UNSEEN_SEGMENT_RATIO > 0.0


# ---- the dominant-segment check ---------------------------------------------------


def _candidate(**products) -> dict:
    """Metrics carrying one `test_pr_auc_product_*` triple per product."""
    out = {}
    for name, (pr_auc, rows, rate) in products.items():
        out[f"test_pr_auc_product_{name}"] = pr_auc
        out[f"product_{name}_rows"] = rows
        out[f"product_{name}_positive_rate"] = rate
    return out


def test_the_dominant_segment_is_the_one_holding_the_traffic():
    """W is 45,426 of 59,054 test rows at PR-AUC 0.2130; C is 6,481 at 0.7265. The check
    must look at W, which is where the model is weak, not at C, which flatters it."""
    candidate = _candidate(W=(0.2130, 45426.0, 0.0198), C=(0.7265, 6481.0, 0.1467))

    name, pr_auc, _rate, share = dominant_segment(candidate)

    assert name == "W"
    assert pr_auc == 0.2130
    assert share > DOMINANT_SEGMENT_SHARE


def test_no_product_metrics_is_not_a_pass():
    """The failure mode this gate has already been caught in twice: a check that cannot
    find its number and therefore does not run. `None` here becomes a FAIL upstream."""
    assert dominant_segment({"test_pr_auc": 0.53}) is None


def test_a_fragmented_population_has_no_dominant_segment():
    """Nothing holds a majority, so this check has no business failing the run."""
    candidate = _candidate(A=(0.5, 1000.0, 0.03), B=(0.5, 1000.0, 0.03), C=(0.5, 1000.0, 0.03))

    assert dominant_segment(candidate) is None


def test_the_bar_is_relative_so_a_harder_segment_is_not_punished_for_being_harder():
    """W's 0.2130 over a 0.0198 base rate is a lift of 10.75; C's 0.7265 over 0.1467 is
    4.95. An absolute PR-AUC floor would rank these the wrong way round."""
    w = _candidate(W=(0.2130, 45426.0, 0.0198))
    c = _candidate(C=(0.7265, 45426.0, 0.1467))

    _, w_pr, w_rate, _ = dominant_segment(w)
    _, c_pr, c_rate, _ = dominant_segment(c)

    assert w_pr < c_pr           # W looks worse on the raw number ...
    assert w_pr / w_rate > c_pr / c_rate  # ... and is better against its own floor
