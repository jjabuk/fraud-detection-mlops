"""The checks a model must clear before it can be promoted.

Pure predicates over a metrics dict — no bucket, no registry, no orchestrator. The asset in
`orchestration/assets/gate.py` reads a candidate's metrics, runs these, and either writes
the promotion marker or fails the run.

The rule they all follow: **absent is not zero.** A check that cannot find its number fails
loudly rather than defaulting into a comparison it would always pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from dagster import Failure

from fraud_detection.core.config import get_orchestration_params

_gate_cfg = get_orchestration_params("gate")

__all__ = [
    "DOMINANT_SEGMENT_SHARE",
    "MIN_DOMINANT_SEGMENT_LIFT",
    "Check",
    "dominant_segment",
    "unseen_segment_pr_auc",
]

UNSEEN_SEGMENT_METRIC = "test_pr_auc_unseen"

MAX_FALSE_POSITIVE_DRIFT = _gate_cfg["max_false_positive_drift"]


# A *relative* bar. `ProductCD == "W"` runs at a 0.0198 positive rate against `C`'s 0.1467,
# so an absolute PR-AUC floor would fail the harder segment for being harder and pass the
# easier one for being easy.
MIN_DOMINANT_SEGMENT_LIFT = _gate_cfg.get("min_dominant_segment_lift", 5.0)

# Below this share a segment is not "where the traffic is" and this check has no business
# failing a run over it.
DOMINANT_SEGMENT_SHARE = _gate_cfg.get("dominant_segment_share", 0.50)


def dominant_segment(candidate: dict) -> tuple[str, float, float, float] | None:
    """The product segment holding most of the scored rows, if any holds a majority.

    Returns `(name, pr_auc, positive_rate, share)`. `None` means training wrote no product
    metrics, and the caller turns that into a failure rather than a pass — a check that
    cannot find its number must not silently skip.
    """
    rows = {
        key[len("test_pr_auc_product_") :]: value
        for key, value in candidate.items()
        if key.startswith("test_pr_auc_product_")
    }
    if not rows:
        return None

    counts = {name: float(candidate.get(f"product_{name}_rows", 0.0)) for name in rows}
    total = sum(counts.values())
    if total <= 0:
        return None

    name = max(counts, key=counts.__getitem__)
    share = counts[name] / total
    if share < DOMINANT_SEGMENT_SHARE:
        return None
    return name, float(rows[name]), float(candidate.get(f"product_{name}_positive_rate", 0.0)), share


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def unseen_segment_pr_auc(candidate: dict) -> float:
    """The cold-entity score, or a failure if training never measured it.

    **Absent is not zero.** A missing metric means the segment was never evaluated, which is
    a reason to look rather than a number to compare against.
    """
    if UNSEEN_SEGMENT_METRIC not in candidate:
        raise Failure(
            f"The candidate's metrics carry no {UNSEEN_SEGMENT_METRIC}, so the cold-entity "
            "segment was never scored and this gate cannot judge it. Training writes the "
            "key only when the segment has both rows and positives."
        )
    return float(candidate[UNSEEN_SEGMENT_METRIC])
