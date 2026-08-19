"""Choosing the score above which a transaction is blocked.

Not `argmax F1`. F1 treats a missed fraud and a blocked customer as equally
bad, which no payments business believes: one is a chargeback, the other is a
person whose card was declined at a checkout. The threshold here comes from a
stated budget for the second kind of error, and is cross-checked against an
explicit cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Block at most this share of legitimate transactions. A business input, not a
# statistical one: it says how much customer friction the operation will
# tolerate, and everything else follows from it.
from fraud_detection.config import get_training_params

_t_config = get_training_params("training.threshold")
DEFAULT_FALSE_POSITIVE_BUDGET = _t_config["default_false_positive_budget"]

# Cost model, in units of the transaction's own currency.
#
# A missed fraud costs its full amount -- the money is gone and charged back.
# A blocked legitimate transaction costs a flat review-and-goodwill figure
# rather than its amount: the sale is usually recoverable, the handling is
# not. Both numbers are assumptions, stated here so they can be argued with
# instead of buried in a comparison.
DEFAULT_FALSE_POSITIVE_COST = _t_config["default_false_positive_cost"]

# How many quantiles of the score distribution the cost search evaluates. Shared
# with `plots.cost_curve_figure`, so the curve anyone reads is the same grid the
# optimum was chosen from rather than a second, prettier sampling of it.
COST_CANDIDATE_COUNT = 200


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    false_positive_rate: float
    true_positive_rate: float
    precision: float
    blocked_count: int
    caught_fraud_count: int
    missed_fraud_count: int
    total_cost: float
    detail: dict[str, float] = field(default_factory=dict)


def _confusion(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    blocked = probabilities >= threshold
    true_positive = int(np.sum(blocked & (labels == 1)))
    false_positive = int(np.sum(blocked & (labels == 0)))
    false_negative = int(np.sum(~blocked & (labels == 1)))
    true_negative = int(np.sum(~blocked & (labels == 0)))
    return true_positive, false_positive, false_negative, true_negative


def total_cost(
    probabilities: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> float:
    """Cost of operating at `threshold`: missed fraud plus blocked customers."""
    blocked = probabilities >= threshold
    missed_fraud = float(np.sum(amounts[(~blocked) & (labels == 1)]))
    blocked_legitimate = int(np.sum(blocked & (labels == 0)))
    return missed_fraud + blocked_legitimate * false_positive_cost


def threshold_for_false_positive_budget(
    probabilities: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    *,
    budget: float = DEFAULT_FALSE_POSITIVE_BUDGET,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> ThresholdChoice:
    """Lowest threshold whose false-positive rate still fits the budget.

    Lowest, not any: among the thresholds that respect the budget, the lowest
    one blocks the most fraud. Spending less of an agreed budget is not a
    virtue, it is recall left on the table.

    Searched over the realised rates rather than taken as the (1 - budget)
    quantile of legitimate scores. The quantile is only equivalent when scores
    are distinct, and after isotonic calibration they are emphatically not:
    isotonic is a step function, so hundreds of rows share an identical
    probability. The quantile then lands on a repeated value and `>=` admits
    the whole tied group, overshooting the budget. Measured on real data
    -- a 1% budget realising 1.07%.
    """
    legitimate = probabilities[labels == 0]
    if legitimate.size == 0:
        raise ValueError("No legitimate transactions, so a false-positive budget is undefined.")

    sorted_legitimate = np.sort(legitimate)
    candidates = np.unique(probabilities)
    # False-positive rate at each candidate: the share of legitimate scores at
    # or above it.
    above = legitimate.size - np.searchsorted(sorted_legitimate, candidates, side="left")
    realised = above / legitimate.size

    qualifying = np.flatnonzero(realised <= budget)
    if qualifying.size == 0:
        raise ValueError(
            f"No threshold satisfies a false-positive budget of {budget}; the lowest "
            f"achievable rate is {realised.min():.4f}."
        )

    threshold = float(candidates[qualifying[0]])
    return evaluate_threshold(
        probabilities,
        labels,
        amounts,
        threshold,
        false_positive_cost=false_positive_cost,
    )


def cost_minimising_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    *,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    n_candidates: int = COST_CANDIDATE_COUNT,
) -> ThresholdChoice:
    """The threshold the cost model alone would pick.

    Reported next to the budget-driven one rather than instead of it. Where
    the two disagree, the gap is the price of the friction policy -- which is
    a number worth putting in front of whoever set the budget.
    """
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, n_candidates)))
    costs = [
        total_cost(
            probabilities, labels, amounts, threshold, false_positive_cost=false_positive_cost
        )
        for threshold in candidates
    ]
    best = float(candidates[int(np.argmin(costs))])
    return evaluate_threshold(
        probabilities, labels, amounts, best, false_positive_cost=false_positive_cost
    )


def evaluate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> ThresholdChoice:
    true_positive, false_positive, false_negative, true_negative = _confusion(
        probabilities, labels, threshold
    )
    blocked = true_positive + false_positive
    negatives = false_positive + true_negative
    positives = true_positive + false_negative

    return ThresholdChoice(
        threshold=threshold,
        false_positive_rate=false_positive / negatives if negatives else 0.0,
        true_positive_rate=true_positive / positives if positives else 0.0,
        precision=true_positive / blocked if blocked else 0.0,
        blocked_count=blocked,
        caught_fraud_count=true_positive,
        missed_fraud_count=false_negative,
        total_cost=total_cost(
            probabilities, labels, amounts, threshold, false_positive_cost=false_positive_cost
        ),
        detail={
            "fraud_amount_missed": float(
                np.sum(amounts[(probabilities < threshold) & (labels == 1)])
            ),
            "fraud_amount_caught": float(
                np.sum(amounts[(probabilities >= threshold) & (labels == 1)])
            ),
        },
    )
