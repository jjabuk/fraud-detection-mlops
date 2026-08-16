from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.training.threshold import (
    evaluate_threshold,
    threshold_for_false_positive_budget,
    total_cost,
)

# 1000 transactions, 5% fraud, a model that ranks fraud above legitimate with
# some overlap -- close enough to the real shape to make the arithmetic mean
# something.
RNG = np.random.default_rng(7)
LABELS = (RNG.random(1000) < 0.05).astype(int)
PROBABILITIES = np.clip(RNG.normal(np.where(LABELS == 1, 0.6, 0.15), 0.12), 0.0, 1.0)
AMOUNTS = RNG.uniform(10.0, 500.0, size=1000)

def test_budget_threshold_respects_the_false_positive_budget():
    for budget in (0.005, 0.01, 0.02, 0.05):
        choice = threshold_for_false_positive_budget(
            PROBABILITIES, LABELS, AMOUNTS, budget=budget
        )
        # The budget is a promise, not a target to land near. At most means at
        # most.
        assert choice.false_positive_rate <= budget



def test_a_tighter_budget_blocks_less_and_catches_less():
    strict = threshold_for_false_positive_budget(PROBABILITIES, LABELS, AMOUNTS, budget=0.005)
    loose = threshold_for_false_positive_budget(PROBABILITIES, LABELS, AMOUNTS, budget=0.05)

    assert strict.threshold > loose.threshold
    assert strict.blocked_count < loose.blocked_count
    assert strict.true_positive_rate <= loose.true_positive_rate

def test_budget_threshold_is_the_lowest_one_that_fits():
    # Among thresholds respecting the budget, the lowest catches the most
    # fraud. Spending less of an agreed budget is recall left on the table,
    # not prudence -- so nudging the threshold down must break the budget.
    budget = 0.02
    choice = threshold_for_false_positive_budget(PROBABILITIES, LABELS, AMOUNTS, budget=budget)

    lower = evaluate_threshold(PROBABILITIES, LABELS, AMOUNTS, choice.threshold - 0.05)

    assert lower.false_positive_rate > budget

def test_cost_counts_missed_fraud_at_full_amount():
    # Threshold above every score: nothing is blocked, so the cost is exactly
    # the money lost to fraud and nothing else.
    cost = total_cost(PROBABILITIES, LABELS, AMOUNTS, threshold=1.1)

    assert cost == pytest.approx(AMOUNTS[LABELS == 1].sum())

def test_cost_counts_blocked_legitimate_transactions_at_a_flat_rate():
    # Threshold below every score: everything is blocked, no fraud is missed,
    # so the cost is purely the review cost of the legitimate transactions.
    cost = total_cost(
        PROBABILITIES, LABELS, AMOUNTS, threshold=-0.1, false_positive_cost=5.0
    )

    assert cost == pytest.approx(int((LABELS == 0).sum()) * 5.0)



def test_evaluate_threshold_confusion_counts_add_up():
    choice = evaluate_threshold(PROBABILITIES, LABELS, AMOUNTS, threshold=0.4)

    assert choice.caught_fraud_count + choice.missed_fraud_count == int(LABELS.sum())
    assert choice.blocked_count == int((PROBABILITIES >= 0.4).sum())
    assert choice.detail["fraud_amount_caught"] + choice.detail["fraud_amount_missed"] == (
        pytest.approx(AMOUNTS[LABELS == 1].sum())
    )

