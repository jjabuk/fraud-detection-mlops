"""Turning LightGBM's ranking scores into probabilities.

A gradient-boosting score orders transactions well and is not a probability.
Every decision rule downstream -- the false-positive budget, the cost matrix,
anything an analyst reads as "87% likely fraud" -- assumes it is one, so the
scores are mapped through an explicit calibrator and the mapping is chosen by
measurement rather than by habit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from fraud_detection.core.config import get_training_params

CALIBRATION_METHODS = ("isotonic", "platt")

# How much ranking a calibrator may destroy before its calibration quality stops counting.
#
# A calibrator relabels scores; it must not reorder them. Isotonic is a step function, and
# a run where it collapsed 59,054 test scores into 93 distinct values cost 0.0211 PR-AUC
# in ties while winning the log-loss comparison by 0.0007.
MAX_PR_AUC_LOSS = get_training_params("training.calibration").get("max_pr_auc_loss", 0.005)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Observed fraud rate against predicted probability, per bin."""

    bin_edges: list[float]
    mean_predicted: list[float]
    observed_rate: list[float]
    bin_count: list[int]

    @property
    def expected_calibration_error(self) -> float:
        """Bin-count-weighted mean gap between predicted and observed.

        Zero means every bucket of predictions came true at exactly the rate
        it claimed. This is the number the reliability diagram shows visually.
        """
        total = sum(self.bin_count)
        if total == 0:
            return 0.0
        return sum(
            count * abs(predicted - observed)
            for count, predicted, observed in zip(
                self.bin_count, self.mean_predicted, self.observed_rate, strict=True
            )
        ) / total


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, method: str):
    """Fits one calibrator. `method` is 'isotonic' or 'platt'."""
    if method == "isotonic":
        # out_of_bounds="clip" so a test score outside the fitted range maps to
        # the nearest endpoint instead of NaN.
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            scores, labels
        )
    if method == "platt":
        return LogisticRegression().fit(scores.reshape(-1, 1), labels)
    raise ValueError(f"Unknown calibration method {method!r}; expected one of {CALIBRATION_METHODS}")


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        return np.asarray(calibrator.predict(scores))
    return np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1])


@dataclass(frozen=True)
class CalibrationChoice:
    """Which calibrator was picked, and both numbers that decided it."""

    method: str
    log_losses: dict[str, float]
    """Cross-validated log loss per method — how well the probabilities are calibrated."""
    pr_auc_losses: dict[str, float]
    """Cross-validated PR-AUC given up against the raw scores — how much ranking each
    method destroys. Positive means the calibrated scores rank worse than the raw ones."""
    budget: float
    disqualified: tuple[str, ...] = ()
    """Methods that calibrated best but broke the ranking budget."""

    @property
    def pr_auc_loss(self) -> float:
        return self.pr_auc_losses[self.method]


def select_calibration_method(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 0,
    max_pr_auc_loss: float = MAX_PR_AUC_LOSS,
) -> CalibrationChoice:
    """Picks isotonic or Platt by cross-validated log loss, subject to a ranking budget.

    Cross-fitting matters here. Isotonic regression is far more flexible than
    a one-parameter sigmoid, so scoring both on the data they were fitted to
    would hand isotonic the win by construction. Held-out folds ask the
    question that matters instead: which mapping generalises.

    Log loss rather than Brier because it punishes confident mistakes harder,
    and a fraud model's confident mistakes are the expensive ones.

    **The ranking budget.** This model has two consumers with different needs: the decision
    rules read the probability, the submission and every ranking metric read the order. A
    criterion looking only at calibration quality will happily choose a calibrator that
    serves the first by wrecking the second, because isotonic's ties are ranking thrown away.

    So: measure both per fold, disqualify anything over the budget, and take the best log
    loss of what survives. If nothing survives, take the least destructive rather than
    failing — a training run has to end with a calibrator, and the choice is recorded.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses: dict[str, list[float]] = {method: [] for method in CALIBRATION_METHODS}
    ranking: dict[str, list[float]] = {method: [] for method in CALIBRATION_METHODS}

    for fit_index, score_index in folds.split(scores, labels):
        held_out_labels = labels[score_index]
        # The raw scores on this fold are the reference: perfect ranking, no calibration.
        raw_pr_auc = float(average_precision_score(held_out_labels, scores[score_index]))
        for method in CALIBRATION_METHODS:
            calibrator = fit_calibrator(scores[fit_index], labels[fit_index], method)
            predicted = apply_calibrator(calibrator, scores[score_index])
            losses[method].append(
                log_loss(held_out_labels, np.clip(predicted, 1e-15, 1 - 1e-15), labels=[0, 1])
            )
            ranking[method].append(
                raw_pr_auc - float(average_precision_score(held_out_labels, predicted))
            )

    mean_loss = {method: float(np.mean(values)) for method, values in losses.items()}
    mean_ranking = {method: float(np.mean(values)) for method, values in ranking.items()}

    affordable = [m for m in CALIBRATION_METHODS if mean_ranking[m] <= max_pr_auc_loss]
    disqualified = tuple(m for m in CALIBRATION_METHODS if m not in affordable)

    if affordable:
        method = min(affordable, key=mean_loss.__getitem__)
    else:
        method = min(CALIBRATION_METHODS, key=mean_ranking.__getitem__)

    return CalibrationChoice(
        method=method,
        log_losses=mean_loss,
        pr_auc_losses=mean_ranking,
        budget=max_pr_auc_loss,
        disqualified=disqualified,
    )


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> ReliabilityCurve:
    """Bins predictions and reports what actually happened in each bin.

    Fixed-width bins on [0, 1] rather than quantile bins: with a 3.5% base
    rate almost every prediction sits near zero, and quantile binning would
    hide that by spreading the low end across most of the chart.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=False keeps 0.0 in the first bin; the final bin absorbs 1.0.
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)

    mean_predicted: list[float] = []
    observed_rate: list[float] = []
    bin_count: list[int] = []
    for bin_number in range(n_bins):
        mask = index == bin_number
        count = int(mask.sum())
        bin_count.append(count)
        mean_predicted.append(float(probabilities[mask].mean()) if count else 0.0)
        observed_rate.append(float(labels[mask].mean()) if count else 0.0)

    return ReliabilityCurve(
        bin_edges=[float(edge) for edge in edges],
        mean_predicted=mean_predicted,
        observed_rate=observed_rate,
        bin_count=bin_count,
    )


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(
            log_loss(labels, np.clip(probabilities, 1e-15, 1 - 1e-15), labels=[0, 1])
        ),
        "expected_calibration_error": reliability_curve(
            probabilities, labels
        ).expected_calibration_error,
    }
