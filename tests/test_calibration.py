from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.training.calibration import (
    apply_calibrator,
    calibration_metrics,
    fit_calibrator,
    reliability_curve,
    select_calibration_method,
)

RNG = np.random.default_rng(0)


def _miscalibrated_scores(n: int = 4000, base_rate: float = 0.035):
    """Scores that rank well but are systematically overconfident.

    This is what an uncalibrated gradient-boosting model trained with
    scale_pos_weight actually looks like: the ordering is informative, the
    numbers are not probabilities.
    """
    labels = (RNG.random(n) < base_rate).astype(int)
    scores = np.clip(
        RNG.normal(loc=np.where(labels == 1, 0.75, 0.25), scale=0.15), 0.001, 0.999
    )
    return scores, labels


def test_calibration_moves_predictions_towards_the_observed_rate():
    scores, labels = _miscalibrated_scores()

    calibrator = fit_calibrator(scores, labels, "isotonic")
    calibrated = apply_calibrator(calibrator, scores)

    # The raw scores average far above the true fraud rate; calibrated ones
    # must average close to it, because that is what calibration means.
    assert scores.mean() > labels.mean() * 3
    assert calibrated.mean() == pytest.approx(labels.mean(), abs=0.01)


def test_calibrated_probabilities_stay_in_the_unit_interval():
    scores, labels = _miscalibrated_scores()

    for method in ("isotonic", "platt"):
        calibrated = apply_calibrator(fit_calibrator(scores, labels, method), scores)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0


def test_isotonic_clips_scores_outside_the_fitted_range():
    # A test score below or above anything seen during calibration must map to
    # the nearest endpoint. Without out_of_bounds="clip" it becomes NaN, and a
    # NaN probability silently poisons every metric downstream.
    scores, labels = _miscalibrated_scores()
    calibrator = fit_calibrator(scores, labels, "isotonic")

    extreme = apply_calibrator(calibrator, np.array([-5.0, 5.0]))

    assert not np.isnan(extreme).any()
    assert 0.0 <= extreme[0] <= 1.0
    assert 0.0 <= extreme[1] <= 1.0


def test_method_selection_is_cross_fitted_not_in_sample():
    # Isotonic is far more flexible than a sigmoid, so scoring both on the
    # data they were fitted to would hand it the win by construction. The
    # selection has to report a loss per method from held-out folds.
    scores, labels = _miscalibrated_scores()

    # A generous budget, so this test still asks only the question it was written to ask:
    # that the losses come from held-out folds and the best one wins. The budget's own
    # behaviour is tested separately below.
    choice = select_calibration_method(scores, labels, n_splits=3, max_pr_auc_loss=1.0)
    losses = choice.log_losses

    assert choice.method in {"isotonic", "platt"}
    assert set(losses) == {"isotonic", "platt"}
    assert all(loss > 0 for loss in losses.values())
    assert losses[choice.method] == min(losses.values())


def test_a_calibrator_that_destroys_ranking_is_disqualified():
    """The rule that would have caught the 2026-08-16 isotonic run.

    A zero budget disqualifies anything that gives up any ranking at all. Isotonic is a
    step function and always gives up some, so under a zero budget it can only be chosen
    when Platt is worse — which is what the fallback branch is for.
    """
    scores, labels = _miscalibrated_scores()

    strict = select_calibration_method(scores, labels, n_splits=3, max_pr_auc_loss=0.0)

    assert "isotonic" in strict.disqualified
    assert strict.pr_auc_losses["isotonic"] > 0.0
    # Either Platt survived the budget and was chosen, or nothing did and the least
    # destructive method was chosen. Both are the documented behaviour; a method that was
    # disqualified while an affordable one existed is not.
    if "platt" not in strict.disqualified:
        assert strict.method == "platt"
    else:
        assert strict.pr_auc_loss == min(strict.pr_auc_losses.values())


def test_the_budget_is_read_from_config_by_default():
    """Not a hardcoded constant: the number is a policy dial, so it lives in config."""
    from fraud_detection.training.calibration import MAX_PR_AUC_LOSS

    scores, labels = _miscalibrated_scores()
    assert select_calibration_method(scores, labels, n_splits=3).budget == MAX_PR_AUC_LOSS


def test_reliability_curve_bins_sum_to_the_population():
    scores, labels = _miscalibrated_scores()
    calibrated = apply_calibrator(fit_calibrator(scores, labels, "isotonic"), scores)

    curve = reliability_curve(calibrated, labels, n_bins=10)

    assert sum(curve.bin_count) == len(labels)
    assert len(curve.mean_predicted) == len(curve.observed_rate) == 10
    assert len(curve.bin_edges) == 11


def test_perfect_calibration_has_zero_expected_error():
    # Predictions of exactly 0 and exactly 1 that always come true: the
    # observed rate matches the prediction in every occupied bin.
    probabilities = np.array([0.0] * 50 + [1.0] * 50)
    labels = np.array([0] * 50 + [1] * 50)

    assert reliability_curve(probabilities, labels).expected_calibration_error == pytest.approx(0.0)


def test_expected_calibration_error_catches_systematic_overconfidence():
    # Everything predicted at 0.9, nothing ever happens: the gap is 0.9 and
    # the metric has to say so. A model like this can still rank perfectly,
    # which is why ranking metrics alone cannot be trusted for thresholds.
    probabilities = np.full(100, 0.9)
    labels = np.zeros(100, dtype=int)

    assert reliability_curve(probabilities, labels).expected_calibration_error == pytest.approx(
        0.9, abs=1e-6
    )


def test_calibration_metrics_reports_all_three():
    scores, labels = _miscalibrated_scores()
    calibrated = apply_calibrator(fit_calibrator(scores, labels, "platt"), scores)

    metrics = calibration_metrics(calibrated, labels)

    assert set(metrics) == {"brier", "log_loss", "expected_calibration_error"}
    assert all(value >= 0 for value in metrics.values())


def test_unknown_method_is_rejected():
    scores, labels = _miscalibrated_scores(n=100)

    with pytest.raises(ValueError, match="Unknown calibration method"):
        fit_calibrator(scores, labels, "sigmoid-ish")
