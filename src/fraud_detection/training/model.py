"""Train, calibrate, and pick a threshold — as a function, not as an orchestrator step.

This is the whole modelling recipe: the LightGBM fit, the calibrator that turns its scores
into probabilities, and the cut point those probabilities are read against. It takes three
frames and returns everything a caller needs; it reads nothing, writes nothing, and knows
about neither Dagster nor a cloud.

That is the point. A notebook can call ``train_lightgbm`` on a sample and get the same
model the pipeline produces, and the Dagster asset that wraps it can be about what an asset
should be about — loading inputs, storing artifacts, recording the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from fraud_detection.training.calibration import (
    ReliabilityCurve,
    apply_calibrator,
    calibration_metrics,
    fit_calibrator,
    reliability_curve,
    select_calibration_method,
)
from fraud_detection.training.data import SplitFrame, align_categories, to_lightgbm
from fraud_detection.training.threshold import (
    ThresholdChoice,
    cost_minimising_threshold,
    evaluate_threshold,
    threshold_for_false_positive_budget,
)

__all__ = [
    "EARLY_STOPPING_ROUNDS",
    "LIGHTGBM_PARAMS",
    "NUM_BOOST_ROUND",
    "TrainedModel",
    "train_lightgbm",
]

# scale_pos_weight rather than resampling: at a ~3.5% positive rate, undersampling throws
# away 96% of the honest transactions the model needs in order to know what honest looks
# like, and oversampling duplicates the frauds it is most likely to overfit. Reweighting
# keeps every row and only changes how loudly each counts. The value is computed from the
# training split rather than pinned, so it stays correct if the split boundaries move.
#
# deterministic/force_row_wise/num_threads exist so two runs over identical input produce
# identical numbers. LightGBM's default multithreaded histogram construction is not
# bit-reproducible, and observed run-to-run movement was ~0.003 PR-AUC -- the same size as
# several findings this project has recorded. A comparison that cannot distinguish a real
# effect from thread scheduling is not a comparison.
from fraud_detection.config import get_training_params

LIGHTGBM_PARAMS = get_training_params("training.lightgbm")
PRODUCT_COLUMN = "ProductCD"
"""The segment axis the gate checks. A constant here rather than config because the
metric *keys* it produces are read by name in `assets/gate.py`, and a configurable key is
a key nothing can be relied on to write."""

NUM_BOOST_ROUND = get_training_params("training.lightgbm")["num_boost_round"]
EARLY_STOPPING_ROUNDS = get_training_params("training.lightgbm")["early_stopping_rounds"]


@dataclass
class TrainedModel:
    """A model, the mapping that makes its scores probabilities, and the cut point.

    One object because they are one decision. Shipping any of the three without the others
    produces something that cannot be used to decline a payment.
    """

    booster: lgb.Booster
    calibrator: object
    calibration_method: str
    calibration_losses: dict[str, float]
    feature_names: list[str]
    scale_pos_weight: float

    budget_choice: ThresholdChoice
    cost_choice: ThresholdChoice
    test_at_threshold: ThresholdChoice

    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    test_scores: np.ndarray
    """The raw booster output, before calibration. Kept because it is what the Kaggle
    submission carries and what every ranking metric should be read from -- see the note
    on `uncalibrated_test_roc_auc` in `_metrics`."""
    uncalibrated_test_pr_auc: float
    reliability: ReliabilityCurve
    metrics: dict[str, float] = field(default_factory=dict)
    search_history: list[dict] = field(default_factory=list)

    # The full CalibrationChoice — both criteria and anything the ranking budget
    # disqualified. Defaulted so a hand-built stand-in in a test need not supply it.
    calibration_choice: object = None

    # The contract this model was trained against, stamped by the caller after the fit.
    # `train_lightgbm` neither knows nor needs to know what a contract is; what matters is
    # that the fingerprint travels with the model, because a check comparing a contract
    # against itself passes unconditionally and proves nothing.
    contract_fingerprint: str = ""

    @property
    def threshold(self) -> float:
        return self.budget_choice.threshold

    def as_artifact(self) -> dict:
        """What has to travel together for the model to be servable -- and auditable.

        `metrics` is not needed to score a transaction, and is here anyway: this dict is
        what lands in the model registry, and a registry entry that does not carry the
        numbers it was promoted on cannot be audited after the fact. It is also what the
        blocking checks read, since the IO manager hands consumers this dict rather than
        the TrainedModel it was built from.
        """
        return {
            "booster": self.booster,
            "calibrator": self.calibrator,
            "calibration_method": self.calibration_method,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "contract_fingerprint": self.contract_fingerprint,
            "metrics": dict(self.metrics),
        }


def train_lightgbm(
    train: SplitFrame,
    val: SplitFrame,
    test: SplitFrame,
    *,
    search_space: dict | None = None,
    n_iter: int = 1,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    seed: int | None = None,
) -> TrainedModel:
    """Fit, calibrate, choose a threshold, and score the test split once.

    **Split discipline**, which is the only reason the numbers mean anything: the model
    fits on ``train``. Early stopping and the calibrator both use ``val``. The threshold is
    chosen on ``val`` too. ``test`` is touched exactly once, at the end, to report numbers
    nothing was selected on.

    ``seed`` overrides the seed in `config/training.toml`. The pipeline never passes it —
    production training is deterministic and pinned — but characterising the run-to-run
    noise band means refitting the same procedure with the seed as the only thing that
    moves, and a measurement that had to edit a config file to run would not be one anybody
    reruns. Bagging and feature sampling are what it perturbs.

    Raises ``ValueError`` rather than a Dagster failure — the caller decides what a failure
    looks like in its own runtime.
    """
    positives = int(train.labels.sum())
    if positives == 0:
        raise ValueError("Training split contains no fraud cases; nothing to learn.")
    scale_pos_weight = (len(train) - positives) / positives

    val_features = align_categories(val.features, train.features)
    test_features = align_categories(test.features, train.features)

    # Through `to_lightgbm`: LightGBM cannot read a polars categorical over Arrow.
    train_ds = lgb.Dataset(to_lightgbm(train.features), label=train.labels)
    val_ds = lgb.Dataset(to_lightgbm(val_features), label=val.labels)

    if search_space:
        import logging
        logging.getLogger().info("Running Random Search with %d iterations...", n_iter)
        sampled_params_list = list(ParameterSampler(search_space, n_iter=n_iter, random_state=42))
    else:
        # Fallback to single iteration with empty params
        sampled_params_list = [{}]

    best_booster = None
    best_val_pr_auc = -1.0
    best_params = {}
    search_history = []

    for idx, sample_params in enumerate(sampled_params_list):
        current_params = {**LIGHTGBM_PARAMS, **sample_params, "scale_pos_weight": scale_pos_weight}
        if seed is not None:
            # `seed` is LightGBM's master seed; the per-purpose ones are derived from it
            # unless set explicitly, and none of them are in LIGHTGBM_PARAMS.
            current_params["seed"] = seed
        booster = lgb.train(
            current_params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[val_ds],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_scores = np.asarray(booster.predict(to_lightgbm(val_features), num_iteration=booster.best_iteration))
        val_pr_auc = average_precision_score(val.labels.to_numpy(), val_scores)
        
        search_history.append({
            "val_pr_auc": val_pr_auc,
            "best_iteration": booster.best_iteration,
            **sample_params
        })
        
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_booster = booster
            best_params = sample_params
            
    if search_space:
        import logging
        logging.getLogger().info("Best val PR-AUC: %.4f with params: %s", best_val_pr_auc, best_params)

    booster = best_booster
    val_scores = np.asarray(booster.predict(to_lightgbm(val_features), num_iteration=booster.best_iteration))
    test_scores = np.asarray(booster.predict(to_lightgbm(test_features), num_iteration=booster.best_iteration))

    calibration = select_calibration_method(val_scores, val.labels.to_numpy())
    method, losses = calibration.method, calibration.log_losses
    calibrator = fit_calibrator(val_scores, val.labels.to_numpy(), method)
    val_probabilities = apply_calibrator(calibrator, val_scores)
    test_probabilities = apply_calibrator(calibrator, test_scores)

    budget_choice = threshold_for_false_positive_budget(
        val_probabilities, val.labels.to_numpy(), val.amounts.to_numpy()
    )
    cost_choice = cost_minimising_threshold(
        val_probabilities, val.labels.to_numpy(), val.amounts.to_numpy()
    )
    # On val the budget holds by construction, so only this says whether the policy
    # survives being carried into the next period.
    test_at_threshold = evaluate_threshold(
        test_probabilities,
        test.labels.to_numpy(),
        test.amounts.to_numpy(),
        budget_choice.threshold,
    )

    model = TrainedModel(
        booster=booster,
        calibrator=calibrator,
        calibration_method=method,
        calibration_losses=losses,
        feature_names=list(train.features.columns),
        scale_pos_weight=scale_pos_weight,
        budget_choice=budget_choice,
        cost_choice=cost_choice,
        test_at_threshold=test_at_threshold,
        val_probabilities=val_probabilities,
        test_probabilities=test_probabilities,
        test_scores=test_scores,
        uncalibrated_test_pr_auc=float(average_precision_score(test.labels, test_scores)),
        reliability=reliability_curve(test_probabilities, test.labels.to_numpy()),
        search_history=search_history,
        calibration_choice=calibration,
    )
    model.metrics = _metrics(model, val, test)
    return model


def _metrics(model: TrainedModel, val: SplitFrame, test: SplitFrame) -> dict[str, float]:
    labels = test.labels.to_numpy()
    probabilities = model.test_probabilities

    metrics: dict[str, float] = {
        "val_pr_auc": float(average_precision_score(val.labels, model.val_probabilities)),
        "test_pr_auc": float(average_precision_score(test.labels, probabilities)),
        "test_roc_auc": float(roc_auc_score(test.labels, probabilities)),
        # On the *raw* scores: that is what the submission carries and what any ranking
        # consumer reads. `test_roc_auc` above uses the calibrated probabilities, and a step
        # calibrator creates ties that cost ranking -- so the two can differ materially.
        "uncalibrated_test_roc_auc": float(roc_auc_score(test.labels, model.test_scores)),
        "test_positive_rate": float(test.labels.mean()),
        "best_iteration": float(model.booster.best_iteration),
        "threshold": model.budget_choice.threshold,
        "threshold_false_positive_rate": model.budget_choice.false_positive_rate,
        "threshold_recall": model.budget_choice.true_positive_rate,
        "threshold_precision": model.budget_choice.precision,
        "cost_at_budget_threshold": model.budget_choice.total_cost,
        "cost_at_cost_minimising_threshold": model.cost_choice.total_cost,
        "cost_minimising_threshold": model.cost_choice.threshold,
        "test_threshold_false_positive_rate": model.test_at_threshold.false_positive_rate,
        "test_threshold_recall": model.test_at_threshold.true_positive_rate,
        "test_threshold_precision": model.test_at_threshold.precision,
        **{f"test_{k}": v for k, v in calibration_metrics(probabilities, labels).items()},
    }

    # Segment the headline number by whether the client was ever seen in training. 
    # This evaluates the cold-entity case the model meets every time a genuinely new customer arrives.
    seen = test.seen_in_train.to_numpy()
    for segment, mask in (("seen", seen), ("unseen", ~seen)):
        if mask.sum() and labels[mask].sum():
            metrics[f"test_pr_auc_{segment}"] = float(
                average_precision_score(labels[mask], probabilities[mask])
            )
            metrics[f"{segment}_rows"] = float(mask.sum())

    # Product is the other axis this model is uneven on, and the larger one: PR-AUC runs
    # 0.82 on `ProductCD == "R"` against 0.20 on `W`, which is 77% of test rows. Read off
    # `test.features` -- what the model was shown -- so that a contract dropping `ProductCD`
    # stops writing these keys and the gate's check fails loudly.
    if PRODUCT_COLUMN in test.features.columns:
        products = test.features.get_column(PRODUCT_COLUMN).cast(pl.String).to_numpy()
        for level in np.unique(products[~pl.Series(products).is_null().to_numpy()]):
            mask = products == level
            # Same floors as every other segment report here: a number estimated on a
            # handful of positives is not a number the gate should be allowed to act on.
            if mask.sum() >= 500 and labels[mask].sum() >= 20:
                metrics[f"test_pr_auc_product_{level}"] = float(
                    average_precision_score(labels[mask], probabilities[mask])
                )
                metrics[f"product_{level}_rows"] = float(mask.sum())
                metrics[f"product_{level}_positive_rate"] = float(labels[mask].mean())

    return metrics
