"""Figures that make a training run reviewable.

Each one exists because a single number hides something specific:

- PR-AUC hides *where* on the curve the model is usable, and the operating
  point is the only part of the curve anyone will ever run at.
- ROC-AUC hides class imbalance almost entirely -- at a 3.5% positive rate it
  flatters every model, which is why it is plotted but never gated on.
- Brier score hides systematic over- or under-confidence in the narrow
  probability band where decisions actually get made.
- A cost total hides how sharp the optimum is: a flat cost curve means the
  threshold barely matters, a steep one means it matters enormously.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from fraud_detection.core.config import get_training_params
from fraud_detection.training.calibration import ReliabilityCurve
from fraud_detection.training.threshold import (
    DEFAULT_FALSE_POSITIVE_COST,
    total_cost,
)

_p_cfg = get_training_params("training.plots")
FIGSIZE = tuple(_p_cfg["figsize"])
DPI = _p_cfg["dpi"]


def _finish(figure, axes, destination: Path, title: str) -> Path:
    axes.set_title(title)
    axes.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    return destination


def precision_recall_figure(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    pr_auc: float,
    destination: Path,
) -> Path:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    base_rate = float(labels.mean())

    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.plot(recall, precision, linewidth=1.6, label=f"model (PR-AUC {pr_auc:.4f})")
    axes.axhline(
        base_rate,
        linestyle="--",
        linewidth=1,
        color="grey",
        label=f"random ranker ({base_rate:.4f})",
    )

    # The operating point is the whole reason the curve is worth drawing.
    index = int(np.searchsorted(thresholds, threshold))
    index = min(index, len(precision) - 1)
    axes.plot(recall[index], precision[index], "o", markersize=8, color="crimson")
    axes.annotate(
        f"operating point\nthreshold {threshold:.3f}\n"
        f"precision {precision[index]:.3f}, recall {recall[index]:.3f}",
        xy=(recall[index], precision[index]),
        xytext=(0.35, 0.75),
        textcoords="axes fraction",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )

    axes.set_xlabel("recall (fraud caught)")
    axes.set_ylabel("precision (of blocked transactions)")
    axes.legend(loc="upper right", fontsize=8)
    return _finish(figure, axes, destination, "Precision-recall on test")


def roc_figure(
    labels: np.ndarray, probabilities: np.ndarray, roc_auc: float, destination: Path
) -> Path:
    fpr, tpr, _ = roc_curve(labels, probabilities)

    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.plot(fpr, tpr, linewidth=1.6, label=f"model (ROC-AUC {roc_auc:.4f})")
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="random")
    axes.set_xlabel("false-positive rate")
    axes.set_ylabel("true-positive rate")
    axes.legend(loc="lower right", fontsize=8)
    return _finish(
        figure,
        axes,
        destination,
        "ROC on test — plotted for completeness, not gated on\n"
        "(at a 3.5% positive rate it flatters every model)",
    )


def score_distribution_figure(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, destination: Path
) -> Path:
    figure, axes = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, 51)
    axes.hist(
        probabilities[labels == 0], bins=bins, alpha=0.65, label="legitimate", log=True
    )
    axes.hist(probabilities[labels == 1], bins=bins, alpha=0.65, label="fraud", log=True)
    axes.axvline(
        threshold, color="crimson", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.3f}"
    )
    axes.set_xlabel("calibrated fraud probability")
    axes.set_ylabel("transactions (log scale)")
    axes.legend(fontsize=8)
    return _finish(
        figure, axes, destination, "Where the two classes actually sit on the score scale"
    )


def cost_curve_figure(
    labels: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    budget_threshold: float,
    cost_threshold: float,
    destination: Path,
    *,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> Path:
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.80, 1.0, 120)))
    costs = [
        total_cost(probabilities, labels, amounts, t, false_positive_cost=false_positive_cost)
        for t in candidates
    ]

    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.plot(candidates, costs, linewidth=1.6, label="total cost")
    axes.axvline(
        budget_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"1% FP budget ({budget_threshold:.3f})",
    )
    axes.axvline(
        cost_threshold,
        color="seagreen",
        linestyle=":",
        linewidth=1.4,
        label=f"cost-minimising ({cost_threshold:.3f})",
    )
    axes.set_xlabel("threshold")
    axes.set_ylabel("missed fraud + review cost")
    axes.legend(fontsize=8)
    return _finish(
        figure,
        axes,
        destination,
        "Cost against threshold — the gap between the two lines\nis the price of the friction policy",
    )


def reliability_figure(
    reliability: ReliabilityCurve, method: str, destination: Path
) -> Path:
    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="perfect")
    axes.plot(
        reliability.mean_predicted,
        reliability.observed_rate,
        marker="o",
        linewidth=1.6,
        label=f"{method} calibrated",
    )
    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed fraud rate")
    axes.legend(fontsize=8)
    return _finish(
        figure,
        axes,
        destination,
        f"Reliability on test — {method}, "
        f"ECE {reliability.expected_calibration_error:.4f}",
    )


def segment_figure(segments: dict[str, float], baseline: float, destination: Path) -> Path:
    """PR-AUC per segment against the baseline it has to beat.

    98.6% of test rows sit on a card the model saw in training, so the
    aggregate number is almost entirely the 'seen' bar. This chart exists so
    that fact cannot hide.
    """
    names = list(segments)
    values = [segments[name] for name in names]

    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.bar(names, values, width=0.55)
    axes.axhline(
        baseline, color="crimson", linestyle="--", linewidth=1.2, label=f"baseline {baseline:.4f}"
    )
    for index, value in enumerate(values):
        axes.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    axes.set_ylabel("PR-AUC")
    axes.legend(fontsize=8)
    return _finish(figure, axes, destination, "PR-AUC by segment")


def roc_points_for_vertex(
    labels: np.ndarray, probabilities: np.ndarray, max_points: int = 200
) -> tuple[list[float], list[float], list[float]]:
    """ROC curve downsampled to something Vertex will accept and render."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    if len(fpr) > max_points:
        index = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr, thresholds = fpr[index], tpr[index], thresholds[index]
    # roc_curve puts inf in the first threshold; Vertex wants finite numbers.
    thresholds = np.clip(thresholds, 0.0, 1.0)
    return [float(v) for v in fpr], [float(v) for v in tpr], [float(v) for v in thresholds]


def confusion_matrix_at(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> list[list[int]]:
    blocked = probabilities >= threshold
    return [
        [int(np.sum(~blocked & (labels == 0))), int(np.sum(blocked & (labels == 0)))],
        [int(np.sum(~blocked & (labels == 1))), int(np.sum(blocked & (labels == 1)))],
    ]
