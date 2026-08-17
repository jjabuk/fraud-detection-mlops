"""Figures that make a training run reviewable.

Each one exists because a single number hides something specific:

- PR-AUC hides *where* on the curve the model is usable, and the operating
  point is the only part of the curve anyone will ever run at.
- ROC-AUC hides class imbalance almost entirely -- at a 3.5% positive rate it
  flatters every model, which is why it is plotted but never gated on.
- A calibration error hides *where* the miscalibration sits, and the band that
  matters is the narrow one around the threshold.
- A cost total hides how sharp the optimum is: a flat cost curve means the
  threshold barely matters, a steep one means it matters enormously.

Plotly with a static PNG export, because these are read in a pull request, in
the Dagster catalog and in the model card -- none of which run a browser.
Nothing here recomputes a metric: every number a figure draws is passed in by
the caller that already computed it, so a plot cannot disagree with the run it
describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from fraud_detection.core.config import get_training_params
from fraud_detection.training.calibration import ReliabilityCurve
from fraud_detection.training.threshold import (
    COST_CANDIDATE_COUNT,
    DEFAULT_FALSE_POSITIVE_COST,
    total_cost,
)

_p_cfg = get_training_params("training.plots")
FIGSIZE = tuple(_p_cfg["figsize"])
DPI = _p_cfg["dpi"]

# The config is in inches and DPI because it predates plotly; one multiplication
# keeps both the file and the exported size where they were.
WIDTH_PX = int(FIGSIZE[0] * DPI)
HEIGHT_PX = int(FIGSIZE[1] * DPI)

TEMPLATE = "simple_white"
MODEL_COLOUR = "#1f4e79"
REFERENCE_COLOUR = "#8c8c8c"
BUDGET_COLOUR = "#c1440e"
COST_COLOUR = "#2e7d5b"


def _write(figure: go.Figure, destination: Path, title: str) -> Path:
    figure.update_layout(
        title={"text": title, "font": {"size": 13}},
        template=TEMPLATE,
        width=WIDTH_PX,
        height=HEIGHT_PX,
        font={"size": 12},
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        legend={"font": {"size": 11}, "bgcolor": "rgba(255,255,255,0.7)"},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(destination)
    return destination


def precision_recall_figure(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    pr_auc: float,
    destination: Path,
) -> Path:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    base_rate = float(np.mean(labels))

    # `precision_recall_curve` returns one more point than it does thresholds:
    # the last point is the (recall 0, precision 1) sentinel, which no threshold
    # produces. Indexing has to stay inside `thresholds`, or an operating point
    # above every observed score gets annotated onto that sentinel.
    index = int(np.searchsorted(thresholds, threshold, side="left"))
    index = min(index, thresholds.size - 1)

    figure = go.Figure()
    figure.add_scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"model (PR-AUC {pr_auc:.4f})",
        line={"width": 2, "color": MODEL_COLOUR},
    )
    figure.add_hline(
        y=base_rate,
        line={"width": 1, "dash": "dash", "color": REFERENCE_COLOUR},
        annotation_text=f"random ranker ({base_rate:.4f})",
        annotation_position="bottom right",
        annotation_font_size=11,
    )
    figure.add_scatter(
        x=[recall[index]],
        y=[precision[index]],
        mode="markers",
        name="operating point",
        marker={"size": 11, "color": BUDGET_COLOUR},
    )
    figure.add_annotation(
        x=recall[index],
        y=precision[index],
        text=(
            f"threshold {threshold:.3f}<br>"
            f"precision {precision[index]:.3f}, recall {recall[index]:.3f}"
        ),
        showarrow=True,
        arrowhead=2,
        ax=60,
        ay=-50,
        font={"size": 11},
        bgcolor="rgba(255,255,255,0.8)",
    )
    figure.update_xaxes(title_text="recall (fraud caught)", range=[0, 1])
    figure.update_yaxes(title_text="precision (of blocked transactions)", range=[0, 1])
    return _write(figure, destination, "Precision-recall on test")


def roc_figure(
    labels: np.ndarray, probabilities: np.ndarray, roc_auc: float, destination: Path
) -> Path:
    fpr, tpr, _ = roc_curve(labels, probabilities)

    figure = go.Figure()
    figure.add_scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"model (ROC-AUC {roc_auc:.4f})",
        line={"width": 2, "color": MODEL_COLOUR},
    )
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="random",
        line={"width": 1, "dash": "dash", "color": REFERENCE_COLOUR},
    )
    figure.update_xaxes(title_text="false-positive rate", range=[0, 1])
    figure.update_yaxes(title_text="true-positive rate", range=[0, 1])
    return _write(
        figure,
        destination,
        "ROC on test, plotted for completeness and never gated on<br>"
        "<sub>at a 3.5% positive rate it flatters every model</sub>",
    )


def score_distribution_figure(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, destination: Path
) -> Path:
    bins = {"start": 0.0, "end": 1.0, "size": 0.02}

    figure = go.Figure()
    figure.add_histogram(
        x=probabilities[labels == 0],
        xbins=bins,
        name="legitimate",
        opacity=0.65,
        marker={"color": MODEL_COLOUR},
    )
    figure.add_histogram(
        x=probabilities[labels == 1],
        xbins=bins,
        name="fraud",
        opacity=0.65,
        marker={"color": BUDGET_COLOUR},
    )
    figure.add_vline(
        x=threshold,
        line={"width": 1.5, "dash": "dash", "color": BUDGET_COLOUR},
        annotation_text=f"threshold {threshold:.3f}",
        annotation_position="top right",
        annotation_font_size=11,
    )
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text="calibrated fraud probability", range=[0, 1])
    figure.update_yaxes(title_text="transactions", type="log")
    return _write(figure, destination, "Where the two classes sit on the score scale")


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
    """Total cost against threshold, with both candidate operating points on it.

    The curve is evaluated on the same grid `cost_minimising_threshold` searched,
    plus the two thresholds themselves. Sampling a narrower band, as this used to,
    can leave the cost-minimising line outside the plotted range whenever the
    optimum sits below that band -- a chart that silently omits the point it exists
    to make.
    """
    searched = np.quantile(probabilities, np.linspace(0.0, 1.0, COST_CANDIDATE_COUNT))
    candidates = np.unique(np.concatenate([searched, [budget_threshold, cost_threshold]]))
    costs = [
        total_cost(probabilities, labels, amounts, t, false_positive_cost=false_positive_cost)
        for t in candidates
    ]

    # Both operating points stay inside the view; the cost scale is uninformative
    # far below them, where everything is blocked.
    busiest = float(np.quantile(probabilities, 0.9))
    tail = float(np.quantile(probabilities, 0.999))
    low = max(0.0, min(budget_threshold, cost_threshold, busiest) - 0.05)
    high = min(1.0, max(budget_threshold, cost_threshold, tail) + 0.05)

    figure = go.Figure()
    figure.add_scatter(
        x=candidates,
        y=costs,
        mode="lines",
        name="total cost",
        line={"width": 2, "color": MODEL_COLOUR},
    )
    # Legend entries rather than in-plot annotations: the two thresholds can sit
    # arbitrarily close together, and annotated vertical lines then overprint each
    # other exactly when the chart is most interesting.
    ceiling = max(costs)
    for threshold, colour, dash, name in (
        (budget_threshold, BUDGET_COLOUR, "dash", "false-positive budget"),
        (cost_threshold, COST_COLOUR, "dot", "cost-minimising"),
    ):
        figure.add_scatter(
            x=[threshold, threshold],
            y=[0, ceiling],
            mode="lines",
            name=f"{name} ({threshold:.3f})",
            line={"width": 1.5, "dash": dash, "color": colour},
        )
    figure.update_xaxes(title_text="threshold", range=[low, high])
    figure.update_yaxes(title_text="missed fraud + review cost", rangemode="tozero")
    return _write(
        figure,
        destination,
        "Cost against threshold<br>"
        "<sub>the gap between the two lines is the price of the friction policy</sub>",
    )


def reliability_figure(reliability: ReliabilityCurve, method: str, destination: Path) -> Path:
    figure = go.Figure()
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="perfect",
        line={"width": 1, "dash": "dash", "color": REFERENCE_COLOUR},
    )
    figure.add_scatter(
        x=reliability.mean_predicted,
        y=reliability.observed_rate,
        mode="lines+markers",
        name=f"{method} calibrated",
        line={"width": 2, "color": MODEL_COLOUR},
        marker={"size": 8},
    )
    figure.update_xaxes(title_text="mean predicted probability")
    figure.update_yaxes(title_text="observed fraud rate")
    return _write(
        figure,
        destination,
        f"Reliability on test, {method} calibration<br>"
        f"<sub>expected calibration error {reliability.expected_calibration_error:.4f}</sub>",
    )


def segment_figure(segments: dict[str, float], baseline: float, destination: Path) -> Path:
    """PR-AUC per segment against the base rate a random ranker would score.

    The pooled metric is a row-weighted average, so a dominant segment carries it
    almost entirely: the product segment holding 77% of scored rows scores roughly
    a quarter of the best segment, and no aggregate number shows that.
    """
    names = list(segments)
    values = [segments[name] for name in names]

    figure = go.Figure()
    figure.add_bar(
        x=names,
        y=values,
        width=0.55,
        marker={"color": MODEL_COLOUR},
        text=[f"{value:.4f}" for value in values],
        textposition="outside",
        name="PR-AUC",
    )
    figure.add_hline(
        y=baseline,
        line={"width": 1.5, "dash": "dash", "color": BUDGET_COLOUR},
        annotation_text=f"base rate {baseline:.4f}",
        annotation_position="top right",
        annotation_font_size=11,
    )
    figure.update_yaxes(title_text="PR-AUC", rangemode="tozero")
    return _write(figure, destination, "PR-AUC by segment")


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
    """Rows are the truth, columns the decision. `>=` matches `threshold._confusion`."""
    blocked = probabilities >= threshold
    return [
        [int(np.sum(~blocked & (labels == 0))), int(np.sum(blocked & (labels == 0)))],
        [int(np.sum(~blocked & (labels == 1))), int(np.sum(blocked & (labels == 1)))],
    ]
