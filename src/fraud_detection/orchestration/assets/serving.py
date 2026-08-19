from __future__ import annotations

import pickle

from dagster import AutomationCondition, Failure, asset
from google.cloud import storage

from fraud_detection.config import get_orchestration_params
from fraud_detection.orchestration.catalog import (
    CODE_VERSION,
    GCS,
    INFERENCE,
)
from fraud_detection.orchestration.resources import BigQueryResource, ModelArtifactStore
from fraud_detection.registry.promotion import (
    PromotedModel,
    PromotionError,
    assert_marker_is_current,
    parse_promotion_marker,
    split_gcs_uri,
)

PROMOTION_MARKER_PATH = get_orchestration_params("gate")["promotion_marker_path"]


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=GCS,
    owners=INFERENCE,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="model_serving",
    description="The model the validation gate promoted, read from the promotion marker.",
)
def best_model(
    context, bigquery_resource: BigQueryResource, model_artifact_store: ModelArtifactStore
) -> dict:
    """The model the validation gate promoted — not the most recently written one.

    Taking `max(blobs, key=lambda b: b.updated)` over `lightgbm/` instead returns the
    newest model rather than the best one, and which bypassed the gate entirely: training
    uploads the artifact before the gate runs, so a rejected candidate was picked up by
    the next scoring run exactly as readily as an approved one. The marker is the gate's
    output, so reading it is what makes "promotion is never unconditional" true at the one
    point where the model is actually used.
    """
    bucket = storage.Client(project=bigquery_resource.project).bucket(model_artifact_store.bucket)

    try:
        promoted = load_promoted_model(bucket)
    except PromotionError as exc:
        raise Failure(str(exc)) from exc

    context.log.info(
        "Promoted model @%s: %s (test PR-AUC %s, calibration %s).",
        promoted.alias,
        promoted.artifact_prefix,
        promoted.test_pr_auc,
        promoted.calibration_method,
    )

    _, blob_path = split_gcs_uri(promoted.model_uri)
    model = pickle.loads(bucket.blob(blob_path).download_as_bytes())
    model["promotion"] = {
        "alias": promoted.alias,
        "artifact_prefix": promoted.artifact_prefix,
        "run": promoted.run,
    }
    return model


def load_promoted_model(bucket) -> PromotedModel:
    """Reads the marker off `bucket` and checks it against the newest training run.

    Separate from the asset so the Cloud Run job can call it without a Dagster context —
    one implementation of "which model is production", used by both paths.
    """
    marker = bucket.blob(PROMOTION_MARKER_PATH)
    if not marker.exists():
        raise PromotionError(
            f"No promotion marker at gs://{bucket.name}/{PROMOTION_MARKER_PATH}. "
            "No model has passed the validation gate, so there is nothing to score with."
        )

    promoted = parse_promotion_marker(marker.download_as_text())
    assert_marker_is_current(promoted, latest_training_run(bucket))
    return promoted


def latest_training_run(bucket) -> str | None:
    """The newest run under `lightgbm/`, promoted or not."""
    blobs = [b for b in bucket.list_blobs(prefix="lightgbm/") if b.name.endswith("model.pkl")]
    if not blobs:
        return None
    return max(blobs, key=lambda b: b.updated).name.split("/")[1]
