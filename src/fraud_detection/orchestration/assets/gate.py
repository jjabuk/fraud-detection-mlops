from __future__ import annotations

import json
import os

from dagster import AssetIn, AssetKey, AutomationCondition, Failure, MaterializeResult, asset
from google.cloud import storage

from fraud_detection.config import get_orchestration_params
from fraud_detection.orchestration.catalog import (
    CODE_VERSION,
    MODEL_FACTORY,
    VERTEX,
)
from fraud_detection.orchestration.gate_checks import (
    MIN_DOMINANT_SEGMENT_LIFT,
    Check,
    dominant_segment,
    unseen_segment_pr_auc,
)
from fraud_detection.orchestration.resources import BigQueryResource, ModelArtifactStore
from fraud_detection.registry.provenance import UNKNOWN_SHA
from fraud_detection.training.threshold import DEFAULT_FALSE_POSITIVE_BUDGET

_gate_cfg = get_orchestration_params("gate")
_vertex_cfg = get_orchestration_params("vertex")

PRODUCTION_ALIAS = _gate_cfg["production_alias"]
VERTEX_MODEL_DISPLAY_NAME = _vertex_cfg["model_display_name"]
PROMOTION_MARKER_PATH = _gate_cfg["promotion_marker_path"]

# The serving container the registry entry points at. Env wins over config so a CI run can
# register the image it just pushed without editing a file.
SERVING_IMAGE_URI = os.getenv("SERVING_IMAGE_URI", _vertex_cfg["serving_image_uri"])
SERVING_HEALTH_ROUTE = _vertex_cfg["serving_health_route"]
SERVING_PREDICT_ROUTE = _vertex_cfg["serving_predict_route"]
SERVING_PORT = int(_vertex_cfg["serving_port"])

MIN_PR_AUC_LIFT = _gate_cfg["min_pr_auc_lift"]

# The BQML logistic baseline's PR-AUC, pinned. See the note at its use.
BASELINE_PR_AUC = _gate_cfg["baseline_pr_auc"]

# The entity is the reconstructed client (card1 + addr1 + D1), not the card. 98.9% of test
# rows sit on a card seen in training against 35% on a client, so a per-card check would ask
# almost nothing.
MIN_UNSEEN_SEGMENT_RATIO = _gate_cfg["min_unseen_segment_ratio"]

# Predicted probabilities must mean what they say to within two points. The
# threshold and the cost model are both read off the probability scale, so a
# systematically overconfident model would silently block the wrong volume.
MAX_EXPECTED_CALIBRATION_ERROR = _gate_cfg["max_expected_calibration_error"]

# The threshold is fitted on validation, where it meets the budget by construction, and
# checked on test. Some drift is inevitable; more than a quarter of the budget means the
# policy does not survive being carried forward.
MAX_FALSE_POSITIVE_DRIFT = _gate_cfg["max_false_positive_drift"]

@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=VERTEX,
    owners=MODEL_FACTORY,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="model_registry", 
    ins={"model_explanations": AssetIn(key=AssetKey(["fraud_detection", "model_explanations"]))},
    description="Promotes a model to the registry only if it clears every check.",
)
def validation_gate(
    context,
    bigquery_resource: BigQueryResource,
    model_artifact_store: ModelArtifactStore,
    model_explanations: dict,
) -> MaterializeResult:
    """Promotes a model to the registry only if it clears every check.

    The thresholds live in code that fails the run, so "it looked better" is not a
    judgement anyone makes at the end of an afternoon.
    """
    bucket = storage.Client(project=bigquery_resource.project).bucket(model_artifact_store.bucket)

    candidate_prefix = _latest_candidate_prefix(bucket)
    if candidate_prefix is None:
        raise Failure(
            f"No LightGBM run found under gs://{model_artifact_store.bucket}/lightgbm/. "
            "Materialize lightgbm_model first."
        )

    candidate = json.loads(bucket.blob(f"{candidate_prefix}/metrics.json").download_as_text())

    # Pinned rather than recomputed: the BQML baseline is one CREATE MODEL statement over a
    # fixed split, so it returns a constant. Re-derive with
    # `dagster asset materialize --select fraud_detection/bqml_baseline` if the split moves.
    baseline_pr_auc = float(BASELINE_PR_AUC)
    checks = [
        Check(
            "pr_auc_above_baseline",
            candidate["test_pr_auc"] >= baseline_pr_auc * MIN_PR_AUC_LIFT,
            f"test PR-AUC {candidate['test_pr_auc']:.4f} vs baseline {baseline_pr_auc:.4f} "
            f"× {MIN_PR_AUC_LIFT}",
        ),
        Check(
            "no_cold_entity_regression",
            unseen_segment_pr_auc(candidate) >= baseline_pr_auc * MIN_UNSEEN_SEGMENT_RATIO,
            f"unseen-entity PR-AUC {unseen_segment_pr_auc(candidate):.4f} "
            f"vs baseline {baseline_pr_auc:.4f} × {MIN_UNSEEN_SEGMENT_RATIO} "
            f"over {int(candidate.get('unseen_rows', 0)):,} rows",
        ),
        Check(
            "calibration_sane",
            candidate["test_expected_calibration_error"] <= MAX_EXPECTED_CALIBRATION_ERROR,
            f"ECE {candidate['test_expected_calibration_error']:.4f} "
            f"<= {MAX_EXPECTED_CALIBRATION_ERROR}",
        ),
        Check(
            "threshold_survives_out_of_sample",
            candidate["test_threshold_false_positive_rate"]
            <= DEFAULT_FALSE_POSITIVE_BUDGET * MAX_FALSE_POSITIVE_DRIFT,
            f"test FPR at threshold {candidate['test_threshold_false_positive_rate']:.4f} "
            f"<= budget {DEFAULT_FALSE_POSITIVE_BUDGET} × {MAX_FALSE_POSITIVE_DRIFT} "
            f"(val FPR {candidate['threshold_false_positive_rate']:.4f})",
        ),
    ]

    # Conditional, so it is appended rather than inlined above.
    dominant = dominant_segment(candidate)
    if dominant is None:
        checks.append(
            Check(
                "dominant_segment_not_regressed",
                False,
                "training wrote no per-product metrics, so the segment holding most of the "
                "traffic was never scored. Either the contract stopped admitting ProductCD "
                "or every segment fell under the reporting floor.",
            )
        )
    else:
        name, segment_pr_auc, positive_rate, share = dominant
        lift = segment_pr_auc / positive_rate if positive_rate else 0.0
        checks.append(
            Check(
                "dominant_segment_not_regressed",
                lift >= MIN_DOMINANT_SEGMENT_LIFT,
                f"{name} holds {share:.0%} of scored rows: PR-AUC {segment_pr_auc:.4f} "
                f"over a {positive_rate:.4f} positive rate is a lift of {lift:.2f}, "
                f"floor {MIN_DOMINANT_SEGMENT_LIFT}",
            )
        )

    for check in checks:
        context.log.info("%s %s — %s", "PASS" if check.passed else "FAIL", check.name, check.detail)

    failed = [check for check in checks if not check.passed]
    if failed:
        raise Failure(
            "Validation gate rejected the model; it stays out of the registry. Failed: "
            + "; ".join(f"{check.name} ({check.detail})" for check in failed)
        )

    marker_uri = _record_promotion(bucket, candidate_prefix=candidate_prefix, candidate=candidate)
    context.log.info(
        "Promoted %s to @%s; marker at %s.", candidate_prefix, PRODUCTION_ALIAS, marker_uri
    )

    registry_uri = _register_in_vertex(
        context,
        project=bigquery_resource.project,
        location=bigquery_resource.location,
        artifact_prefix=f"gs://{bucket.name}/{candidate_prefix}",
        code_version=str(candidate.get("code_version", UNKNOWN_SHA)),
        contract_fingerprint=str(candidate.get("contract_fingerprint", "")),
    )

    return MaterializeResult(
        metadata={
            "promoted": True,
            "candidate": candidate_prefix,
            "promotion_marker": marker_uri,
            "vertex_model": registry_uri,
            "baseline_pr_auc": baseline_pr_auc,
            "test_pr_auc": candidate["test_pr_auc"],
            **{check.name: check.detail for check in checks},
        }
    )


def _vertex_label(value: str) -> str:
    """Coerces a value into something the Vertex label validator accepts.

    Labels take lowercase letters, digits, `-` and `_`, up to 63 characters. A git SHA
    already satisfies that and so does `unknown`; the sanitiser exists for the `-dirty`
    suffix and for whatever a hand-set `GIT_SHA` might contain. Registration failing on a
    label would fail the *promotion*, which would be an absurd way to lose a good model.
    """
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower())
    return cleaned[:63] or UNKNOWN_SHA


def _register_in_vertex(
    context,
    *,
    project: str,
    location: str,
    artifact_prefix: str,
    code_version: str = UNKNOWN_SHA,
    contract_fingerprint: str = "",
) -> str:
    """Uploads the promoted artifact as a Vertex AI Model version.

    `Model.upload` needs an image that answers a health route and a predict route, 
    LightGBM has no prebuilt one, and pointing the entry at the prebuilt
    sklearn image would register a model that cannot serve. The image built from this
    repository's Dockerfile does answer both routes, so the entry is now true.

    Registration comes after the checks and after the marker. The marker stays the record
    the scoring path reads: it is written into the same bucket as the artifact it promotes,
    so it cannot describe a model the registry disagrees about.
    """
    if not SERVING_IMAGE_URI:
        context.log.info(
            "No SERVING_IMAGE_URI configured; the promotion marker is the record of "
            "record for this run and Vertex was not called."
        )
        return ""

    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    # parent_model makes each promotion a new *version* of one model rather than a new
    # model per training run. Version history in the registry is the point; a registry
    # with 40 unrelated entries called fraud-lightgbm is a listing, not a history.
    existing = aiplatform.Model.list(filter=f'display_name="{VERTEX_MODEL_DISPLAY_NAME}"')
    model = aiplatform.Model.upload(
        display_name=VERTEX_MODEL_DISPLAY_NAME,
        parent_model=existing[0].resource_name if existing else None,
        artifact_uri=artifact_prefix,
        serving_container_image_uri=SERVING_IMAGE_URI,
        serving_container_health_route=SERVING_HEALTH_ROUTE,
        serving_container_predict_route=SERVING_PREDICT_ROUTE,
        serving_container_ports=[SERVING_PORT],
        version_aliases=[PRODUCTION_ALIAS],
        is_default_version=True,
        # Provenance on the registry entry itself, so the chain is readable from the end
        # a reviewer actually starts at. The description is the human-readable copy; the
        # labels are what a `Model.list(filter=...)` can select on when the question is
        # "which versions came from the commit we just rolled back?".
        version_description=(
            f"code {code_version}, contract {contract_fingerprint or 'unrecorded'}"
        ),
        labels={
            "code_version": _vertex_label(code_version),
            "contract_fingerprint": _vertex_label(contract_fingerprint or "unrecorded"),
        },
    )
    context.log.info("Registered %s @%s.", model.versioned_resource_name, PRODUCTION_ALIAS)
    return model.versioned_resource_name


def _record_promotion(bucket, *, candidate_prefix: str, candidate: dict) -> str:
    """Writes the promotion marker: which artifact carries the alias.

    The marker is what the scoring path reads, and it stays that way now that
    the Vertex registry entry exists alongside it. Two reasons it is not
    redundant. It is written into the same versioned bucket as the artifact it
    promotes, in one operation, so "which model is production" survives Vertex
    being unreachable and cannot drift from the bytes it describes. And it is
    the record for a run configured without a serving image, where
    registration is skipped rather than faked.

    Read back by `core.promotion.parse_promotion_marker`, which is also what
    the Cloud Run batch job uses -- one answer to "which model", not two.
    """
    marker = {
        "alias": PRODUCTION_ALIAS,
        "display_name": VERTEX_MODEL_DISPLAY_NAME,
        "artifact_prefix": f"gs://{bucket.name}/{candidate_prefix}",
        "test_pr_auc": candidate["test_pr_auc"],
        "threshold": candidate["threshold"],
        "calibration_method": candidate["calibration_method"],
        # Copied from the candidate's metrics, not recomputed here. The gate may run in a
        # different process from training one day, and the commit that matters is the one
        # that produced the artifact -- not the one that approved it.
        "code_version": candidate.get("code_version", UNKNOWN_SHA),
        "contract_fingerprint": candidate.get("contract_fingerprint", ""),
        "registry_status": (
            f"registered against {SERVING_IMAGE_URI}"
            if SERVING_IMAGE_URI
            else "not registered: no serving image configured (SERVING_IMAGE_URI unset)"
        ),
    }
    blob = bucket.blob(PROMOTION_MARKER_PATH)
    blob.upload_from_string(json.dumps(marker, indent=2), content_type="application/json")
    return f"gs://{bucket.name}/{PROMOTION_MARKER_PATH}"


def _latest_candidate_prefix(bucket) -> str | None:
    blobs = [
        blob for blob in bucket.list_blobs(prefix="lightgbm/") if blob.name.endswith("metrics.json")
    ]
    if not blobs:
        return None
    return max(blobs, key=lambda blob: blob.updated).name.rsplit("/", 1)[0]
