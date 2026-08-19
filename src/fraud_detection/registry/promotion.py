"""Which artifact carries the production alias — read as data, never guessed from a listing.

The validation gate writes one JSON object saying *this* prefix passed *these* checks, and
anything that scores a transaction reads that object. Picking the most recently modified
`model.pkl` instead answers a different question: training uploads before the gate runs, so
the newest artifact is the promoted one only by coincidence.

Pure on purpose: this parses and validates a dict. The bucket read belongs to whoever holds
a client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

__all__ = [
    "PromotedModel",
    "PromotionError",
    "assert_marker_is_current",
    "parse_promotion_marker",
    "split_gcs_uri",
]


class PromotionError(Exception):
    """Raised when no model is promoted, or the promoted one is not the current one."""


@dataclass(frozen=True)
class PromotedModel:
    """The gate's decision, as the scoring path needs to see it."""

    alias: str
    artifact_prefix: str
    """A `gs://bucket/lightgbm/<run>` URI — the directory, not the pickle."""
    display_name: str = ""
    threshold: float | None = None
    calibration_method: str = ""
    test_pr_auc: float | None = None
    code_version: str = ""
    """The commit that trained the artifact, as `core.provenance` described it."""
    contract_fingerprint: str = ""

    @property
    def model_uri(self) -> str:
        return f"{self.artifact_prefix.rstrip('/')}/model.pkl"

    @property
    def run(self) -> str:
        """The training run's identifier — the last path segment of the prefix."""
        return self.artifact_prefix.rstrip("/").rsplit("/", 1)[-1]


def parse_promotion_marker(text: str) -> PromotedModel:
    """Reads the marker the gate wrote.

    Raises `PromotionError` rather than returning None: a scoring run with no promoted
    model has nothing correct to do, and continuing with a default would reintroduce
    exactly the behaviour this module exists to remove.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PromotionError(f"promotion marker is not valid JSON: {exc}") from exc

    prefix = payload.get("artifact_prefix")
    if not prefix:
        raise PromotionError(
            "promotion marker carries no artifact_prefix; it was not written by the "
            "validation gate"
        )

    return PromotedModel(
        alias=payload.get("alias", ""),
        artifact_prefix=prefix,
        display_name=payload.get("display_name", ""),
        threshold=payload.get("threshold"),
        calibration_method=payload.get("calibration_method", ""),
        test_pr_auc=payload.get("test_pr_auc"),
        # Absent on markers written before provenance was recorded. Empty rather than
        # raising: an old marker is a marker whose code version is unknown, which is a
        # fact about it, not a reason to refuse to score.
        code_version=payload.get("code_version", ""),
        contract_fingerprint=payload.get("contract_fingerprint", ""),
    )


def assert_marker_is_current(promoted: PromotedModel, latest_run: str | None) -> None:
    """Fails when a training run happened after the last promotion.

    That state means one of two things: a model was trained and the gate rejected it, or
    a model was trained and the gate has not run yet. Both make the promoted artifact
    stale relative to the code and data that produced `latest_run`, and in neither case
    is scoring with the newer model correct — that is the behaviour being removed. Falling
    back to the older *promoted* model would also be wrong silently, so this raises and
    lets an operator decide.
    """
    if latest_run is None:
        return
    if promoted.run != latest_run:
        raise PromotionError(
            f"promoted model is {promoted.run} but the newest training run is "
            f"{latest_run}. Either the gate rejected {latest_run} or it has not run "
            "against it yet; scoring with either model would be a guess. Materialize "
            "validation_gate."
        )


def split_gcs_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/a/b` -> `("bucket", "a/b")`."""
    if not uri.startswith("gs://"):
        raise PromotionError(f"not a GCS URI: {uri}")
    bucket, _, path = uri[len("gs://") :].partition("/")
    if not bucket or not path:
        raise PromotionError(f"GCS URI names no object: {uri}")
    return bucket, path
