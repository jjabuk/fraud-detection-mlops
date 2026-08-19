"""Which model scores, and why "the newest one" is not an answer.

The bug this covers was not a crash. `serving.best_model` took the most recently modified
`model.pkl` under `lightgbm/`, which meant a candidate the validation gate had rejected was
picked up by the next scoring run exactly as readily as a promoted one -- so the gate had
no effect at the only point where the model does anything. These tests assert the marker
is read and that a stale one fails loudly.
"""

from __future__ import annotations

import json

import pytest

from fraud_detection.registry.promotion import (
    PromotionError,
    assert_marker_is_current,
    parse_promotion_marker,
    split_gcs_uri,
)

MARKER = {
    "alias": "production",
    "display_name": "fraud-lightgbm",
    "artifact_prefix": "gs://project-models/lightgbm/abc12345",
    "test_pr_auc": 0.5091,
    "threshold": 0.31,
    "calibration_method": "isotonic",
}


def test_the_marker_names_the_artifact_and_the_run():
    promoted = parse_promotion_marker(json.dumps(MARKER))

    assert promoted.artifact_prefix == "gs://project-models/lightgbm/abc12345"
    assert promoted.model_uri == "gs://project-models/lightgbm/abc12345/model.pkl"
    assert promoted.run == "abc12345"
    assert promoted.threshold == 0.31


def test_a_marker_without_an_artifact_prefix_is_not_a_marker():
    # An empty dict is what an interrupted write leaves behind, and defaulting to
    # "score with something" is the behaviour being removed.
    with pytest.raises(PromotionError, match="artifact_prefix"):
        parse_promotion_marker(json.dumps({"alias": "production"}))


def test_unparseable_marker_fails_rather_than_returning_nothing():
    with pytest.raises(PromotionError, match="valid JSON"):
        parse_promotion_marker("not json")


def test_a_training_run_newer_than_the_promotion_fails():
    """The rejected-candidate case, which is the whole point.

    Training uploads the artifact, then the gate runs and may reject it. So a run newer
    than the marker means either "the gate said no" or "the gate has not run" -- and
    scoring with the newer model is precisely the old bug, while silently falling back to
    the older one hides a rejection an operator should see.
    """
    promoted = parse_promotion_marker(json.dumps(MARKER))

    with pytest.raises(PromotionError, match="def67890"):
        assert_marker_is_current(promoted, "def67890")


def test_the_promoted_run_being_the_newest_run_is_the_happy_path():
    promoted = parse_promotion_marker(json.dumps(MARKER))

    assert_marker_is_current(promoted, "abc12345")
    assert_marker_is_current(promoted, None)  # nothing trained yet in this bucket


def test_gcs_uris_are_split_into_bucket_and_object():
    assert split_gcs_uri("gs://b/a/c.pkl") == ("b", "a/c.pkl")

    with pytest.raises(PromotionError):
        split_gcs_uri("/local/path.pkl")

    with pytest.raises(PromotionError):
        split_gcs_uri("gs://bucket-only")
