"""The HTTP surface Vertex AI requires of a custom serving container.

This exists for one specific reason, and it is worth being exact about what that reason is
so the README does not have to overstate it.

`Model.upload` will not register a model without a serving container: either one of
Vertex's prebuilt images, or one of yours that answers a health route and a predict route.
LightGBM has no prebuilt image. Registering the model against the prebuilt sklearn image
would produce a registry entry that cannot serve, which is worse than no entry -- it is an
untruth the registry then repeats to everything that reads it. So the registry stayed empty
and the promotion decision lived in a bucket.

The batch job needed a container anyway. Giving that same image the two routes Vertex asks
for costs a few dozen lines and makes the registry entry real: one image is both the Cloud
Run Job that scores the test period and a genuine custom-container model version.

**What this is not.** It is not the online serving path from architecture.md §3.2, and the
architecture doc still lists online serving as out of scope. The velocity features are
windowed aggregates over an entity's history, so a real endpoint needs a point lookup
against entity state; this handler takes the retrieved values in the request instead. That
is the honest boundary: the model, the contract projection and the decision rule are the
production ones, the feature retrieval is the caller's.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Vertex's custom-container contract. It sets these; the defaults are what Cloud Run and a
# local `docker run` see. Reading them rather than hardcoding is the whole contract --
# a container that listens on the wrong port is a model version that never passes its
# health check and is impossible to debug from the registry side.
PORT = int(os.getenv("AIP_HTTP_PORT", "8080"))
HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")
STORAGE_URI = os.getenv("AIP_STORAGE_URI", "")

app = FastAPI(title="fraud-detection", version="1")

_model: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    """Vertex's request envelope: a list of instances, whatever an instance means.

    Here an instance is a flat mapping of feature name to value, carrying both the request
    fields and the entity state the caller retrieved. Absent features arrive as nulls,
    which is what LightGBM was trained to handle -- it is a value, not an error.
    """

    instances: list[dict[str, Any]] = Field(min_length=1)


def load_model() -> dict[str, Any]:
    """The artifact, from wherever this container was told to find it.

    Vertex sets `AIP_STORAGE_URI` to the directory it copied the artifact into; a local run
    points `MODEL_PATH` at a file. Loaded once and cached, because unpickling a booster per
    request would dominate the latency it is supposed to be measuring.
    """
    global _model
    if _model is not None:
        return _model

    local_path = os.getenv("MODEL_PATH")
    if local_path:
        with open(local_path, "rb") as handle:
            _model = pickle.load(handle)
        return _model

    if not STORAGE_URI:
        raise RuntimeError(
            "Neither MODEL_PATH nor AIP_STORAGE_URI is set; this container has no model "
            "to serve."
        )

    from google.cloud import storage

    from fraud_detection.registry.promotion import split_gcs_uri

    bucket_name, prefix = split_gcs_uri(f"{STORAGE_URI.rstrip('/')}/model.pkl")
    client = storage.Client()
    _model = pickle.loads(client.bucket(bucket_name).blob(prefix).download_as_bytes())
    return _model


@app.get(HEALTH_ROUTE)
def health() -> dict[str, Any]:
    """Ready only once the model is loaded and usable.

    A health route that returns 200 before the artifact is readable reports a model
    version as serving when the first real request will fail, so this does the load.
    """
    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc
    return {"status": "healthy", "features": len(model["feature_names"])}


@app.post(PREDICT_ROUTE)
def predict(request: PredictRequest) -> dict[str, Any]:
    """One probability, one action, and the model that produced them.

    The action comes off the **calibrated** probability, and the raw score is returned
    alongside it rather than instead of it. Those are two different numbers for two
    different purposes -- the batch path makes the same split, and MEASUREMENTS.md records
    why: the threshold and the cost matrix are defined on the probability scale, while
    anything that only ranks is better off with the raw score.
    """
    from fraud_detection.training.calibration import apply_calibrator
    from fraud_detection.training.data import prepare_features, to_lightgbm

    model = load_model()
    booster = model["booster"]
    feature_names = booster.feature_name()
    threshold = float(model.get("threshold", 0.5))

    frame = pl.DataFrame(request.instances, infer_schema_length=None)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        # Explicit rather than null-filled. A caller that omits the retrieved entity state
        # would otherwise get a confident-looking score computed as if the card had no
        # history, which is the same skew the batch path was fixed for.
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"{len(missing)} feature(s) missing from the instance",
                "missing": missing[:20],
            },
        )

    features = prepare_features(frame).select(feature_names)
    raw = booster.predict(to_lightgbm(features), num_iteration=booster.best_iteration)
    # `apply_calibrator`, not `.predict`: isotonic and Platt have different interfaces
    # and the winner is chosen per training run, so calling one of them directly works
    # only until the other wins.
    calibrated = apply_calibrator(model["calibrator"], raw)

    return {
        "predictions": [
            {
                "fraud_probability": float(probability),
                "raw_score": float(score),
                "action": "block" if probability > threshold else "allow",
                "threshold": threshold,
                "contract_fingerprint": model.get("contract_fingerprint", ""),
            }
            for score, probability in zip(raw, calibrated)
        ]
    }
