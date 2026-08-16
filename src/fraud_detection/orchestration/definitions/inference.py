"""Code location 3 — the inference & serving pipeline.

This location exposes the validated model for serving and runs inference pipelines 
such as the Kaggle test set generation.

It depends on `validation_gate` from `model_factory` (via the marker) and `model_input`
from `feature_platform` to reconstruct exact training schemas.
"""

from dagster import AssetSpec, Definitions, define_asset_job

from fraud_detection.orchestration.assets.inference import (
    kaggle_submission,
    kaggle_test_joined,
    kaggle_test_model_input,
    scoring_history,
)
from fraud_detection.orchestration.assets.serving import best_model
from fraud_detection.orchestration.resources import BigQueryResource, ModelArtifactStore

EXTERNAL_ASSETS = [
    AssetSpec(
        key=["fraud_detection", "model_input"],
        description="features.model_input, built by the feature platform. Used to fetch training categories.",
        metadata={"owner": "feature_platform"},
    ),
    AssetSpec(
        key=["fraud_detection", "validation_gate"],
        description="The validation gate from the model factory. Triggers when a new model is promoted.",
        metadata={"owner": "model_factory"},
    )
]


defs = Definitions(
    assets=[
        *EXTERNAL_ASSETS,
        best_model,
        kaggle_test_joined,
        scoring_history,
        kaggle_test_model_input,
        kaggle_submission,
    ],
    resources={
        "bigquery_resource": BigQueryResource(),
        "model_artifact_store": ModelArtifactStore(),
    },
    jobs=[define_asset_job("inference_job", selection="*")],
    # No schedule, for the same reason and more sharply: scoring should happen **because a
    # new model was promoted**, which is a dependency already in the graph. An 08:00 cron
    # would score with whatever the marker pointed at, including yesterday's model when
    # today's training failed -- and the scoring path would then refuse, correctly, leaving
    # a red run every morning that meant nothing.
)
