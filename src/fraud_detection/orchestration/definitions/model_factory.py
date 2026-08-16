"""Code location 2 — the model factory.

See README.md for architectural documentation and graphs.
"""

from dagster import AssetSpec, Definitions, define_asset_job

from fraud_detection.orchestration.assets.baseline import bqml_baseline
from fraud_detection.orchestration.assets.explainability import model_explanations
from fraud_detection.orchestration.assets.gate import validation_gate
from fraud_detection.orchestration.assets.splits import split_assignment
from fraud_detection.orchestration.assets.training import (
    lightgbm_model,
    model_features_admitted_check,
    test_pr_auc_threshold_check,
)
from fraud_detection.orchestration.resources import (
    BigQueryResource,
    ExperimentTracker,
    GCSModelIOManager,
    ModelArtifactStore,
)

# Declared, not built. Dagster stitches these onto the real assets in the feature platform
# location by key, so the lineage graph spans both and a stale upstream is visible from
# here -- without this location importing a single line of the platform's code.
EXTERNAL_ASSETS = [
    AssetSpec(
        key=["fraud_detection", "model_input"],
        description=(
            "features.model_input, built by the feature platform. Depended on by key: "
            "this location reads the table, it does not receive its name as a value."
        ),
        metadata={"owner": "feature_platform"},
    ),
    AssetSpec(
        key=["fraud_detection", "feature_contract"],
        description=(
            "references/feature-contract.json, built by the feature platform. Decides "
            "which columns training may use."
        ),
        metadata={"owner": "feature_platform"},
    ),
]


defs = Definitions(
    assets=[
        *EXTERNAL_ASSETS,
        split_assignment,
        bqml_baseline,
        lightgbm_model,
        model_explanations,
        validation_gate,
    ],
    asset_checks=[
        model_features_admitted_check,
        test_pr_auc_threshold_check,
    ],
    resources={
        "bigquery_resource": BigQueryResource(),
        "experiment_tracker": ExperimentTracker(),
        "model_artifact_store": ModelArtifactStore(),
        "gcs_model_io_manager": GCSModelIOManager(),
    },
    jobs=[define_asset_job("model_factory_job", selection="*")],
    # No schedule. Training should follow the feature platform, not the clock: a 04:00 cron
    # retrains on whatever the contract happened to be at 04:00, including a contract that
    # did not change and a contract that failed its checks. `lightgbm_model` carries
    # `AutomationCondition.eager()`, so it runs when its upstreams have actually moved.
)
