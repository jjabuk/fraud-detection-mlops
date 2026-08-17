from __future__ import annotations

import os
import pickle
from pathlib import Path

from dagster import ConfigurableIOManager, ConfigurableResource, InputContext, OutputContext
from google.cloud import bigquery

# Single source of truth for the GCP project every resource below bills
# API calls to. google.cloud clients that talk to a gs:// URI (storage.
# Client() in particular) can't reliably infer a project from this
# machine's impersonated-service-account ADC the way bigquery.Client()
# does when given one explicitly -- pass this everywhere instead of
# relying on inference.
from fraud_detection.core.config import get_orchestration_params

_gcp_cfg = get_orchestration_params("gcp")
DEFAULT_GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "local-test-project")

# Single source of truth for where the full Kaggle dumps live once staged.
# raw_transaction_kaggle_to_gcs (assets/kaggle_source.py) writes here;
# RawCsvSourceResource.uri gets pointed at this same value once you're
# ready to switch validation/load off the committed sample.
_storage_cfg = get_orchestration_params("storage")
RAW_DUMP_GCS_URI = _storage_cfg["raw_dump_gcs_uri"].format(project_id=DEFAULT_GCP_PROJECT_ID)
IDENTITY_RAW_DUMP_GCS_URI = _storage_cfg["identity_raw_dump_gcs_uri"].format(project_id=DEFAULT_GCP_PROJECT_ID)
TEST_DUMP_GCS_URI = _storage_cfg["test_dump_gcs_uri"].format(project_id=DEFAULT_GCP_PROJECT_ID)
TEST_IDENTITY_DUMP_GCS_URI = _storage_cfg["test_identity_dump_gcs_uri"].format(project_id=DEFAULT_GCP_PROJECT_ID)

# Hand-maintained (see schemas/README.md). Consumed by IaC and by the BigQuery load jobs,
# so a type change is a reviewable diff rather than a re-inference on the next load.
BQ_SCHEMA_PATH_TRANSACTIONS = Path(__file__).parent.parent.parent.parent / "schemas" / "bq_schema_train_transaction.json"
BQ_SCHEMA_PATH_IDENTITY = Path(__file__).parent.parent.parent.parent / "schemas" / "bq_schema_train_identity.json"


class RawCsvSourceResource(ConfigurableResource):
    """The one true location of the static Kaggle train_transaction.csv dump.

    `uri` is either a gs:// URI (production: enables load_table_from_uri in
    raw_transactions_bigquery, so the file's bytes never pass through this
    process) or a local filesystem path (dev/test: the committed sample).
    `project` is only consulted for gs:// URIs, to bill the storage.Client()
    read in raw_transactions_validation.
    """

    uri: str = "data/raw/train_transaction_sample.csv"
    project: str = DEFAULT_GCP_PROJECT_ID

    @property
    def is_gcs(self) -> bool:
        return self.uri.startswith("gs://")

class IdentityRawCsvSourceResource(ConfigurableResource):
    """The location of the train_identity.csv file.

    Unlike RawCsvSourceResource, this default points at a file that is NOT in
    the repository -- there is no committed identity sample. definitions.py
    overrides it with IDENTITY_RAW_DUMP_GCS_URI, so the wired pipeline is
    unaffected; a bare instantiation will fail loudly instead of skipping.
    """

    uri: str = "data/raw/train_identity_sample.csv"
    project: str = DEFAULT_GCP_PROJECT_ID

    @property
    def is_gcs(self) -> bool:
        return self.uri.startswith("gs://")


class KaggleRawDumpResource(ConfigurableResource):
    """Where to fetch the full raw dump from Kaggle, and where it lands in
    GCS. Consumed only by raw_transaction_kaggle_to_gcs -- a rare, by-hand
    materialization, not part of the recurring validate/load chain.

    Credentials: kaggle's own KaggleApi.authenticate() resolves them itself
    (checks ~/.kaggle/access_token, then legacy KAGGLE_USERNAME/KAGGLE_KEY,
    then OAuth) -- nothing secret is modeled as resource config here.
    """

    competition: str = "ieee-fraud-detection"
    file_name: str = "train_transaction.csv"
    gcs_uri: str = RAW_DUMP_GCS_URI
    project: str = DEFAULT_GCP_PROJECT_ID

class KaggleIdentityRawDumpResource(ConfigurableResource):
    """Where to fetch the identity raw dump from Kaggle."""
    competition: str = "ieee-fraud-detection"
    file_name: str = "train_identity.csv"
    gcs_uri: str = IDENTITY_RAW_DUMP_GCS_URI
    project: str = DEFAULT_GCP_PROJECT_ID


class BigQueryResource(ConfigurableResource):
    project: str = DEFAULT_GCP_PROJECT_ID
    location: str = "europe-central2"

    def get_client(self) -> bigquery.Client:
        return bigquery.Client(project=self.project, location=self.location)


# ---- the same assets, against a local warehouse --------------------------------
#
# assets that transform data do not know which one they were handed. That is the point: the
# join, the feature aggregates, the model input and the splits are each one SQL statement,
# and a second implementation of any of them in pandas would be a second definition of the
# feature. The dialect difference is four substitutions in fraud_detection.dialect.
#
# What this deliberately does not cover: `CREATE MODEL` (BigQuery ML has no equivalent) and
# the two ingestion assets, which use load jobs and pinned schemas. Loading a CSV is
# genuinely engine-specific and computes nothing, so two loaders is not two definitions.



# Vertex AI Experiments. Named for the role rather than the backend, because
# assets should not know which tracker they are talking to -- but there is
# exactly one implementation, and adding a second before anything needs it
# would be building an abstraction for an imagined requirement.
#
# Experiment and run identifiers are constrained by Vertex: lowercase letters,
# digits and hyphens, starting with a letter or digit.
_vertex_cfg = get_orchestration_params("vertex")
VERTEX_EXPERIMENT_NAME = _vertex_cfg["experiment_name"]


class ModelArtifactStore(ConfigurableResource):
    """Where trained model artifacts land.

    A separate bucket from the raw dump: that one holds an immutable input
    that must never be lost, this one holds regenerable outputs rewritten on
    every training run.
    """

    bucket: str = f"{DEFAULT_GCP_PROJECT_ID}-models"

    def uri(self, path: str) -> str:
        return f"gs://{self.bucket}/{path}"


class ExperimentTracker(ConfigurableResource):
    """Records a training run's parameters and metrics.

    Exposes one method rather than the SDK itself, so assets never touch the
    tracker's global state and tests can substitute a recorder without
    patching a client library.
    """

    project: str = DEFAULT_GCP_PROJECT_ID
    location: str = "europe-central2"
    experiment_name: str = VERTEX_EXPERIMENT_NAME

    def log_run(
        self,
        run_name: str,
        params: dict[str, object],
        metrics: dict[str, float],
        classification: dict | None = None,
    ) -> str:
        """Records one run. `classification` is optional and, when given, must
        carry `labels`, `matrix`, `fpr`, `tpr` and `threshold`.

        Those go through Vertex's own classification-metrics API rather than
        being uploaded as images, because Vertex renders them interactively in
        the console — a reviewer can read the confusion matrix at the operating
        point without downloading anything. Everything Vertex will not render
        (precision-recall, cost curves, reliability) is a PNG in the model
        bucket, linked from the run's params.
        """
        from google.cloud import aiplatform

        aiplatform.init(
            project=self.project,
            location=self.location,
            experiment=self.experiment_name,
        )

        # Vertex refuses to reopen a finished run, so the caller passes a name
        # that is unique per execution. end_run() sits in a finally block
        # because a run left open stays in RUNNING state and blocks its own
        # name from being reused.
        aiplatform.start_run(run=run_name)
        try:
            aiplatform.log_params({key: str(value) for key, value in params.items()})
            aiplatform.log_metrics(metrics)
            if classification:
                aiplatform.log_classification_metrics(
                    display_name=classification.get("display_name", "test"),
                    labels=classification["labels"],
                    matrix=classification["matrix"],
                    fpr=classification["fpr"],
                    tpr=classification["tpr"],
                    threshold=classification["threshold"],
                )
        finally:
            aiplatform.end_run()

        return run_name

class GCSModelIOManager(ConfigurableIOManager):
    """Saves TrainedModel objects to GCS and loads the model artifacts."""

    bucket: str = f"{DEFAULT_GCP_PROJECT_ID}-models"
    # storage.Client() with no project falls back to the ambient environment, and a
    # service-account credentials file does not carry one -- so a run that trained fine
    # died at handle_output with "Project was not passed and could not be determined".
    # Every other GCS caller in this file passes the project explicitly; this one did not.
    project: str = DEFAULT_GCP_PROJECT_ID

    def handle_output(self, context: OutputContext, obj):
        from google.cloud import storage
        if obj is None:
            return

        client = storage.Client(project=self.project)
        run_id = context.run_id[:8] if context.run_id else "local"
        blob_path = f"lightgbm/{run_id}/model.pkl"
        bucket = client.bucket(self.bucket)
        
        # Serialize the subset of the model needed for serving
        bucket.blob(blob_path).upload_from_string(pickle.dumps(obj.as_artifact()))
        context.add_output_metadata({"artifact_uri": f"gs://{self.bucket}/{blob_path}"})

    def load_input(self, context: InputContext):
        from google.cloud import storage
        client = storage.Client(project=self.project)
        bucket = client.bucket(self.bucket)
        
        blobs = [b for b in bucket.list_blobs(prefix="lightgbm/") if b.name.endswith("model.pkl")]
        if not blobs:
            raise RuntimeError(f"No model found in gs://{self.bucket}/lightgbm/")
        newest = max(blobs, key=lambda b: b.updated)
        
        context.log.info(f"Loading model from gs://{self.bucket}/{newest.name}")
        return pickle.loads(newest.download_as_bytes())
