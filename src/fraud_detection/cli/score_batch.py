"""Run the batch scoring path locally, without Dagster's run machinery.

**This is an entrypoint, not a second implementation**, and the distinction is the whole
reason the file is allowed to exist. `cli/generate_submission.py` was deleted because it
carried its own DuckDB pipeline, its own feature SQL and its own idea of which model was
current — three definitions of things the pipeline already defined, drifting quietly apart.

This calls the asset functions themselves. Every statement executed here is the statement
the Cloud Run Job executes: the same join, the same `scoring_history` union, the same
`build_sql`, the same promotion marker, the same raw-score submission written to the same
bucket. If the job's behaviour changes, this changes with it or stops compiling.

What it is for: getting a submission without a container, and reading the failure directly
when the job misbehaves.

    uv run score-batch                 # scores and leaves the CSV in GCS
    uv run score-batch --out sub.csv   # ... and downloads it here
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Read `.env` the way a Dagster run would, so both entrypoints see one configuration.

    Only fills variables that are not already set, so an explicit export still wins.
    """
    env = Path(path)
    if not env.exists():
        return

    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Before the imports below, and that is the point rather than an oversight:
# `resources.py` reads GCP_PROJECT_ID at **module import time**, so a call inside `main()`
# would arrive after the default had already been frozen to the `local-test-project`
# placeholder. Dagster does the same thing for its own runs; `uv run` does not.
_load_dotenv()

from dagster import build_asset_context
from google.cloud import storage

from fraud_detection.core.promotion import split_gcs_uri
from fraud_detection.orchestration.assets.inference import (
    kaggle_submission,
    kaggle_test_joined,
    kaggle_test_model_input,
    scoring_history,
)
from fraud_detection.orchestration.assets.serving import best_model
from fraud_detection.orchestration.resources import (
    BigQueryResource,
    ModelArtifactStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="GCP project; defaults to the resource default")
    parser.add_argument("--out", default=None, help="Download the submission to this path")
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Score against the existing features.test_model_input instead of rebuilding it.",
    )
    args = parser.parse_args()






    bigquery_resource = (
        BigQueryResource(project=args.project) if args.project else BigQueryResource()
    )
    if bigquery_resource.project == "local-test-project":
        sys.exit(
            "No GCP project resolved — the default placeholder is still in place.\n"
            "Pass --project, or set GCP_PROJECT_ID in the environment or in .env."
        )

    model_artifact_store = ModelArtifactStore()
    context = build_asset_context()

    print(f"Project: {bigquery_resource.project}")

    if args.skip_features:
        from fraud_detection.core.schema import (
            FEATURES_DATASET,
            TEST_MODEL_INPUT_TABLE,
            qualified,
        )

        model_input = qualified(
            bigquery_resource.project, FEATURES_DATASET, TEST_MODEL_INPUT_TABLE
        )
        print(f"[1-3/5] skipped; scoring {model_input}")
    else:
        print("[1/5] joining the test tables ...")
        joined = kaggle_test_joined(context, bigquery_resource)

        print("[2/5] building scoring_history (train ∪ test) ...")
        history = scoring_history(context, joined, bigquery_resource)

        print("[3/5] feature engineering over the union, assembled for the test rows ...")
        model_input = kaggle_test_model_input(context, joined, history, bigquery_resource)

    print("[4/5] resolving the promoted model from its marker ...")
    model = best_model(context, bigquery_resource, model_artifact_store)

    print("[5/5] scoring ...")
    result = kaggle_submission(
        context, model, model_input, bigquery_resource, model_artifact_store
    )

    metadata = {key: getattr(value, "value", value) for key, value in result.metadata.items()}
    uri = metadata["submission_uri"]
    print(f"\nrows: {metadata['rows']:,}")
    print(f"scored with: {metadata['scored_with']}")
    print(f"submission: {uri}")

    if args.out:
        bucket_name, blob_path = split_gcs_uri(uri)
        destination = Path(args.out)
        storage.Client(project=bigquery_resource.project).bucket(bucket_name).blob(
            blob_path
        ).download_to_filename(destination)
        print(f"downloaded to: {destination.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
