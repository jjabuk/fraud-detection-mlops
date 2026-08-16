from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import requests
from dagster import Failure, MaterializeResult, asset
from google.cloud import storage
from kaggle.api.kaggle_api_extended import KaggleApi

from fraud_detection.orchestration.catalog import (
    CODE_VERSION,
    FEATURE_PLATFORM,
    GCS,
)
from fraud_detection.orchestration.resources import (
    KaggleIdentityRawDumpResource,
    KaggleRawDumpResource,
)


@asset(
    kinds=GCS,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="raw_ingestion",
    description="Stages the full IEEE-CIS train_transaction.csv from Kaggle into GCS.",
)
def raw_transaction_kaggle_to_gcs(
    context,
    kaggle_raw_dump: KaggleRawDumpResource,
) -> MaterializeResult:
    """Stages the full IEEE-CIS train_transaction.csv from Kaggle into GCS.

    Deliberately NOT a dependency of raw_transactions_validation /
    raw_transactions_bigquery: the dataset is a static, one-off dump, so
    this is meant to be materialized by hand, rarely, whenever the staged
    GCS file needs (re)seeding -- not re-run on every ingestion pass. See
    README for the one-time flip of RawCsvSourceResource.uri once this has
    been materialized at least once.

    Downloads to a temp dir and streams that file straight to GCS -- at no
    point does the ~650MB payload pass through this process's memory, only
    local disk (cleaned up on exit) and network.
    """
    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            api.competition_download_file(
                kaggle_raw_dump.competition,
                kaggle_raw_dump.file_name,
                path=tmp_dir,
                force=True,
                quiet=True,
            )
        except requests.exceptions.HTTPError as exc:
            raise Failure(
                f"Kaggle download of '{kaggle_raw_dump.file_name}' from "
                f"'{kaggle_raw_dump.competition}' failed: {exc}. If this is "
                "a 403, accept the competition rules at "
                f"https://www.kaggle.com/c/{kaggle_raw_dump.competition}/rules "
                "first."
            ) from exc

        local_path = Path(tmp_dir) / kaggle_raw_dump.file_name
        if not local_path.exists():
            # Kaggle's single-file competition download wraps the file in
            # a zip named "<original_name>.zip" (e.g.
            # "train_transaction.csv.zip", full name + ".zip", not the
            # extension replaced) -- confirmed against a real download,
            # not a guess. Fall back to any lone .zip in the temp dir too,
            # in case that naming shifts again.
            candidates = [
                local_path.with_name(local_path.name + ".zip"),
                *Path(tmp_dir).glob("*.zip"),
            ]
            zip_path = next((c for c in candidates if c.exists()), None)
            if zip_path is None:
                raise Failure(
                    f"Kaggle download did not produce {local_path} or a "
                    f".zip containing it; contents of {tmp_dir}: "
                    f"{sorted(p.name for p in Path(tmp_dir).iterdir())}"
                )
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(kaggle_raw_dump.file_name, path=tmp_dir)

        size_bytes = local_path.stat().st_size

        if not kaggle_raw_dump.gcs_uri.startswith("gs://"):
            raise Failure(
                f"kaggle_raw_dump.gcs_uri must be a gs:// URI, got {kaggle_raw_dump.gcs_uri!r}."
            )
        bucket_name, _, blob_path = kaggle_raw_dump.gcs_uri.removeprefix("gs://").partition("/")

        blob = storage.Client(project=kaggle_raw_dump.project).bucket(bucket_name).blob(blob_path)
        blob.upload_from_filename(str(local_path))

    context.log.info(
        "Staged %s (%s bytes) from Kaggle competition '%s' to %s.",
        kaggle_raw_dump.file_name,
        size_bytes,
        kaggle_raw_dump.competition,
        kaggle_raw_dump.gcs_uri,
    )
    return MaterializeResult(
        metadata={
            "gcs_uri": kaggle_raw_dump.gcs_uri,
            "bytes_uploaded": size_bytes,
            "competition": kaggle_raw_dump.competition,
        }
    )


@asset(
    kinds=GCS,
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="raw_ingestion",
    description="Stages the full IEEE-CIS train_identity.csv from Kaggle into GCS.",
)
def raw_identity_kaggle_to_gcs(
    context,
    kaggle_identity_dump: KaggleIdentityRawDumpResource,
) -> MaterializeResult:
    """Stages the full IEEE-CIS train_identity.csv from Kaggle into GCS."""
    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            api.competition_download_file(
                kaggle_identity_dump.competition,
                kaggle_identity_dump.file_name,
                path=tmp_dir,
                force=True,
                quiet=True,
            )
        except requests.exceptions.HTTPError as exc:
            raise Failure(f"Kaggle download failed: {exc}") from exc

        local_path = Path(tmp_dir) / kaggle_identity_dump.file_name
        if not local_path.exists():
            candidates = [
                local_path.with_name(local_path.name + ".zip"),
                *Path(tmp_dir).glob("*.zip"),
            ]
            zip_path = next((c for c in candidates if c.exists()), None)
            if zip_path is None:
                raise Failure("Kaggle download did not produce expected file.")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(kaggle_identity_dump.file_name, path=tmp_dir)

        size_bytes = local_path.stat().st_size
        bucket_name, _, blob_path = kaggle_identity_dump.gcs_uri.removeprefix("gs://").partition(
            "/"
        )

        blob = (
            storage.Client(project=kaggle_identity_dump.project).bucket(bucket_name).blob(blob_path)
        )
        blob.upload_from_filename(str(local_path))

    return MaterializeResult(
        metadata={
            "gcs_uri": kaggle_identity_dump.gcs_uri,
            "bytes_uploaded": size_bytes,
        }
    )
