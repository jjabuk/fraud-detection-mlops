"""Code location 1 — the feature platform.

See README.md for architectural documentation and graphs.
"""


from dagster import Definitions, ScheduleDefinition, define_asset_job

from fraud_detection.orchestration.assets.feature_audit import (
    audit_frame,
    feature_contract,
)
from fraud_detection.orchestration.assets.feature_contract_check import (
    feature_contract_freshness,
    feature_contract_integrity,
)
from fraud_detection.orchestration.assets.feature_engineering import transaction_features
from fraud_detection.orchestration.assets.ingestion import (
    raw_identity_bigquery,
    raw_test_identity_bigquery,
    raw_test_transaction_bigquery,
    raw_transactions_bigquery,
)
from fraud_detection.orchestration.assets.join import joined_transactions_identity
from fraud_detection.orchestration.assets.kaggle_source import (
    raw_identity_kaggle_to_gcs,
    raw_transaction_kaggle_to_gcs,
)
from fraud_detection.orchestration.assets.model_input import model_input
from fraud_detection.orchestration.resources import (
    IDENTITY_RAW_DUMP_GCS_URI,
    RAW_DUMP_GCS_URI,
    BigQueryResource,
    IdentityRawCsvSourceResource,
    KaggleIdentityRawDumpResource,
    KaggleRawDumpResource,
    RawCsvSourceResource,
)

defs = Definitions(
    assets=[
        raw_transaction_kaggle_to_gcs,
        raw_identity_kaggle_to_gcs,
        raw_transactions_bigquery,
        raw_identity_bigquery,
        raw_test_transaction_bigquery,
        raw_test_identity_bigquery,
        joined_transactions_identity,
        transaction_features,
        model_input,
        audit_frame,
    feature_contract,
    ],
    asset_checks=[
        feature_contract_freshness,
        feature_contract_integrity,
    ],
    resources={
        # Local/dev default lives on RawCsvSourceResource itself (the committed sample).
        # Point at RAW_DUMP_GCS_URI -- or via run config -- once
        # raw_transaction_kaggle_to_gcs has staged the full file. See README.
        "raw_csv_source": RawCsvSourceResource(uri=RAW_DUMP_GCS_URI),
        "identity_raw_csv_source": IdentityRawCsvSourceResource(uri=IDENTITY_RAW_DUMP_GCS_URI),
        "bigquery_resource": BigQueryResource(),
        "kaggle_raw_dump": KaggleRawDumpResource(),
        "kaggle_identity_dump": KaggleIdentityRawDumpResource(),
    },
    jobs=[define_asset_job("feature_platform_job", selection="*")],
    # The one place a clock is the right trigger: nothing upstream of ingestion is an asset,
    # so "has the source moved?" can only be answered by looking. Everything downstream of
    # it is driven by AutomationCondition on the assets themselves -- see `assets/`.
    schedules=[ScheduleDefinition(job_name="feature_platform_job", cron_schedule="0 0 * * *")],
)
