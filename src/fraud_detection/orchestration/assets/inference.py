"""Batch scoring: the Kaggle test period, scored by the promoted model.

Three things here are load-bearing and each was a bug before it was a decision.

1. **The windows are computed over train ∪ test, not over test alone.** See
   `SCORING_HISTORY_SQL`.
2. **The model comes from the promotion marker**, not from the newest blob — see
   `fraud_detection.core.promotion`.
3. **The submission carries raw scores, not calibrated ones.** See `kaggle_submission`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from dagster import AssetIn, AssetKey, AutomationCondition, Failure, MaterializeResult, asset
from google.cloud import bigquery, storage

from fraud_detection.core.feature_contract import ContractError, FeatureContract
from fraud_detection.core.feature_contract.admission import load_admission_rules
from fraud_detection.core.provenance import describe_code_version
from fraud_detection.core.schema import (
    FEATURES_DATASET,
    INFERENCE_DATASET,
    JOINED_TABLE,
    PREDICTION_LOG_TABLE,
    RAW_DATASET,
    RAW_TEST_IDENTITY_TABLE,
    RAW_TEST_TRANSACTION_TABLE,
    SCORING_FEATURE_TABLE,
    SCORING_HISTORY_TABLE,
    TEST_JOINED_TABLE,
    TEST_MODEL_INPUT_TABLE,
    qualified,
)
from fraud_detection.feature_engineering.derivations import apply_derivations
from fraud_detection.feature_engineering.features import build_sql
from fraud_detection.feature_engineering.scoring_history import (
    align_to_training_schema,
    build_scoring_history_sql,
)
from fraud_detection.orchestration.assets.feature_audit import CONTRACT_FILE
from fraud_detection.orchestration.assets.model_input import (
    MODEL_INPUT_SQL,
    build_feature_column_list,
)
from fraud_detection.orchestration.catalog import (
    CODE_VERSION,
    INFERENCE,
    LIGHTGBM,
)
from fraud_detection.orchestration.resources import BigQueryResource, ModelArtifactStore
from fraud_detection.training.calibration import apply_calibrator
from fraud_detection.training.data import prepare_features, to_lightgbm


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=LIGHTGBM,
    owners=INFERENCE,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="batch_inference",
    description="Joins the Kaggle test transaction and identity tables.",
)
def kaggle_test_joined(context, bigquery_resource: BigQueryResource) -> str:
    """Joins the Kaggle test tables, the same way the training tables are joined.

    The test CSVs are loaded into `raw.test_transaction` / `raw.test_identity` by the same
    ingestion path as the training tables; this asset only assembles them.
    """
    client = bigquery_resource.get_client()
    project = bigquery_resource.project

    from fraud_detection.orchestration.assets.join import (
        NULL_COUNT_COLUMN,
        build_null_count_expression,
    )

    destination = qualified(project, RAW_DATASET, TEST_JOINED_TABLE)
    client.query(f"""
    CREATE OR REPLACE TABLE `{destination}` AS
    SELECT
        t.*,
        i.* EXCEPT (TransactionID),
        {build_null_count_expression()} AS {NULL_COUNT_COLUMN}
    FROM `{qualified(project, RAW_DATASET, RAW_TEST_TRANSACTION_TABLE)}` AS t
    LEFT JOIN `{qualified(project, RAW_DATASET, RAW_TEST_IDENTITY_TABLE)}` AS i
      USING (TransactionID)
    """).result()

    # The label column exists so the union below is one schema rather than two. It is NULL
    # for every test row -- the whole point of the exercise -- and nothing reads it.
    client.query(f"ALTER TABLE `{destination}` ADD COLUMN IF NOT EXISTS isFraud FLOAT64").result()

    return destination


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=LIGHTGBM,
    owners=INFERENCE,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="batch_inference",
    ins={"kaggle_test_joined": AssetIn(key=AssetKey(["fraud_detection", "kaggle_test_joined"]))},
    description="Train ∪ test, so the test period's velocity windows can see prior history.",
)
def scoring_history(
    context, kaggle_test_joined: str, bigquery_resource: BigQueryResource
) -> str:
    """Train ∪ test, so the test period's velocity windows can see prior history.

    Read `SCORING_HISTORY_SQL` for why this table exists at all.
    """
    client = bigquery_resource.get_client()
    project = bigquery_resource.project

    train_joined = qualified(project, RAW_DATASET, JOINED_TABLE)
    destination = qualified(project, RAW_DATASET, SCORING_HISTORY_TABLE)

    train_schema = client.get_table(train_joined).schema
    test_fields = {field.name for field in client.get_table(kaggle_test_joined).schema}
    train_columns, test_columns = align_to_training_schema(train_schema, test_fields)

    missing = [f.name for f in train_schema if f.name not in test_fields]
    if missing:
        context.log.warning(
            "%s column(s) present in training and absent from the test join, "
            "selected as NULL: %s",
            len(missing),
            missing[:10],
        )

    client.query(
        build_scoring_history_sql(
            destination_table=destination,
            train_joined_table=train_joined,
            test_joined_table=kaggle_test_joined,
            train_columns=train_columns,
            test_columns=test_columns,
        )
    ).result()

    table = client.get_table(destination)
    context.log.info("Scoring history: %s rows across both periods.", table.num_rows)
    return destination


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=LIGHTGBM,
    owners=INFERENCE,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="batch_inference",
    ins={
        "kaggle_test_joined": AssetIn(key=AssetKey(["fraud_detection", "kaggle_test_joined"])),
        "scoring_history": AssetIn(key=AssetKey(["fraud_detection", "scoring_history"])),
    },
    description="Feature engineering over the full history; assembled for the test rows only.",
)
def kaggle_test_model_input(
    context,
    kaggle_test_joined: str,
    scoring_history: str,
    bigquery_resource: BigQueryResource,
) -> str:
    """Runs the feature SQL over train ∪ test, then keeps only the rows to be scored.

    The two halves are deliberately separate. Aggregates are computed over everything, so
    a test row's window contains the entity's real history; the model input is assembled
    from `test_joined`, so only test rows are scored. The inner join in `MODEL_INPUT_SQL`
    does the selection -- no `WHERE origin = 'test'` filter is needed, and a filter applied
    before the windows were computed would put the bug straight back.
    """
    client = bigquery_resource.get_client()
    project = bigquery_resource.project
    rules = load_admission_rules()

    feature_table = qualified(project, FEATURES_DATASET, SCORING_FEATURE_TABLE)
    client.query(
        build_sql(
            source_table=scoring_history,
            derivations=[d for d in rules.derivations if d.name in set(rules.uid_std_of_derived)],
            c_columns=rules.uid_c_columns,
            m_columns=rules.uid_m_columns,
            destination_table=feature_table,
        )
    ).result()

    destination = qualified(project, FEATURES_DATASET, TEST_MODEL_INPUT_TABLE)
    client.query(
        MODEL_INPUT_SQL.format(
            destination_table=destination,
            joined_table=kaggle_test_joined,
            feature_table=feature_table,
            feature_columns=build_feature_column_list(),
        )
    ).result()

    table = client.get_table(destination)
    context.log.info("Scoring input: %s rows, %s columns.", table.num_rows, len(table.schema))
    return destination


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=LIGHTGBM,
    owners=INFERENCE,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name="batch_inference",
    deps=[AssetKey(["fraud_detection", "model_input"])],
    description="Scores the test period with the promoted model and writes the submission to GCS.",
    ins={
        "best_model": AssetIn(key=AssetKey(["fraud_detection", "best_model"])),
        "kaggle_test_model_input": AssetIn(
            key=AssetKey(["fraud_detection", "kaggle_test_model_input"])
        ),
    },
)
def kaggle_submission(
    context,
    best_model: dict,
    kaggle_test_model_input: str,
    bigquery_resource: BigQueryResource,
    model_artifact_store: ModelArtifactStore,
) -> MaterializeResult:
    """Scores the test period and writes the submission to GCS.

    **Raw scores, not calibrated ones**, and the reason is that one artifact has two
    consumers wanting different things from it. Kaggle grades on ROC-AUC, which reads only
    the ranking; the isotonic calibrator is a step function, so it maps distinct scores
    onto shared plateau values and creates ties the ranking then pays for. Measured on the
    test split: PR-AUC 0.5091 uncalibrated against 0.4992 calibrated (MEASUREMENTS.md,
    "Calibration experiment").

    The decision path wants the opposite. The threshold and the cost matrix are both read
    off the probability scale, so anything that blocks a payment uses `cal_probs` and would
    be wrong to use the raw score. So: ranking consumers get `raw_probs`, decision
    consumers get `cal_probs`, and `prediction_logs` below carries both, because a log that
    kept only one of them could not answer questions about the other afterwards.
    """
    client = bigquery_resource.get_client()

    context.log.info("Loading %s ...", kaggle_test_model_input)
    frame = pl.from_arrow(client.query(f"SELECT * FROM `{kaggle_test_model_input}`").result().to_arrow())

    rules = load_admission_rules()
    frame = apply_derivations(frame, rules.derivations)

    booster = best_model["booster"]
    feature_names = booster.feature_name()

    _assert_contract_unchanged(context, best_model)

    features = prepare_features(frame)
    if missing := [name for name in feature_names if name not in features.columns]:
        raise Failure(
            f"{len(missing)} feature(s) the model was fitted on are absent from the "
            f"scoring input, e.g. {missing[:5]}."
        )

    raw_probs = booster.predict(
        to_lightgbm(features.select(feature_names)), num_iteration=booster.best_iteration
    )
    # Through `apply_calibrator`, not `calibrator.predict`. The two calibrators have
    # different interfaces -- isotonic takes a 1D array, Platt is a LogisticRegression and
    # needs `predict_proba` on a 2D one -- and which is fitted is decided per run. Calling
    # `.predict` directly works only for whichever one happens to have won.
    cal_probs = apply_calibrator(best_model["calibrator"], raw_probs)

    raw_series = pl.Series("raw", raw_probs)
    cal_series = pl.Series("calibrated", cal_probs)
    threshold = float(best_model.get("threshold", 0.5))
    context.log.info(
        "Scored %s rows. Raw: mean %.4f, median %.4f. Calibrated: mean %.4f, "
        "above threshold %.2f%%.",
        len(raw_series),
        raw_series.mean(),
        raw_series.median(),
        cal_series.mean(),
        100 * (cal_series > threshold).mean(),
    )

    run = _run_identifier(context)
    promotion = best_model.get("promotion", {})

    # GCS, not the local filesystem. The previous version wrote `kaggle/submission_x.csv`
    # and symlinked `kaggle/submission.csv` at it; a Cloud Run Job's filesystem is gone the
    # moment the container exits, so that output existed only when the pipeline ran on a
    # laptop, which is not an environment this is meant to run in.
    submission = pl.DataFrame(
        {"TransactionID": frame.get_column("TransactionID"), "isFraud": raw_series}
    )
    blob_path = f"submissions/{run}.csv"
    bucket = storage.Client(project=bigquery_resource.project).bucket(model_artifact_store.bucket)
    bucket.blob(blob_path).upload_from_string(submission.write_csv(), content_type="text/csv")
    submission_uri = f"gs://{model_artifact_store.bucket}/{blob_path}"
    context.log.info("Submission written to %s.", submission_uri)

    logged = _log_predictions(
        client,
        project=bigquery_resource.project,
        frame=frame,
        raw=raw_series,
        calibrated=cal_series,
        threshold=threshold,
        run=run,
        model_run=promotion.get("run", ""),
    )

    return MaterializeResult(
        metadata={
            "rows": len(submission),
            "submission_uri": submission_uri,
            "scored_with": promotion.get("artifact_prefix", ""),
            "score_written": "raw (uncalibrated) — Kaggle grades on ROC-AUC",
            "prediction_log_rows": logged,
        }
    )


def _run_identifier(context) -> str:
    """What distinguishes one submission from the next in the bucket.

    A Dagster run has an id worth keeping. A direct call does not: `build_asset_context`
    reports `EPHEMERAL`, so every local scoring run wrote `submissions/EPHEMERA.csv` and
    silently replaced the previous one. A timestamp is the honest substitute — it says when
    rather than which, and two runs can coexist.
    """
    from datetime import UTC, datetime

    run_id = getattr(context, "run_id", None) or ""
    if run_id and not run_id.startswith("EPHEMERAL"):
        return run_id[:8]
    return datetime.now(UTC).strftime("local-%Y%m%dT%H%M%SZ")


def _assert_contract_unchanged(context, model: dict) -> None:
    """The model's stamped contract fingerprint against the contract on disk.

    Cheap, and it closes the loop the fingerprint was built for: the stamp exists so that
    somebody eventually compares it. A contract regenerated after this model was trained
    means the scoring input is assembled from a different admitted set than the one the
    model was fitted on, which is the same class of divergence as the window bug above --
    silent, and visible only as a worse score.
    """
    stamped = model.get("contract_fingerprint") or ""
    try:
        current = FeatureContract.from_json(Path(CONTRACT_FILE).read_text()).fingerprint()
    except (OSError, ContractError) as exc:
        raise Failure(f"Cannot read the feature contract to check it against the model: {exc}") from exc

    if not stamped:
        raise Failure(
            "The promoted model carries no contract fingerprint, so there is nothing to "
            "check it against. Retrain: training stamps it."
        )
    if stamped != current:
        raise Failure(
            f"The promoted model was trained against contract {stamped}, but the contract "
            f"on disk is {current}. The admitted set changed after this model was fitted; "
            "rerun the audit and retrain rather than scoring across the gap."
        )
    context.log.info("Contract fingerprint %s matches the model's.", current)


PREDICTION_LOG_SCHEMA = [
    bigquery.SchemaField("TransactionID", "INT64"),
    bigquery.SchemaField("scored_at", "TIMESTAMP"),
    bigquery.SchemaField("run", "STRING"),
    bigquery.SchemaField("model_run", "STRING"),
    bigquery.SchemaField("raw_score", "FLOAT64"),
    bigquery.SchemaField("calibrated_probability", "FLOAT64"),
    bigquery.SchemaField("threshold", "FLOAT64"),
    bigquery.SchemaField("action", "STRING"),
    # The commit that scored the row. `model_run` says which artifact; this says which code
    # applied it, and the two are not the same question -- the scoring path, the derivations
    # and the contract projection all live in this repository and all change independently
    # of the model. Without it, "why did the same model start blocking twice as much last
    # Tuesday?" has no column to look at.
    bigquery.SchemaField("code_version", "STRING"),
]


def _log_predictions(
    client,
    *,
    project: str,
    frame: pl.DataFrame,
    raw: pl.Series,
    calibrated: pl.Series,
    threshold: float,
    run: str,
    model_run: str,
) -> int:
    """Appends this run's scores to `inference.prediction_logs`.

    Written before anything monitors it, on purpose: a drift job can be added later, but
    it can only look at history that was already being recorded. The action is derived
    from the *calibrated* probability, which is the whole reason both columns are here.
    """
    from datetime import UTC, datetime

    rows = pl.DataFrame(
        {
            "TransactionID": frame.get_column("TransactionID"),
            "raw_score": raw,
            "calibrated_probability": calibrated,
        }
    ).with_columns(
        scored_at=pl.lit(datetime.now(UTC)),
        run=pl.lit(run),
        model_run=pl.lit(model_run),
        threshold=pl.lit(threshold),
        action=pl.when(calibrated > threshold).then(pl.lit("block")).otherwise(pl.lit("allow")),
        code_version=pl.lit(describe_code_version()),
    )

    destination = qualified(project, INFERENCE_DATASET, PREDICTION_LOG_TABLE)
    job = client.load_table_from_dataframe(
        rows.select([field.name for field in PREDICTION_LOG_SCHEMA]).to_pandas(),
        destination,
        job_config=bigquery.LoadJobConfig(
            schema=PREDICTION_LOG_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            # A table written before `code_version` existed has one fewer column, and an
            # append carrying a field the destination does not know is a hard error without
            # this. Additions only -- nothing here ever removes or retypes a column, and a
            # log whose old rows could be reinterpreted would not be a log.
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        ),
    )
    job.result()
    return len(rows)
