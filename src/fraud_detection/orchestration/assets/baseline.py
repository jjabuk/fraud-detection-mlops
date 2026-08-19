from __future__ import annotations

import json
import uuid

from dagster import AssetIn, AssetKey, Failure, Output, asset
from google.cloud import storage
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    MODEL_FACTORY,
)
from fraud_detection.orchestration.resources import (
    BigQueryResource,
    ExperimentTracker,
    ModelArtifactStore,
)
from fraud_detection.schema import (
    FEATURES_DATASET,
    LABEL_COLUMN,
    MODEL_INPUT_TABLE,
    qualified,
)
from fraud_detection.training.bqml import (
    BASELINE_FEATURE_COLUMNS,
    BASELINE_METRICS_PATH,
    BASELINE_MODEL,
    CREATE_MODEL_SQL,
    PREDICT_SQL,
    VERTEX_BASELINE_MODEL_ID,
    build_feature_column_list,
    build_transform_column_list,
)


def build_run_suffix(context) -> str:
    """Short, unique-per-execution suffix for the experiment run name.

    Vertex refuses to reopen a finished run, so a fixed name would make the
    second materialization of this asset fail -- the same problem the Dataproc
    batch id had. The Dagster run id keeps the experiment run traceable back
    to the pipeline run that produced it.

    Directly invoking an asset (which is how the tests call this) leaves
    context.run unset, and Dagster raises rather than returning None, so
    getattr with a default does not cover it.

    Broad except on purpose: the exception Dagster raises here
    (DagsterInvalidPropertyError) is not exported from the public package, and
    importing it from dagster._core would couple this module to an internal
    path for no gain. The fallback is unconditionally safe -- any failure to
    read the run id just means a random suffix instead of a traceable one.
    """
    try:
        return context.run.run_id[:8]
    except Exception:  # noqa: BLE001
        return uuid.uuid4().hex[:8]


@asset(
    kinds=BIGQUERY,
    owners=MODEL_FACTORY,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="model_training", 
    deps=[AssetKey(["fraud_detection", "model_input"])], 
    ins={"split_assignment": AssetIn(key=AssetKey(["fraud_detection", "split_assignment"]))},
    description="Trains the trivial BigQuery ML baseline and tracks it in Vertex AI.",
)
def bqml_baseline(
    context,
    bigquery_resource: BigQueryResource,
    experiment_tracker: ExperimentTracker,
    model_artifact_store: ModelArtifactStore,
    split_assignment: str,
):
    """Trains the trivial BigQuery ML baseline and tracks it in Vertex AI.

    Tracking is wired from the first model rather than the last, on purpose:
    instrumentation added after a good model exists never gets added at all.

    PR-AUC is computed here from the raw scores rather than read out of
    ML.EVALUATE, so that every model in this project -- BQML, LightGBM, any
    later variant -- is scored by one implementation of one metric. A
    comparison between two differently-computed numbers is not a comparison.
    """
    client = bigquery_resource.get_client()
    project = bigquery_resource.project
    model_id = qualified(project, FEATURES_DATASET, BASELINE_MODEL)
    # `model_input` crosses the code-location boundary and is therefore a dependency by
    # key; `split_assignment` is produced in this same location, so its value is passed.
    model_input_table = qualified(project, FEATURES_DATASET, MODEL_INPUT_TABLE)
    split_table = split_assignment
    feature_columns = build_feature_column_list()

    create_model = CREATE_MODEL_SQL.format(
        model_id=model_id,
        label_column=LABEL_COLUMN,
        feature_columns=feature_columns,
        transform_columns=build_transform_column_list(),
        vertex_model_id=VERTEX_BASELINE_MODEL_ID,
        model_input_table=model_input_table,
        split_table=split_table,
    )
    try:
        client.query(create_model).result(timeout=3600)
    except Exception as exc:
        raise Failure(f"BQML baseline training failed for {model_id}: {exc}") from exc

    predict = PREDICT_SQL.format(
        model_id=model_id,
        label_column=LABEL_COLUMN,
        feature_columns=feature_columns,
        model_input_table=model_input_table,
        split_table=split_table,
        split="val",
    )
    try:
        rows = list(client.query(predict).result(timeout=1800))
    except Exception as exc:
        raise Failure(f"BQML baseline scoring failed for {model_id}: {exc}") from exc

    if not rows:
        raise Failure(
            f"BQML baseline scored zero rows from {split_table} split 'val'. "
            "The split assignment produced no validation rows."
        )

    y_true = [row["y_true"] for row in rows]
    y_score = [row["y_score"] for row in rows]
    positives = sum(y_true)

    if positives == 0:
        raise Failure(
            "Validation split contains no positive labels, so PR-AUC is undefined. "
            "The time-based split has put every fraud case outside validation."
        )

    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        # The positive rate is the PR-AUC a random ranker would score, so it
        # is the floor any model has to clear to have done anything at all.
        "positive_rate": positives / len(y_true),
        "val_rows": float(len(y_true)),
    }

    # The validation gate reads this to find out what "above baseline" means.
    # A fixed path, overwritten each run: the bucket is versioned, so the
    # history is kept without the gate needing to work out which of several
    # files is current.
    storage.Client(project=project).bucket(model_artifact_store.bucket).blob(
        BASELINE_METRICS_PATH
    ).upload_from_string(
        json.dumps({"model": "bqml_logreg", **metrics}, indent=2),
        content_type="application/json",
    )

    run_id = experiment_tracker.log_run(
        run_name=f"bqml-baseline-logreg-{build_run_suffix(context)}",
        params={
            "model_type": "LOGISTIC_REG",
            "backend": "bigquery_ml",
            "feature_count": len(BASELINE_FEATURE_COLUMNS),
            "features": ",".join(BASELINE_FEATURE_COLUMNS),
            "eval_split": "val",
        },
        metrics=metrics,
    )

    context.log.info(
        "BQML baseline: PR-AUC %.4f (random floor %.4f), ROC-AUC %.4f on %s rows.",
        metrics["pr_auc"],
        metrics["positive_rate"],
        metrics["roc_auc"],
        len(y_true),
    )
    return Output(metrics, metadata={"model_id": model_id, "experiment_run": run_id, **metrics})
