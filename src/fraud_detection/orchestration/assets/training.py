from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetIn,
    AssetKey,
    AutomationCondition,
    Failure,
    Output,
    asset,
    asset_check,
)
from google.cloud import storage

from fraud_detection.config import get_training_params
from fraud_detection.contract import (
    ContractError,
    FeatureContract,
    assert_model_features_admitted,
    load_admission_rules,
)
from fraud_detection.features.derivations import DerivationError
from fraud_detection.features.entity import seen_entity_flag
from fraud_detection.orchestration.assets.baseline import build_run_suffix
from fraud_detection.orchestration.assets.feature_audit import CONTRACT_FILE
from fraud_detection.orchestration.catalog import (
    CODE_VERSION,
    LIGHTGBM,
    MODEL_FACTORY,
)
from fraud_detection.orchestration.resources import (
    BigQueryResource,
    ExperimentTracker,
    ModelArtifactStore,
)
from fraud_detection.registry.provenance import describe_code_version
from fraud_detection.schema import (
    MODEL_INPUT_TABLE,
)
from fraud_detection.training import plots
from fraud_detection.training.data import load_raw_split, split_with_contract
from fraud_detection.training.model import NUM_BOOST_ROUND, TrainedModel, train_lightgbm
from fraud_detection.training.threshold import DEFAULT_FALSE_POSITIVE_BUDGET

# The bar lives in config/feature-admission.toml, with the reasoning next to the number.


def _upload(bucket, blob_path: str, local_path: Path) -> str:
    bucket.blob(blob_path).upload_from_filename(str(local_path))
    return f"gs://{bucket.name}/{blob_path}"


def _write_artifacts(model: TrainedModel, test, destination: Path) -> dict[str, Path]:
    """Everything that has to survive the run, written into one directory."""
    (destination / "reliability.json").write_text(
        json.dumps(
            {
                "method": model.calibration_method,
                "cross_validated_log_loss": model.calibration_losses,
                "bin_edges": model.reliability.bin_edges,
                "mean_predicted": model.reliability.mean_predicted,
                "observed_rate": model.reliability.observed_rate,
                "bin_count": model.reliability.bin_count,
            },
            indent=2,
        )
    )
    (destination / "metrics.json").write_text(
        json.dumps(
            {
                "model": "lightgbm",
                "calibration_method": model.calibration_method,
                "uncalibrated_test_pr_auc": model.uncalibrated_test_pr_auc,
                # The commit that trained this artifact. Written next to the metrics rather
                # than only into the tracker because this file is what the gate reads and
                # what the marker is built from -- provenance that lives only in a
                # dashboard is provenance the scoring path cannot carry forward.
                "code_version": describe_code_version(),
                "contract_fingerprint": getattr(model, "contract_fingerprint", "") or "",
                **model.metrics,
            },
            indent=2,
        )
    )

    labels = test.labels.to_numpy()
    probabilities = model.test_probabilities
    segments = {
        name.replace("test_pr_auc_", ""): value
        for name, value in model.metrics.items()
        if name.startswith("test_pr_auc_")
    }

    figures = {
        "precision_recall": plots.precision_recall_figure(
            labels,
            probabilities,
            model.threshold,
            model.metrics["test_pr_auc"],
            destination / "precision_recall.png",
        ),
        "roc": plots.roc_figure(
            labels, probabilities, model.metrics["test_roc_auc"], destination / "roc.png"
        ),
        "score_distribution": plots.score_distribution_figure(
            labels, probabilities, model.threshold, destination / "score_distribution.png"
        ),
        "cost_curve": plots.cost_curve_figure(
            labels,
            probabilities,
            test.amounts.to_numpy(),
            model.threshold,
            model.cost_choice.threshold,
            destination / "cost_curve.png",
        ),
        "reliability": plots.reliability_figure(
            model.reliability, model.calibration_method, destination / "reliability.png"
        ),
    }
    if segments:
        figures["segments"] = plots.segment_figure(
            segments, baseline=float(labels.mean()), destination=destination / "segments.png"
        )
    return figures


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=LIGHTGBM,
    owners=MODEL_FACTORY,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="model_training",
    io_manager_key="gcs_model_io_manager",
    deps=[AssetKey(["fraud_detection", "model_input"]), AssetKey(["fraud_detection", "feature_contract"])],
    ins={"split_assignment": AssetIn(key=AssetKey(["fraud_detection", "split_assignment"]))},
    description="Trains LightGBM, calibrates it, and picks a decision threshold.",
)
def lightgbm_model(
    context,
    bigquery_resource: BigQueryResource,
    experiment_tracker: ExperimentTracker,
    model_artifact_store: ModelArtifactStore,
    split_assignment: str,
):
    """Trains LightGBM, calibrates it, and picks a decision threshold.

    The modelling itself lives in `fraud_detection.training.model`, so a notebook can call
    the same function on a sample and get the same model. What remains here is what an
    asset should be about: loading the splits, storing the artifacts, recording the run.

    Training uses only the features the contract admits.  The contract is the output of the
    feature audit — time consistency, distribution shift, redundancy — and its presence as
    an upstream dependency means this asset cannot materialize until the audit has run and
    its blocking checks have passed.
    """
    admission_rules = load_admission_rules()
    contract = FeatureContract.from_json(Path(CONTRACT_FILE).read_text())
    context.log.info(
        "Contract fingerprint %s: %d admitted, %d rejected.",
        contract.fingerprint(),
        len(contract.training_features()),
        len(contract.rejections()),
    )

    client = bigquery_resource.get_client()
    project = bigquery_resource.project

    # load_raw_split qualifies these itself, so it wants bare table names.
    tables = {
        "model_input_table": MODEL_INPUT_TABLE,
        "split_table": split_assignment.split(".")[-1],
    }

    # The frames arrive unfiltered and are projected onto the contract in memory. The
    # cold-entity flag is derived from card1/addr1/D1, which the contract may well reject
    # as features, so pruning in SQL would mean either a second query or losing the
    # segment the promotion gate is built on.
    context.log.info("Loading splits from BigQuery (all columns; projecting onto the contract in memory).")
    train_raw = load_raw_split(client, project, "train", **tables)
    val_raw = load_raw_split(client, project, "val", **tables)
    test_raw = load_raw_split(client, project, "test", **tables)

    # Same constants the feature SQL and the contract's `entity` block are built from, so
    # the cold-entity segment the gate is judged on cannot describe a different client
    # than the one the features were computed for.
    # The entity key is a column, computed once by the feature-engineering statement
    # and carried into `model_input`. Rebuilding it here would be a second
    # implementation of an identifier the warehouse already has.
    def seen(holdout) -> pl.Series:
        return seen_entity_flag(train_raw, holdout).fill_null(False).cast(pl.Boolean)

    train_seen, val_seen, test_seen = seen(train_raw), seen(val_raw), seen(test_raw)

    try:
        derived = admission_rules.derivations
        train = split_with_contract(train_raw, contract, seen_in_train=train_seen, derivations=derived)
        val = split_with_contract(val_raw, contract, seen_in_train=val_seen, derivations=derived)
        test = split_with_contract(test_raw, contract, seen_in_train=test_seen, derivations=derived)
    except (KeyError, DerivationError) as error:
        raise Failure(str(error)) from error

    try:
        search_space = get_training_params().get("lightgbm_search_space")
        search_iter = get_training_params().get("search_iter", 10)
        
        model = train_lightgbm(
            train, 
            val, 
            test, 
            search_space=search_space, 
            n_iter=search_iter
        )
    except ValueError as error:
        raise Failure(str(error)) from error

    # Stamped here, not inside train_lightgbm: the modelling recipe stays free of the
    # contract, and the fingerprint travels with the artifact into the registry.
    model.contract_fingerprint = contract.fingerprint()


    context.log.info(
        "Trained on %s rows (%s fraud, scale_pos_weight %.1f), %s features. "
        # Not "wins on log loss": since the ranking budget was added, the method with the
        # lowest log loss can be disqualified for the PR-AUC it destroys, and this line
        # printed "platt wins on cross-validated log loss (isotonic 0.07746, platt
        # 0.08127)" -- a sentence contradicted by the numbers beside it.
        "Calibration: %s chosen%s; cross-validated log loss (%s).",
        len(train),
        int(train.labels.sum()),
        model.scale_pos_weight,
        len(model.feature_names),
        model.calibration_method,
        (
            f" (disqualified on the ranking budget: "
            f"{', '.join(model.calibration_choice.disqualified)})"
            if getattr(model.calibration_choice, "disqualified", ())
            else " on log loss"
        ),
        ", ".join(f"{n} {v:.5f}" for n, v in sorted(model.calibration_losses.items())),
    )

    run_suffix = build_run_suffix(context)
    bucket = storage.Client(project=project).bucket(model_artifact_store.bucket)
    prefix = f"lightgbm/{run_suffix}"

    figure_uris: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        figures = _write_artifacts(model, test, tmp_path)

        artifact_uri = f"gs://{model_artifact_store.bucket}/lightgbm/{run_suffix}/model.pkl"
        _upload(bucket, f"{prefix}/metrics.json", tmp_path / "metrics.json")
        _upload(bucket, f"{prefix}/reliability.json", tmp_path / "reliability.json")
        for name, path in figures.items():
            figure_uris[f"plot_{name}"] = _upload(bucket, f"{prefix}/{path.name}", path)

    labels = test.labels.to_numpy()
    run_id = experiment_tracker.log_run(
        run_name=f"lightgbm-{run_suffix}",
        params={
            "backend": "lightgbm",
            "calibration_method": model.calibration_method,
            "false_positive_budget": DEFAULT_FALSE_POSITIVE_BUDGET,
            "scale_pos_weight": round(model.scale_pos_weight, 3),
            "feature_count": len(model.feature_names),
            "num_boost_round": NUM_BOOST_ROUND,
            "contract_fingerprint": contract.fingerprint(),
            "code_version": describe_code_version(),
            "artifact_uri": artifact_uri,
            **figure_uris,
        },
        metrics=model.metrics,
        classification={
            "display_name": "test-at-operating-threshold",
            "labels": ["legitimate", "fraud"],
            "matrix": plots.confusion_matrix_at(labels, model.test_probabilities, model.threshold),
            **dict(
                zip(
                    ("fpr", "tpr", "threshold"),
                    plots.roc_points_for_vertex(labels, model.test_probabilities),
                    strict=True,
                )
            ),
        },
    )

    context.log.info(
        "LightGBM: test PR-AUC %.4f (uncalibrated %.4f, floor %.4f). "
        "At a %.0f%% FP budget: threshold %.4f, recall %.3f, precision %.3f.",
        model.metrics["test_pr_auc"],
        model.uncalibrated_test_pr_auc,
        model.metrics["test_positive_rate"],
        DEFAULT_FALSE_POSITIVE_BUDGET * 100,
        model.threshold,
        model.test_at_threshold.true_positive_rate,
        model.test_at_threshold.precision,
    )

    metadata = {
        "experiment_run": run_id,
        **figure_uris,
        "calibration_method": model.calibration_method,
        "uncalibrated_test_pr_auc": model.uncalibrated_test_pr_auc,
        **model.metrics,
    }
    
    if getattr(model, "search_history", None):
        from dagster import MetadataValue
        df = pl.DataFrame(model.search_history)
        
        lines = ["| " + " | ".join(df.columns) + " |"]
        lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
        for row in df.iter_rows():
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
            
        metadata["search_history"] = MetadataValue.md("\n".join(lines))

    return Output(model, metadata=metadata)


@asset_check(
    asset=lightgbm_model, 
    blocking=True,
    description="The cheap assertion that runs every time: the trained model and the contract agree.",
)
def model_features_admitted_check(lightgbm_model: dict) -> AssetCheckResult:
    """The cheap assertion that runs every time: the trained model and the contract agree.

    The fingerprint compared is the one stamped on the model at fit time, against the
    contract on disk *now*. That is the only comparison worth making — re-reading the
    contract and checking it against itself passes unconditionally, and would have gone on
    passing while the audit regenerated the contract mid-flight.
    """
    contract = FeatureContract.from_json(Path(CONTRACT_FILE).read_text())
    # The IO manager stores `TrainedModel.as_artifact()`, so what arrives here is the
    # servable dict -- the same object the registry gets, which is the right thing to
    # check: it is what would actually be deployed.
    features = lightgbm_model["feature_names"]
    stamped = lightgbm_model.get("contract_fingerprint") or None
    try:
        assert_model_features_admitted(contract, features, fingerprint=stamped)
    except ContractError as exc:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "reason": str(exc),
                "model_fingerprint": stamped or "",
                "contract_fingerprint": contract.fingerprint(),
                "model_features": len(features),
                "admitted": len(contract.training_features()),
            },
        )
    return AssetCheckResult(
        passed=True,
        metadata={
            "fingerprint": contract.fingerprint(),
            "features": len(features),
        },
    )


@asset_check(
    asset=lightgbm_model, 
    blocking=True,
    description="The bar: feature reduction must not cost accuracy measured the same way.",
)
def test_pr_auc_threshold_check(lightgbm_model: dict) -> AssetCheckResult:
    """The bar: feature reduction must not cost accuracy measured the same way."""
    bar = get_training_params().get("min_test_pr_auc", 0.0)
    reference = get_training_params().get("reference_test_pr_auc", None)

    score = lightgbm_model["metrics"]["test_pr_auc"]
    passed = score >= bar

    # The bar and the reference are different jobs. The bar fails the run; the reference is
    # the last recorded score, so a run that passes still says which way it moved. With the
    # bar at 0 the check stops blocking and keeps reporting -- which is what makes it
    # useful while the features and the model are being improved rather than defended.
    metadata = {
        "test_pr_auc": round(score, 4),
        "bar": bar,
        "features": len(lightgbm_model["feature_names"]),
    }
    if reference is not None:
        metadata["reference"] = reference
        metadata["delta_vs_reference"] = round(score - reference, 4)
    metadata["reason"] = (
        "passed" if passed else f"scored {score:.4f} against a bar of {bar}"
    )

    return AssetCheckResult(passed=passed, severity=AssetCheckSeverity.ERROR, metadata=metadata)

