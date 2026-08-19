from __future__ import annotations

import numpy as np
from dagster import AssetIn, AssetKey, AutomationCondition, Output, asset

# SHAP values come from LightGBM itself via pred_contrib=True, not from the
# `shap` package. LightGBM implements TreeSHAP in C++ and returns exactly the
# same numbers; the package would add numba and llvmlite as dependencies to
# call into it. Fewer dependencies, identical values.
from fraud_detection.config import get_orchestration_params
from fraud_detection.orchestration.catalog import (
    CODE_VERSION,
    LIGHTGBM,
    MODEL_FACTORY,
)
from fraud_detection.orchestration.resources import BigQueryResource
from fraud_detection.schema import MODEL_INPUT_TABLE
from fraud_detection.training.data import align_categories, to_lightgbm

_expl_cfg = get_orchestration_params("explainability")
SAMPLE_SIZE = _expl_cfg["sample_size"]
TOP_FEATURES = _expl_cfg["top_features"]
INDIVIDUAL_EXPLANATIONS = _expl_cfg["individual_explanations"]


@asset(
    automation_condition=AutomationCondition.eager(),
    kinds=LIGHTGBM,
    owners=MODEL_FACTORY,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name="model_registry", 
    deps=[AssetKey(["fraud_detection", "model_input"])], 
    ins={"split_assignment": AssetIn(key=AssetKey(["fraud_detection", "split_assignment"])), "lightgbm_model": AssetIn(key=AssetKey(["fraud_detection", "lightgbm_model"]))},
    description="Global feature importance plus a handful of individual decisions.",
)
def model_explanations(
    context,
    bigquery_resource: BigQueryResource,
    split_assignment: str,
    lightgbm_model: dict,
):
    """Global feature importance plus a handful of individual decisions.

    In a fraud system an unexplainable block is an operational problem before
    it is an analytical one: someone has to tell a customer why their card was
    declined, and "the model said so" is not an answer a chargeback team can
    use.
    """
    project = bigquery_resource.project

    bundle = lightgbm_model
    booster = bundle["booster"]

    # `load_raw_split` + `apply_derivations`, not `load_split`. `load_split` returns the
    # table as it stands, and a derived column does not exist there -- the contract says
    # how to compute it. Training has always gone through `split_with_contract`, which
    # applies the derivations first; this asset did not, and the mismatch was invisible
    # for as long as every declared derivation happened to be *rejected* by the audits.
    # The moment 22 encoded columns were admitted, the model's feature_names named columns
    # this frame had never heard of.
    from fraud_detection.contract.admission import load_admission_rules
    from fraud_detection.features.derivations import apply_derivations
    from fraud_detection.training.data import load_raw_split, prepare_features

    raw = load_raw_split(
        bigquery_resource.get_client(),
        project,
        "test",
        model_input_table=MODEL_INPUT_TABLE,
        split_table=split_assignment.split(".")[-1],
    )
    test_features = prepare_features(apply_derivations(raw, load_admission_rules().derivations))
    # `.select(...)`, not `df[[...]]`: bracket indexing with a list is ambiguous in
    # polars -- it reads the list as row positions when it can, and raised
    # "building Series of type Int64; found value of type String: 'TransactionAmt'".
    # `.select` says "columns" and cannot mean anything else.
    features = align_categories(test_features, test_features).select(bundle["feature_names"])

    # `seed`, not pandas' `random_state`: `features` is a polars frame, and the pandas
    # spelling raises TypeError rather than sampling differently -- the same migration
    # leftover that stopped `lightgbm_model` upstream.
    sample = features.sample(n=min(SAMPLE_SIZE, len(features)), seed=0)
    # (rows, features + 1): the trailing column is the base value, the model's
    # output before any feature moved it.
    contributions = booster.predict(to_lightgbm(sample), num_iteration=booster.best_iteration, pred_contrib=True)
    feature_contributions = contributions[:, :-1]
    base_value = float(contributions[0, -1])

    mean_absolute = np.abs(feature_contributions).mean(axis=0)
    order = np.argsort(mean_absolute)[::-1][:TOP_FEATURES]
    global_importance = [
        {"feature": bundle["feature_names"][index], "mean_abs_shap": float(mean_absolute[index])}
        for index in order
    ]

    # Individual explanations for the rows the model was most certain about:
    # those are the decisions that actually get acted on.
    scores = booster.predict(to_lightgbm(sample), num_iteration=booster.best_iteration)
    most_confident = np.argsort(scores)[::-1][:INDIVIDUAL_EXPLANATIONS]
    individual = []
    for row in most_confident:
        row_contributions = feature_contributions[row]
        top = np.argsort(np.abs(row_contributions))[::-1][:10]
        individual.append(
            {
                "raw_score": float(scores[row]),
                "base_value": base_value,
                "drivers": [
                    {
                        "feature": bundle["feature_names"][index],
                        "shap": float(row_contributions[index]),
                        # By column *name*, not by position. Positional indexing quietly
                        # assumed `sample`'s column order matched the model's
                        # `feature_names`, which is a different object that happens to
                        # agree until a contract change reorders one of them -- and then
                        # every reported driver names one column and shows another's
                        # value. It stopped agreeing on contract bdb97707 and raised
                        # instead, which was the lucky outcome.
                        "value": _readable(
                            sample.get_column(bundle["feature_names"][index])[int(row)]
                        ),
                    }
                    for index in top
                ],
            }
        )

    payload = {
        "sample_size": len(sample),
        "base_value": base_value,
        "global_importance": global_importance,
        "individual_explanations": individual,
    }

    context.log.info(
        "Top drivers: %s",
        ", ".join(item["feature"] for item in global_importance[:5]),
    )
    return Output(
        payload,
        metadata={
            "sample_size": len(sample),
            "top_features": ", ".join(item["feature"] for item in global_importance[:10]),
        },
    )


def _readable(value):
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return None if value is None else str(value)
