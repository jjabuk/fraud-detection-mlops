"""How much does a metric move when nothing changes but the seed?

Every comparison in MEASUREMENTS.md is a difference between two numbers, and a difference
is only a finding if it is larger than the band the pipeline produces on its own. That band
was quoted as "~0.003 PR-AUC" from an informal observation and had never been measured; for
ROC-AUC -- the metric the leaderboard grades -- it had never been quoted at all.

**What this measures, exactly.** The chosen model configuration, refitted N times with the
LightGBM seed as the only thing that moves. The seed perturbs bagging and feature sampling,
so this is the variance of the fit.

**What it deliberately does not measure.** The hyperparameter search. Production training
runs a 10-point random search and keeps the best on validation PR-AUC, so a full
end-to-end refit also carries search-selection variance on top of fit variance. Including
it would cost ten times the compute for a second-order effect, so the number this reports
is a **lower bound** on the run-to-run band. A difference smaller than this bound is
certainly noise; a difference larger than it is not certainly signal.

Nothing is uploaded. Five candidate artifacts under `lightgbm/` would break
`_latest_candidate_prefix`, which is how the promotion gate finds the model it is judging.

Splits come from the parquet cache when it is warm (`training/data.py`), so the cost is
N fits and, at most, one load.

    uv run noise-band --seeds 42,7,1337,2024,91 --out noise_band.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.contract import CONTRACT_FILE, FeatureContract
from fraud_detection.contract.admission import load_admission_rules
from fraud_detection.features.entity import seen_entity_flag
from fraud_detection.schema import (
    MODEL_INPUT_TABLE,
    SPLIT_TABLE,
)
from fraud_detection.training.data import load_raw_split, split_with_contract
from fraud_detection.training.model import train_lightgbm

DEFAULT_SEEDS = (42, 7, 1337, 2024, 91)

# The configuration a production run actually settled on, so the band describes the model
# that gets promoted rather than the base config nothing is trained with. Refresh from
# `booster.params` of the current promoted artifact when the search space or the data move.
WINNING_PARAMS: dict[str, Any] = {
    "num_leaves": 96,
    "learning_rate": 0.05,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "min_child_samples": 80,
}


def _load_splits(project: str):
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    contract = FeatureContract.from_json(CONTRACT_FILE.read_text())
    rules = load_admission_rules()

    tables = {"model_input_table": MODEL_INPUT_TABLE, "split_table": SPLIT_TABLE}
    raw = {name: load_raw_split(client, project, name, **tables) for name in ("train", "val", "test")}


    def seen(holdout) -> pl.Series:
        return seen_entity_flag(raw["train"], holdout).fill_null(False).cast(pl.Boolean)

    splits = {
        name: split_with_contract(
            frame, contract, seen_in_train=seen(frame), derivations=rules.derivations
        )
        for name, frame in raw.items()
    }
    return contract, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--out", default="noise_band.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    project = os.environ["GCP_PROJECT_ID"]

    contract, splits = _load_splits(project)
    print(
        f"contract {contract.fingerprint()}: {len(contract.training_features())} features; "
        f"{ {name: len(frame.labels) for name, frame in splits.items()} }",
        flush=True,
    )

    # A one-point "search space": train_lightgbm takes the grid, and a grid with one value
    # per axis is how the winning configuration is pinned without a second code path.
    space = {key: [value] for key, value in WINNING_PARAMS.items()}
    labels = splits["test"].labels.to_numpy()

    rows = []
    for seed in seeds:
        started = time.time()
        model = train_lightgbm(
            splits["train"], splits["val"], splits["test"],
            search_space=space, n_iter=1, seed=seed,
        )
        row = {
            "seed": seed,
            "raw_roc_auc": float(roc_auc_score(labels, model.test_scores)),
            "raw_pr_auc": float(average_precision_score(labels, model.test_scores)),
            "cal_roc_auc": float(roc_auc_score(labels, model.test_probabilities)),
            "cal_pr_auc": float(average_precision_score(labels, model.test_probabilities)),
            "calibration": model.calibration_method,
            "calibration_pr_auc_loss": float(model.calibration_choice.pr_auc_loss),
            "calibration_disqualified": list(model.calibration_choice.disqualified),
            "distinct_calibrated": int(np.unique(model.test_probabilities).size),
            "best_iteration": int(model.booster.best_iteration),
            "seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    print(f"\n=== fit-only noise band over {len(seeds)} seeds ===", flush=True)
    for metric in ("raw_roc_auc", "raw_pr_auc", "cal_roc_auc", "cal_pr_auc"):
        values = [row[metric] for row in rows]
        spread = max(values) - min(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"{metric:14s} mean {statistics.mean(values):.4f}  sd {sd:.4f}  "
            f"min {min(values):.4f}  max {max(values):.4f}  range {spread:.4f}"
        )
    print("\ncalibration chosen :", [row["calibration"] for row in rows])
    print("disqualified       :", [row["calibration_disqualified"] for row in rows])

    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
