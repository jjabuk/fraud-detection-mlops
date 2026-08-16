"""Blocking checks on the feature contract — the gate before training.

A contract that is stale, tampered, or degenerate should not be the basis for a
training run.  These checks catch that.  They are ``@asset_check``s rather than
assertions inside the asset itself because the failure mode is different: a check
that fails blocks materialization of downstream consumers (``lightgbm_model``)
without deleting the contract.  The contract stays as the last known-good state,
and the failure is visible in the Dagster UI as a check result, not as an asset
that vanished.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dagster import AssetCheckResult, AssetCheckSeverity, asset_check

from fraud_detection.core.feature_contract import (
    ContractError,
    FeatureContract,
    load_admission_rules,
)
from fraud_detection.orchestration.assets.feature_audit import CONTRACT_FILE, feature_contract

__all__ = [
    "feature_contract_freshness",
    "feature_contract_integrity",
]

# Both live in config/feature-admission.toml, alongside every other threshold the audits
# run under. Read per check rather than cached at import so an edit to the policy takes
# effect on the next materialization, not the next process restart.


@asset_check(
    asset=feature_contract, 
    blocking=True,
    description="The contract must exist and must not be older than the staleness window.",
)
def feature_contract_freshness() -> AssetCheckResult:
    """The contract must exist and must not be older than the staleness window.

    A contract created 90 days ago was built on data the model has long since
    moved past.  The staleness window is a policy knob: weekly retrains want 7
    days, monthly cadence wants 30.
    """
    path = Path(CONTRACT_FILE)
    if not path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "contract file does not exist", "path": str(path)},
        )

    try:
        contract = FeatureContract.from_json(path.read_text())
    except (json.JSONDecodeError, ContractError, KeyError) as exc:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": f"contract file is unreadable: {exc}"},
        )

    if not contract.created_at:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "contract has no created_at timestamp"},
        )

    max_staleness_days = load_admission_rules().max_staleness_days
    created = datetime.fromisoformat(contract.created_at)
    age_days = (datetime.now(UTC) - created).total_seconds() / 86_400

    passed = age_days <= max_staleness_days
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "created_at": contract.created_at,
            "age_days": round(age_days, 1),
            "max_staleness_days": max_staleness_days,
        },
    )


@asset_check(
    asset=feature_contract, 
    blocking=True,
    description="The contract's fingerprint must match its contents, and its admitted set must be large enough to train on.",
)
def feature_contract_integrity() -> AssetCheckResult:
    """The contract's fingerprint must match its contents, and its admitted set
    must be large enough to train on.

    A fingerprint mismatch means someone edited the JSON by hand — which is fine
    as a human override, but it has to be re-signed (regenerated) to prove the
    edit was intentional.  An empty admitted set means the audits rejected
    everything, which is a data problem, not a model problem.
    """
    path = Path(CONTRACT_FILE)
    if not path.exists():
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "contract file does not exist"},
        )

    try:
        contract = FeatureContract.from_json(path.read_text())
    except ContractError as exc:
        # from_dict raises ContractError on fingerprint mismatch
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": f"fingerprint mismatch: {exc}"},
        )
    except (json.JSONDecodeError, KeyError) as exc:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": f"contract file is unreadable: {exc}"},
        )

    minimum = load_admission_rules().min_admitted_features
    admitted = len(contract.training_features())
    if admitted < minimum:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={
                "admitted": admitted,
                "min_admitted_features": minimum,
                "reason": (
                    f"only {admitted} features admitted, below minimum of "
                    f"{minimum} — the audits may have rejected too aggressively"
                ),
            },
        )

    return AssetCheckResult(
        passed=True,
        metadata={
            "fingerprint": contract.fingerprint(),
            "admitted": admitted,
            "rejected": len(contract.rejections()),
        },
    )
