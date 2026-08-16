"""Tests for the feature contract asset checks.

The checks read a committed JSON file rather than querying a resource, so testing
them is straightforward: write a file, run the check function, assert the result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from fraud_detection.core.feature_contract import (
    FeatureContract,
    Fragment,
    Rejection,
    load_admission_rules,
)
from fraud_detection.orchestration.assets.feature_contract_check import (
    feature_contract_freshness,
    feature_contract_integrity,
)

# The checks read references/audit-policy.toml, so the tests read it too rather than
# pinning a second copy of the numbers that would drift from the policy silently.
_RULES = load_admission_rules()
MAX_STALENESS_DAYS = _RULES.max_staleness_days
MIN_ADMITTED_FEATURES = _RULES.min_admitted_features


def _minimal_contract(
    *,
    n_features: int = 100,
    created_at: str | None = None,
    reject: int = 0,
) -> FeatureContract:
    """Build a small but valid contract."""
    columns = {f"f{i}": ("request", "float") for i in range(n_features)}
    fragments = ()
    if reject > 0:
        fragments = (
            Fragment(
                check="test",
                rejections=tuple(
                    Rejection(column=f"f{i}", by="test") for i in range(reject)
                ),
            ),
        )
    contract = FeatureContract.build(columns, fragments)
    if created_at is not None:
        # Override the auto-generated created_at
        contract = FeatureContract(
            columns=contract.columns,
            data=contract.data,
            fragments=contract.fragments,
            entity=contract.entity,
            created_at=created_at,
            version=contract.version,
        )
    return contract


def _write_contract(tmp_path: Path, contract: FeatureContract) -> Path:
    path = tmp_path / "references" / "feature-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contract.to_json())
    return path


# ---- freshness check --------------------------------------------------------


class TestFreshnessCheck:
    def test_fresh_contract_passes(self, tmp_path):
        contract = _minimal_contract(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        path = _write_contract(tmp_path, contract)
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_freshness()
        assert result.passed is True

    def test_stale_contract_fails(self, tmp_path):
        stale = datetime.now(UTC) - timedelta(days=MAX_STALENESS_DAYS + 10)
        contract = _minimal_contract(
            created_at=stale.isoformat(timespec="seconds"),
        )
        path = _write_contract(tmp_path, contract)
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_freshness()
        assert result.passed is False

    def test_missing_file_fails(self, tmp_path):
        path = tmp_path / "does-not-exist.json"
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_freshness()
        assert result.passed is False

    def test_no_created_at_fails(self, tmp_path):
        contract = _minimal_contract()
        # Write with created_at wiped
        path = tmp_path / "references" / "feature-contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        d = contract.to_dict()
        d["created_at"] = ""
        # Also strip the fingerprint so from_dict doesn't complain about mismatch
        # Actually, we can just remove it entirely
        path.write_text(json.dumps(d))
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_freshness()
        assert result.passed is False

    def test_corrupt_json_fails(self, tmp_path):
        path = tmp_path / "references" / "feature-contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json")
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_freshness()
        assert result.passed is False


# ---- integrity check --------------------------------------------------------


class TestIntegrityCheck:
    def test_valid_contract_passes(self, tmp_path):
        contract = _minimal_contract(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        path = _write_contract(tmp_path, contract)
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_integrity()
        assert result.passed is True
        assert result.metadata["admitted"].value == 100

    def test_tampered_fingerprint_fails(self, tmp_path):
        contract = _minimal_contract()
        path = tmp_path / "references" / "feature-contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        d = contract.to_dict()
        d["fingerprint"] = "deadbeefdeadbeef"
        path.write_text(json.dumps(d))
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_integrity()
        assert result.passed is False

    def test_too_few_admitted_fails(self, tmp_path):
        # Create a contract with fewer than MIN_ADMITTED_FEATURES admitted
        contract = _minimal_contract(
            n_features=MIN_ADMITTED_FEATURES + 10,
            reject=MIN_ADMITTED_FEATURES,
        )
        path = _write_contract(tmp_path, contract)
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_integrity()
        assert result.passed is False

    def test_missing_file_fails(self, tmp_path):
        path = tmp_path / "does-not-exist.json"
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_integrity()
        assert result.passed is False

    def test_enough_admitted_passes(self, tmp_path):
        # Exactly at the minimum
        contract = _minimal_contract(
            n_features=MIN_ADMITTED_FEATURES,
            reject=0,
        )
        path = _write_contract(tmp_path, contract)
        with mock.patch(
            "fraud_detection.orchestration.assets.feature_contract_check.CONTRACT_FILE", path
        ):
            result = feature_contract_integrity()
        assert result.passed is True
        assert result.metadata["admitted"].value == MIN_ADMITTED_FEATURES
