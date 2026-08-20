"""Where the contract and its inputs live.

Two path constants, in the package that owns the artefacts rather than in the
Dagster asset that happens to materialise one of them. That is not tidiness: a
constant owned by an asset can only be read by importing the asset, which drags
Dagster and a cloud SDK into anything that wants to know where the contract is
-- including a command-line tool and a notebook, neither of which should need
either.
"""

from __future__ import annotations

from pathlib import Path

from fraud_detection.config import get_orchestration_params

__all__ = ["CONTRACT_FILE", "DECLARATION_FILE", "FRAGMENT_DIR"]

_cfg = get_orchestration_params("feature_audit")

#: The stamped contract, committed so a change to the admitted set arrives as a diff.
CONTRACT_FILE = Path(_cfg["contract_file"])

#: Where the statistical audits leave their verdicts, for `stamp-contract` to merge.
#: They are written by a separate repository, so this points outside this one and is only
#: read when somebody re-stamps by hand. See `audit_repo` in config/orchestration.toml.
FRAGMENT_DIR = Path(_cfg.get("fragment_dir", "../ieee-cis-fraud-detection-eda/out/fragments"))

#: The column list, sources and dtypes the same audit run declared, beside those fragments.
DECLARATION_FILE = Path(
    _cfg.get("declaration_file", "../ieee-cis-fraud-detection-eda/out/declaration.json")
)
