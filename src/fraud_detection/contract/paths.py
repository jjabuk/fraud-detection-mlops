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

__all__ = ["CONTRACT_FILE", "FRAGMENT_DIR"]

_cfg = get_orchestration_params("feature_audit")

#: The stamped contract, committed so a change to the admitted set arrives as a diff.
CONTRACT_FILE = Path(_cfg["contract_file"])

#: Where the statistical audits leave their verdicts, for `stamp-contract` to merge.
FRAGMENT_DIR = Path(_cfg.get("fragment_dir", "analysis/out/fragments"))
