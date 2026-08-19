"""Write the frame the audits are supposed to see.

The audits live in ``analysis/`` and are R. What they must audit is not the table as
BigQuery stores it but the table *as the model receives it* — which means with the
declared derivations applied. Eight of the thirty derived columns were rejected by an
audit the last time the whole set was measured, so handing R the raw export would
admit them without anyone having looked.

Deliberately not solved by reimplementing the derivations in R. A derivation is a
transformation the pipeline performs, and a second implementation of it would be a
second thing to keep in step — precisely the duplication the split was designed to
avoid. Python owns the transformations, R audits their output, and this command is
the handover.

    uv run export-audit-frame --out data/local/cache/audit_frame.parquet

Then point the R side at it:

    FRAUDAUDIT_PARQUET=data/local/cache/audit_frame.parquet \\
        Rscript -e 'targets::tar_make()'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from fraud_detection.contract import load_admission_rules
from fraud_detection.features.derivations import apply_derivations

AUDIT_FRAME_PATH = Path("data/local/cache/audit_frame.parquet")


def build_audit_frame(frame: pl.DataFrame, rules=None) -> pl.DataFrame:
    """Apply the declared derivations, so the audits see what the model will.

    One function, two callers: the Dagster asset reads `model_input` from BigQuery
    and this command reads a local export. Neither reimplements the step, because a
    derivation applied one way in the pipeline and another way in the audit is the
    duplication this whole boundary exists to avoid.

    Missing inputs raise rather than being skipped: a derivation that silently does
    not happen produces an audit of fewer columns than the contract will declare.
    """
    return apply_derivations(frame, (rules or load_admission_rules()).derivations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/local/cache/fraud-detection-504617_features_model_input.parquet"),
        help="an export of features.model_input",
    )
    parser.add_argument("--out", type=Path, default=AUDIT_FRAME_PATH)
    args = parser.parse_args(argv)

    frame = pl.read_parquet(args.source)
    before = frame.width
    frame = build_audit_frame(frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    print(
        f"{args.out}: {frame.height:,} rows, {frame.width} columns "
        f"({frame.width - before} derived added)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
