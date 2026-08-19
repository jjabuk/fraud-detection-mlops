"""Turn the audit fragments into the contract the pipeline reads.

This is the seam. Everything upstream of it is statistics and lives in
``analysis/`` as R; everything downstream is the model lifecycle and lives here.
The command itself decides nothing: it merges verdicts that were already reached,
applies the admission policy, and stamps a fingerprint.

Why the fingerprint is stamped here rather than written by the audits.
``FeatureContract.fingerprint`` hashes a canonical rendering of the admitted set
and the policy, and ``from_dict`` refuses a file whose stored hash disagrees with
its contents — that is how a hand-edited contract is caught. If the R side wrote
the hash, that detector would depend on two JSON serialisers agreeing on key
order and float formatting forever. One writer, one hash.

The declaration is read from the audit's output rather than from BigQuery on
purpose: stamping is assembly, and assembly should not need cloud credentials or
the data. What it does need is that the declaration and the fragments came from
the same run, which is what ``--declaration`` and ``--fragments`` being siblings
means.

    uv run stamp-contract \\
        --declaration analysis/out/declaration.json \\
        --fragments analysis/out/fragments \\
        --out references/feature-contract.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fraud_detection.contract import (
    ContractError,
    FeatureContract,
    from_admission_rules,
    load_admission_rules,
    read_fragments,
)
from fraud_detection.schema import (
    CLIENT_ENTITY_ANCHOR,
    CLIENT_ENTITY_COMPONENTS,
)

#: The order checks are credited in. It does not change which columns are admitted
#: — a column caught twice is out either way — but it decides which check the
#: report attributes each rejection to, so it is stated rather than left to the
#: order the filesystem happens to list.
PRECEDENCE = (
    "time_consistency",
    "distribution_shift",
    "redundancy",
    "segment_qualification",
)


def declared_from(document: dict) -> dict[str, tuple[str, str]]:
    """Read the column declaration the audit recorded.

    Source cannot be inferred from the data — no statistic knows whether a column
    arrives with the request or has to be looked up — and getting it wrong is how a
    request schema ends up asking callers for features they cannot have. It comes
    from the BigQuery schemas, which is where that fact actually lives.
    """
    columns = document.get("columns")
    if not columns:
        raise ContractError("the declaration lists no columns")
    return {c["name"]: (c["source"], c["dtype"]) for c in columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--declaration", type=Path, default=Path("analysis/out/declaration.json")
    )
    parser.add_argument("--fragments", type=Path, default=Path("analysis/out/fragments"))
    parser.add_argument("--out", type=Path, default=Path("references/feature-contract.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="stamp into memory and compare with --out instead of writing; exits "
        "non-zero when they differ, which is what CI runs",
    )
    args = parser.parse_args(argv)

    declaration = json.loads(args.declaration.read_text())
    declared = declared_from(declaration)
    rules = load_admission_rules()

    fragments = read_fragments(args.fragments, PRECEDENCE)
    # The blacklist is policy, not a measurement, so it is applied here and not by
    # any audit. It comes last: a column somebody decided to exclude by hand should
    # be attributed to that decision only when no check objected on its own.
    fragments = [*fragments, from_admission_rules(rules.blacklist)]

    contract = FeatureContract.build(
        declared,
        fragments,
        data=declaration.get("data", {}),
        entity={"columns": list(CLIENT_ENTITY_COMPONENTS), "anchor": CLIENT_ENTITY_ANCHOR},
        admission_rules=rules.as_dict(),
        overrides=rules.overrides,
    )

    rendered = contract.to_json()
    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        current = FeatureContract.from_json(args.out.read_text())
        if current.fingerprint() != contract.fingerprint():
            print(
                f"contract is stale: file is {current.fingerprint()}, "
                f"the fragments stamp to {contract.fingerprint()}",
                file=sys.stderr,
            )
            return 1
        print(f"contract is current at {contract.fingerprint()}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(
        f"{args.out}: {len(contract.training_features())} admitted of {len(declared)} "
        f"declared, fingerprint {contract.fingerprint()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
