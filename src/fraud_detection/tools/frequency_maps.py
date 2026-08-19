"""Fit the frequency-encoding count tables and commit them.

Frequency encoding replaces a category with how often it occurs. That count has to come
from somewhere fixed: counting the frame being transformed gives one mapping for training
and another for the scoring period, and counting train and test together is transductive.
So the counts are fitted **once, on the training split only**, written to
`references/frequency-maps.json`, and committed — a change to the map arrives as a
reviewable diff, exactly like the feature contract.

    uv run build-frequency-maps

Re-run it when the split boundaries or the raw load change; the file records which split
and how many rows it was fitted on, so a stale map is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fraud_detection.features.derivations import FREQUENCY_MAP_FILE
from fraud_detection.registry.provenance import describe_code_version
from fraud_detection.schema import MODEL_INPUT_TABLE, SPLIT_TABLE

# High cardinality is the *precondition*, not the reason. One-hot on `DeviceInfo` (1,786
# levels) or `id_31` (130) would add that many mostly-zero columns where this adds one
# integer -- but that only argues frequency encoding is *possible* here, not that it is
# worth anything. The reason is measured: rarity has to predict fraud. It does for these
# five (id_31 runs 6.77x -> 2.17x base fraud rate across frequency bands, monotonically),
# and it does not for card1 (0.85 / 0.76 / 1.19 / 0.98) or card2, which were in this tuple
# until the question was actually asked. See config/feature-admission.toml.
#
# Low-cardinality columns are absent for the opposite reason: on a 5-level column this is
# strictly *less* expressive than the native categorical split it would replace, and it
# risks merging levels whose counts happen to be close.
FREQUENCY_COLUMNS = (
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_31",
)

# Values seen this rarely in training are dropped from the map, so they encode as null
# ("training has nothing to say") rather than as a count fitted on a handful of rows. A
# count of 1 is not a frequency, it is a fingerprint of one transaction, and a model that
# splits on it has memorised a row.
MIN_COUNT = 2


def build(project: str, columns=FREQUENCY_COLUMNS) -> dict:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    selected = ", ".join(f"m.`{c}`" for c in columns)
    query = f"""
    SELECT {selected}
    FROM `{project}.features.{MODEL_INPUT_TABLE}` AS m
    JOIN `{project}.features.{SPLIT_TABLE}` AS s USING (TransactionID)
    WHERE s.split = 'train'
    """
    frame = pl.from_arrow(client.query(query).result().to_arrow())

    maps: dict[str, dict[str, int]] = {}
    summary = {}
    for column in columns:
        values = frame.get_column(column).cast(pl.String).drop_nulls()
        counts = values.value_counts()
        table = {
            row[column]: int(row["count"])
            for row in counts.iter_rows(named=True)
            if row["count"] >= MIN_COUNT and row[column] is not None
        }
        maps[column] = table
        summary[column] = {
            "levels_seen": int(counts.height),
            "levels_kept": len(table),
            "coverage": round(sum(table.values()) / max(len(frame), 1), 4),
        }

    return {
        "fitted_on": {
            "table": f"{project}.features.{MODEL_INPUT_TABLE}",
            "split": "train",
            "rows": len(frame),
        },
        "min_count": MIN_COUNT,
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": describe_code_version(),
        "summary": summary,
        "maps": maps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=FREQUENCY_MAP_FILE)
    args = parser.parse_args()

    payload = build(os.environ["GCP_PROJECT_ID"])
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"fitted on {payload['fitted_on']['rows']:,} training rows")
    for column, stats in payload["summary"].items():
        print(
            f"  {column:16s} {stats['levels_kept']:>7,} / {stats['levels_seen']:>7,} levels kept"
            f"   coverage {stats['coverage']:.1%}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
