"""The audit assets, and the contract they feed.

These do not run with training. They run when the **data** changes, and their output is a
committed artefact. That is not a cost decision — time consistency over 377 columns takes
about four minutes on eight cores, which no pipeline would notice. It is that the answer is
a property of the data, so re-deriving it on every run recomputes a constant and gives the
run one more way to fail.

They are therefore materialized manually. What stops "manual" from meaning "forgotten" is
the staleness check on ``feature_contract``, not anyone's memory.

One asset per technique, fanning into one contract:

    model_input ──┬──→ time_consistency_report   ──┐
                  ├──→ distribution_shift_report ──┼──→ feature_contract
                  └──→ redundancy_report         ──┘
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import replace
from pathlib import Path

import polars as pl
from dagster import AssetKey, MetadataValue, Output, asset
from google.cloud import bigquery

from fraud_detection.core.feature_contract import (
    FeatureContract,
    from_admission_rules,
    from_distribution_shift,
    from_segment_qualification,
    from_time_consistency,
    load_admission_rules,
)
from fraud_detection.core.feature_contract.declaration import declare_columns
from fraud_detection.core.schema import (
    CLIENT_ENTITY_ANCHOR,
    CLIENT_ENTITY_COMPONENTS,
    FEATURES_DATASET,
    LABEL_COLUMN,
    MODEL_INPUT_TABLE,
    TIME_COLUMN,
    qualified,
)
from fraud_detection.evaluation.distribution_shift import Reference
from fraud_detection.evaluation.redundancy import (
    audit_partition,
    from_redundancy,
    load_partition,
    select_representatives,
)
from fraud_detection.evaluation.segment_qualification import qualify
from fraud_detection.evaluation.time_consistency import scan, time_windows
from fraud_detection.feature_engineering.derivations import apply_derivations
from fraud_detection.orchestration import contract_catalog
from fraud_detection.orchestration.catalog import (
    BIGQUERY,
    CODE_VERSION,
    FEATURE_PLATFORM,
)
from fraud_detection.orchestration.resources import BigQueryResource

GROUP = "feature_validation"
from fraud_detection.core.config import get_orchestration_params

_audit_cfg = get_orchestration_params("feature_audit")
PARTITION_FILE = Path(_audit_cfg["partition_file"])
CONTRACT_FILE = Path(_audit_cfg["contract_file"])

# The audit report tables, named as constants for the same reason the seam uses them: an
# asset that receives its upstream's table name as a *value* depends on the orchestrator's
# local storage as well as on the data. That storage is disposable -- clearing
# .dagster_home, or running on a different machine, breaks the run even though every table
# in BigQuery is intact. The real artefact is the table; the name of the table is a
# constant; so depend on the key and name the table yourself.
TIME_CONSISTENCY_TABLE = "time_consistency_report"
DISTRIBUTION_SHIFT_TABLE = "distribution_shift_report"
REDUNDANCY_AUDIT_TABLE = "redundancy_audit"

# Thresholds live in references/audit-rules.toml, not here. A constant in an asset file
# cannot be recorded in the contract it produced, which is why every fragment carries
# `params: {}` and no contract could say which numbers made it.

# The audit pulls data into the orchestrator's process, which the rest of the pipeline
# deliberately avoids -- feature engineering is one BigQuery statement whose result never
# leaves the warehouse. The exception is accepted because LightGBM cannot be expressed in
# SQL, and it is bounded: two windows of ~100k rows over ~440 columns is a few hundred MB.
# It is also the first asset that makes "where does Dagster run" a real question rather
# than a deferred one. See docs/architecture.md.
#
# No EXCEPT list: `split` and `card_seen_in_train` do not exist
# in `features.model_input` -- they belong to `split_assignment` -- so every audit failed
# with a 400 before running a single check. What actually keeps identifiers, the label and
# the experiment metadata out of the audited set is `declare_columns`, via
# `schema.EXCLUDED_COLUMNS`, applied to the frame after it lands. One place, and the same
# one the contract itself is built from.
AUDIT_COLUMNS_SQL = """
SELECT *
FROM `{table}`
"""


def _table(resource: BigQueryResource, name: str) -> str:
    return qualified(resource.project, FEATURES_DATASET, name)


def _load(resource: BigQueryResource, table: str) -> pl.DataFrame:
    cache_dir = Path("data/local/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    safe_table_name = table.replace("`", "").replace(".", "_")
    cache_file = cache_dir / f"{safe_table_name}.parquet"
    
    ttl_seconds = int(os.getenv("DAGSTER_CACHE_TTL_SECONDS", "86400")) # 1 day default
    
    if cache_file.exists():
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < ttl_seconds:
            return pl.read_parquet(cache_file)
            
    query = AUDIT_COLUMNS_SQL.format(table=table)
    frame = pl.from_arrow(resource.get_client().query(query).to_arrow())
    frame.write_parquet(cache_file)
    return frame


def _write(resource: BigQueryResource, frame: pl.DataFrame, name: str) -> None:
    client = resource.get_client()
    job = client.load_table_from_dataframe(
        frame.to_pandas(),
        _table(resource, name),
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()


def _declare(frame: pl.DataFrame, rules) -> dict:
    """The declaration the contract is built against, over the *derived* frame.

    Derivations are applied before this rather than after, so a column the contract defines
    is audited by exactly the same five checks as one the table already had. A derived
    feature admitted without being audited would be the one column in the contract nothing
    had ever checked.
    """
    return declare_columns(
        {c: str(frame.schema[c]) for c in frame.columns},
        derived=frozenset(d.name for d in rules.derivations),
    )


def _feature_columns(frame: pl.DataFrame, rules) -> list[str]:
    return list(_declare(frame, rules))


def _partition_digest() -> str:
    """Hash of the file that defines the redundancy groups.

    The redundancy fragment records `{"method": "nan-group then within-group
    correlation"}` -- prose, not a parameter. The actual parameter of this check is the
    partition, and it lives in a file, so the contract could not tell you which grouping
    produced its rejections. That also broke the fingerprint's own promise: a partition
    edited without the admitted set moving left the fingerprint unchanged, which is exactly
    the case the policy is in the payload to catch.
    """
    return hashlib.sha256(PARTITION_FILE.read_bytes()).hexdigest()[:16]


def _time_consistency_qualification(frame: pl.DataFrame, rules, rejections) -> dict:
    """Do the inversions reproduce under a different window?

    `Fragment.qualification` says it carries evidence about whether an audit's verdicts are
    reproducible, and it shipped empty on every fragment. This fills it for the check that
    needs it most: the verdict is a threshold crossing, so a column sitting near the margin
    can flip on the window alone.

    **Only the per-column rejections are retested**, and that restriction is the whole
    point rather than an optimisation. Under `reject_by_block` most rejections are a
    column's block falling, not the column inverting on its own — asking whether such a
    column re-inverts in isolation tests a decision nobody made. The first version of this
    retested all 133 and reported 25% reproduction, which read as "the audit is noise" when
    it actually measured the wrong thing.

    Only rejected columns are scanned: the question is whether *these* rejections hold, and
    re-scanning all 479 would triple the audit to answer something nobody asked.
    """
    per_column = [r.column for r in rejections if r.unit == "column"]
    by_block = len(rejections) - len(per_column)
    if not per_column:
        return {"rejections_by_block": by_block, "rejections_retested": 0}

    train_window = rules.time_windows["train"]
    width = train_window[1] - train_window[0]
    # Wider windows on both ends: more rows per fit, at the cost of the gap between them.
    alternative = round(width * 1.5, 3)
    train, holdout = time_windows(
        frame, TIME_COLUMN, train=(0.0, alternative), holdout=(1 - alternative, 1.0)
    )

    report = scan(train, holdout, per_column, LABEL_COLUMN, n_jobs=-1)
    verdicts = dict(zip(report["feature"].to_list(), report["verdict"].to_list()))
    reproduced = sum(1 for c in per_column if verdicts.get(c) == "inverted")

    return {
        "reproduced_at_window": alternative,
        "rejections_retested": len(per_column),
        "rejections_reproduced": reproduced,
        "reproduced_share": round(reproduced / len(per_column), 4),
        # Recorded, not tested: a block falls on its family's behaviour, so a per-column
        # re-scan cannot confirm or deny it.
        "rejections_by_block": by_block,
    }


def _distribution_shift_qualification(frame: pl.DataFrame, rules, features: list[str]) -> dict:
    """Does the PSI verdict survive a different binning?

    PSI is deterministic given its edges, so "does it reproduce" is a question about the
    bins rather than about sampling. Refitting the reference at a different bin count and
    counting how many rejections survive says whether a column cleared the threshold on the
    strength of its shift or on the strength of the histogram.
    """
    train, holdout = time_windows(frame, TIME_COLUMN, **rules.time_windows)

    def rejected_at(bins: int) -> set[str]:
        report = Reference.fit(train, features, bins=bins).psi(holdout)
        # Through the adapter, not a `psi > threshold` filter of its own. Polars orders
        # NaN above every number, so the naive filter counted all 165 degenerate columns
        # as drift -- the outcome `reject_degenerate = false` prevents. The adapter
        # separates "undefined" from "large" and honours the policy.
        fragment = from_distribution_shift(
            report,
            psi_threshold=rules.psi_threshold,
            reject_degenerate=rules.reject_degenerate,
        )
        return {r.column for r in fragment.rejections}

    base, alternative = rejected_at(10), rejected_at(20)
    if not base:
        return {"bins_compared": [10, 20], "rejections_at_default_bins": 0}

    return {
        "bins_compared": [10, 20],
        "rejections_at_default_bins": len(base),
        "rejections_at_alternative_bins": len(alternative),
        "reproduced_share": round(len(base & alternative) / len(base), 4),
    }


@asset(
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name=GROUP, 
    kinds=BIGQUERY | {"lightgbm", "polars"}, 
    deps=[AssetKey(["fraud_detection", "model_input"])],
    description="Which features rank one way in the past and the opposite way later.",
)
def time_consistency_report(bigquery_resource: BigQueryResource):
    """Which features rank one way in the past and the opposite way later.

    Metadata carries the qualification numbers, not just the verdict counts: a holdout with
    too few positives produces verdicts that do not reproduce, and the run that produced
    them should say so on its own materialization.
    """
    rules = load_admission_rules()
    model_input = _table(bigquery_resource, MODEL_INPUT_TABLE)
    frame = apply_derivations(_load(bigquery_resource, model_input), rules.derivations)
    features = _feature_columns(frame, rules)
    train, holdout = time_windows(frame, "TransactionDT", **rules.time_windows)

    report = scan(train, holdout, features, "isFraud", n_jobs=-1)
    _write(bigquery_resource, report, TIME_CONSISTENCY_TABLE)

    counts_df = report.get_column("verdict").value_counts()
    counts = {row["verdict"]: row["count"] for row in counts_df.iter_rows(named=True)}
    day = 86_400
    return Output(
        _table(bigquery_resource, TIME_CONSISTENCY_TABLE),
        metadata={
            "features": len(features),
            **{f"verdict.{k}": int(v) for k, v in counts.items()},
            "holdout_positives": int(holdout.get_column("isFraud").sum()),
            "train_days": round(
                float(train.get_column("TransactionDT").max() - train.get_column("TransactionDT").min()) / day, 1
            ),
            "holdout_days": round(
                float(holdout.get_column("TransactionDT").max() - holdout.get_column("TransactionDT").min()) / day, 1
            ),
            "gap_days": round(
                float(holdout.get_column("TransactionDT").min() - train.get_column("TransactionDT").max()) / day, 1
            ),
            "worst": MetadataValue.md(report.head(10).to_pandas().to_markdown(index=False)),
        },
    )


@asset(
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name=GROUP, 
    kinds=BIGQUERY | {"polars"}, 
    deps=[AssetKey(["fraud_detection", "model_input"])],
    description="PSI of the later window against a reference pinned on the earlier one.",
)
def distribution_shift_report(bigquery_resource: BigQueryResource):
    """PSI of the later window against a reference pinned on the earlier one.

    The reference is serialized alongside the report. Recomputing bucket edges per call
    defines every bucket to hold a tenth of whatever it is handed, so a sample compares as
    unshifted against itself no matter how far it moved.
    """
    rules = load_admission_rules()
    model_input = _table(bigquery_resource, MODEL_INPUT_TABLE)
    frame = apply_derivations(_load(bigquery_resource, model_input), rules.derivations)
    features = _feature_columns(frame, rules)
    train, holdout = time_windows(frame, "TransactionDT", **rules.time_windows)

    reference = Reference.fit(train, features, meta={"table": model_input, **rules.time_windows})
    report = reference.psi(holdout)

    _write(bigquery_resource, report, DISTRIBUTION_SHIFT_TABLE)

    return Output(
        _table(bigquery_resource, DISTRIBUTION_SHIFT_TABLE),
        metadata={
            "features": len(features),
            "psi_above_threshold": int((report.get_column("psi") > rules.psi_threshold).sum()),
            "psi_degenerate": int(report.get_column("psi").null_count()),
            "psi_max": round(float(report.get_column("psi").max()), 4),
            "worst": MetadataValue.md(report.head(10).to_pandas().to_markdown(index=False)),
        },
    )


@asset(
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"], 
    group_name=GROUP, 
    kinds=BIGQUERY | {"polars"}, 
    deps=[AssetKey(["fraud_detection", "model_input"])],
    description="Collapse the V block to one representative per correlated group, and audit the partition that defines those groups.",
)
def redundancy_report(bigquery_resource: BigQueryResource):
    """Collapse the V block to one representative per correlated group, and audit the
    partition that defines those groups."""
    rules = load_admission_rules()
    frame = apply_derivations(
        _load(bigquery_resource, _table(bigquery_resource, MODEL_INPUT_TABLE)), rules.derivations
    )
    partition = load_partition(PARTITION_FILE).with_uncovered()

    kept, dropped = select_representatives(frame, partition.groups)
    audit = audit_partition(frame, partition)
    _write(bigquery_resource, audit, REDUNDANCY_AUDIT_TABLE)

    measurable = audit.filter(pl.col("holds").is_not_null())
    return Output(
        _table(bigquery_resource, REDUNDANCY_AUDIT_TABLE),
        metadata={
            "kept": len(kept),
            "dropped": len(dropped),
            "groups": len(partition.groups),
            "groups_measurable": len(measurable),
            "partition_holds": round(float(measurable.get_column("holds").mean()), 4)
            if len(measurable)
            else None,
            "provenance": MetadataValue.json(partition.provenance),
        },
    )


@asset(
    owners=FEATURE_PLATFORM,
    code_version=CODE_VERSION,
    key_prefix=["fraud_detection"],
    group_name=GROUP,
    kinds={"json"},
    description="One admitted list, assembled from every audit that ran.",
    deps=[
        AssetKey(["fraud_detection", "model_input"]),
        AssetKey(["fraud_detection", "time_consistency_report"]),
        AssetKey(["fraud_detection", "distribution_shift_report"]),
        AssetKey(["fraud_detection", "redundancy_report"]),
    ],
)
def feature_contract(bigquery_resource: BigQueryResource):
    """One admitted list, assembled from every audit that ran.

    Written to a file in the repository rather than to a bucket, so a change arrives as a
    reviewable diff. "22 columns rejected by time consistency" is a thing somebody should
    approve, not a thing that happens quietly between runs.
    """
    rules = load_admission_rules()
    model_input = _table(bigquery_resource, MODEL_INPUT_TABLE)
    frame = apply_derivations(_load(bigquery_resource, model_input), rules.derivations)
    declared = _declare(frame, rules)
    partition = load_partition(PARTITION_FILE).with_uncovered()
    client = bigquery_resource.get_client()

    # Fragment order is the order rejections are attributed in: the first check to object
    # owns the column. Policy goes first, because "we chose to exclude this" should not be
    # displaced in the record by a check that happened to also reject it.
    fragments = [from_admission_rules(rules.blacklist)] if rules.blacklist else []

    if rules.enabled("time_consistency"):
        tc = pl.from_arrow(client.query(
            f"SELECT * FROM `{_table(bigquery_resource, TIME_CONSISTENCY_TABLE)}`"
        ).to_arrow())
        fragment = from_time_consistency(
            tc,
            blocks=_blocks(partition) if rules.time_consistency.get("reject_by_block", True) else None,
            params=dict(rules.time_consistency),
        )
        fragments.append(
            replace(
                fragment,
                qualification=_time_consistency_qualification(
                    frame, rules, fragment.rejections
                ),
            )
        )

    if rules.enabled("distribution_shift"):
        psi = pl.from_arrow(client.query(
            f"SELECT * FROM `{_table(bigquery_resource, DISTRIBUTION_SHIFT_TABLE)}`"
        ).to_arrow())
        fragments.append(
            from_distribution_shift(
                psi,
                psi_threshold=rules.psi_threshold,
                reject_degenerate=rules.reject_degenerate,
                params=dict(rules.distribution_shift),
                qualification=_distribution_shift_qualification(
                    frame, rules, _feature_columns(frame, rules)
                ),
            )
        )

    # No report asset and no table behind this one, unlike the three above. Those cache
    # because they are expensive -- a model fit per column, PSI binning, pairwise
    # correlation. This is a rank statistic over a frame already in memory and runs in
    # seconds, so a table would be a second copy of the truth to keep in step for no gain.
    if rules.segment_enabled and rules.segment_column in frame.columns:
        scored = qualify(
            frame,
            _feature_columns(frame, rules),
            LABEL_COLUMN,
            rules.segment_column,
        )
        # The largest segment, chosen by row count rather than named in config: the audit
        # is about the population most of the traffic sits in, and hard-coding "W" would
        # make the policy wrong the day the mix changes.
        sizes = frame.get_column(rules.segment_column).value_counts(sort=True)
        largest = str(sizes.get_column(rules.segment_column)[0])
        fragments.append(
            from_segment_qualification(
                scored,
                segment=largest,
                pooled_floor=rules.segment_pooled_floor,
                max_drop=rules.segment_max_drop,
                reject_unmeasurable=rules.segment_reject_unmeasurable,
                reject=rules.segment_reject,
                params=dict(rules.segment_qualification),
                qualification={
                    "segment_rows": int(sizes.get_column("count")[0]),
                    "segment_share": round(float(sizes.get_column("count")[0]) / len(frame), 4),
                },
            )
        )

    if rules.enabled("redundancy"):
        _, dropped = select_representatives(frame, partition.groups)
        # The partition audit already ran in `redundancy_report` and its verdict is the
        # qualification this fragment was missing: a grouping that does not hold on the
        # data is a grouping whose rejections mean nothing.
        audit = pl.from_arrow(client.query(
            f"SELECT * FROM `{_table(bigquery_resource, REDUNDANCY_AUDIT_TABLE)}`"
        ).to_arrow())
        measurable = audit.filter(pl.col("holds").is_not_null())
        fragments.append(
            from_redundancy(
                dropped,
                partition,
                params={
                    "partition_file": PARTITION_FILE.name,
                    "partition_digest": _partition_digest(),
                    "groups": len(partition.groups),
                },
                qualification={
                    "groups_measurable": len(measurable),
                    "partition_holds": round(float(measurable["holds"].mean()), 4)
                    if len(measurable)
                    else None,
                },
            )
        )

    contract = FeatureContract.build(
        declared,
        fragments,
        # `dataset.table`, not the fully qualified name: the project id would put a GCP
        # account number into a committed artefact, and would make the contract differ
        # between environments for a reason that says nothing about the features. It is
        # provenance, and it is not in the fingerprint either way.
        data={"table": f"{FEATURES_DATASET}.{MODEL_INPUT_TABLE}", "rows": len(frame)},
        entity={"columns": list(CLIENT_ENTITY_COMPONENTS), "anchor": CLIENT_ENTITY_ANCHOR},
        admission_rules=rules.as_dict(),
        overrides=rules.overrides,
    )

    CONTRACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_FILE.write_text(contract.to_json())

    return Output(
        str(CONTRACT_FILE),
        metadata={
            "fingerprint": contract.fingerprint(),
            "declared": len(declared),
            "admitted": len(contract.training_features()),
            "rejected": len(contract.rejections()),
            "overridden": len(contract.overrides()),
            "rules_digest": rules.digest(),
            "checks": ", ".join(f.check for f in contract.fragments),
            "request_fields": len(contract.request_fields()),
            "retrieved_fields": len(contract.retrieved_fields()),
            "path": str(CONTRACT_FILE),
            # The column-level view, handed to the catalog in the shape it renders. The
            # contract has always recorded which columns survived and why; this is the same
            # record expressed as `TableSchema` and `TableColumnLineage` so the graph can
            # answer "why is this column not in the model" without opening the JSON.
            **contract_catalog.contract_metadata(
                contract, AssetKey(["fraud_detection", "model_input"])
            ),
        },
    )


def _blocks(partition) -> dict[str, list[str]]:
    return {name: [c for g in groups for c in g] for name, groups in partition.blocks.items()}
