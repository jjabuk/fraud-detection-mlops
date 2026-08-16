"""Synthetic ground truth: we build the families, so we know which columns are redundant."""

from __future__ import annotations

import numpy as np
import polars as pl

from fraud_detection.evaluation.redundancy import (
    Partition,
    audit_partition,
    load_partition,
    nan_groups,
    select_representatives,
)

N = 3_000
REFERENCE = "references/column-groups-v.json"

def make(seed: int = 0) -> pl.DataFrame:
    """Two families. Inside each, columns are near-copies of a driver.

    - family A (`a1..a4`): never null. `a1,a2` share a driver; `a3,a4` share another.
    - family B (`b1,b2`): 40% null on the same rows, both driven by one signal.
    """
    rng = np.random.default_rng(seed)
    d1, d2, d3 = (rng.normal(size=N) for _ in range(3))
    noise = lambda s: s + rng.normal(scale=0.01, size=N)

    df = pl.DataFrame({
        "a1": noise(d1).round(3),
        "a2": noise(d1).round(1),  # coarser: fewer distinct values
        "a3": noise(d2).round(3),
        "a4": noise(d2).round(1),
        "b1": noise(d3).round(3),
        "b2": noise(d3).round(1),
    })
    hidden = rng.random(N) < 0.4
    df = df.with_columns([
        pl.when(pl.Series(hidden)).then(None).otherwise(pl.col("b1")).alias("b1"),
        pl.when(pl.Series(hidden)).then(None).otherwise(pl.col("b2")).alias("b2"),
    ])
    return df

PARTITION = Partition(
    blocks={"A": (("a1", "a2"), ("a3", "a4")), "B": (("b1", "b2"),)},
    provenance={"author": "test"},
)

def test_nan_groups_separates_families_by_null_count():
    groups = nan_groups(make(), ["a1", "a2", "a3", "a4", "b1", "b2"])

    assert len(groups) == 2
    assert sorted(groups[0]) == ["a1", "a2", "a3", "a4"]
    assert sorted(next(v for k, v in groups.items() if k > 0)) == ["b1", "b2"]

def test_nan_groups_puts_the_largest_family_first():
    groups = nan_groups(make(), ["a1", "a2", "a3", "a4", "b1", "b2"])
    assert len(next(iter(groups.values()))) == 4

def test_the_representative_is_the_column_with_the_most_distinct_values():
    df = make()
    kept, dropped = select_representatives(df, PARTITION.groups)

    assert kept == ["a1", "a3", "b1"]
    assert dropped == ["a2", "a4", "b2"]

def test_selection_is_deterministic():
    df = make()
    assert select_representatives(df, PARTITION.groups) == select_representatives(df, PARTITION.groups)

def test_one_column_per_group_survives():
    df = make()
    kept, _ = select_representatives(df, PARTITION.groups)
    assert len(kept) == len(PARTITION.groups)

# ---- the audit ----------------------------------------------------------------

def test_a_true_partition_holds():
    audit = audit_partition(make(), PARTITION, sample=None)
    auditable = audit.filter(pl.col("block") == "A")

    assert set(auditable.get_column("holds")) == {True}
    assert (auditable.get_column("min_within") > 0.9).all()

def test_a_block_with_a_single_group_cannot_be_audited():
    """There is nothing outside the group to compare against, so the verdict is not
    False — it is unknown, and saying so beats inventing a pass."""
    audit = audit_partition(make(), PARTITION, sample=None)
    lone = audit.filter(pl.col("block") == "B").row(0, named=True)

    assert lone["holds"] is None
    assert lone["max_outside"] is None
    assert lone["min_within"] > 0.9  # the group itself is still coherent

def test_a_scrambled_partition_does_not_hold():
    """Mixing members of two drivers into one group should be visible as a failure."""
    wrong = Partition(blocks={"A": (("a1", "a3"), ("a2", "a4"))}, provenance={})
    audit = audit_partition(make(), wrong, sample=None)

    assert not audit.get_column("holds").any()
    assert (audit.get_column("min_within") < audit.get_column("max_outside")).all()

def test_the_audit_reports_a_row_per_group():
    audit = audit_partition(make(), PARTITION, sample=None)
    assert len(audit) == len(PARTITION.groups)
    assert set(audit.get_column("block")) == {"A", "B"}

# ---- coverage and the contract fragment ----------------------------------------

# ---- the pinned reference file --------------------------------------------------

def test_the_pinned_partition_loads_and_is_attributed():
    p = load_partition(REFERENCE)

    assert len(p.blocks) == 21
    assert len(p.groups) == 128
    assert p.provenance["author"] and p.provenance["url"]

def test_the_pinned_partition_covers_every_v_column_once_uncovered_is_restored():
    p = load_partition(REFERENCE)

    assert len(p.columns) == 338
    assert p.uncovered == ("V155",)
    assert len(p.with_uncovered().columns) == 339
    assert len(set(p.with_uncovered().columns)) == 339  # no column in two groups

