"""Handing the feature contract to Dagster in the shape its catalog can render.

The contract already records which columns reach the model, which were rejected, and by
which audit. That is column-level lineage — it was simply never expressed as
`TableSchema` and `TableColumnLineage`, so the graph could not show it and the JSON had to
be opened by hand.

Nothing here decides anything. It translates a record that already exists.
"""

from __future__ import annotations

from dagster import (
    MetadataValue,
    TableColumn,
    TableColumnDep,
    TableColumnLineage,
    TableSchema,
)

__all__ = ["contract_column_lineage", "contract_metadata", "contract_table_schema"]


def contract_table_schema(contract) -> TableSchema:
    """Every admitted column, with the audit trail for the ones that did not make it.

    Rejected columns are described rather than omitted: the catalog's job is to answer
    "why is this column not in the model", and a schema that lists only survivors cannot.
    """
    return TableSchema(
        columns=[
            TableColumn(
                name=column.name,
                type=column.dtype or "unknown",
                description=(
                    f"admitted, source={getattr(column.source, 'value', column.source)}"
                    if column.admitted
                    else f"rejected by {column.rejected_by} ({column.rejected_value})"
                ),
            )
            for column in contract.columns
        ]
    )


def contract_column_lineage(contract, upstream_asset_key) -> TableColumnLineage:
    """Which upstream column each admitted column comes from.

    One dependency per column, because every admitted column is either passed through from
    the joined table or derived from named inputs the contract records. A derived column
    names its inputs; a retrieved aggregate names the entity key it was computed over.
    """
    # The policy the contract was built under, as it was serialised into it -- so the
    # lineage describes the contract on disk rather than whatever config happens to be
    # loaded now.
    derived_inputs = {
        d["name"]: list(d["inputs"])
        for d in (contract.admission_rules or {}).get("derivations", [])
    }

    deps = {}
    for column in contract.columns:
        if not column.admitted:
            continue
        sources = derived_inputs.get(column.name, [column.name])
        deps[column.name] = [
            TableColumnDep(asset_key=upstream_asset_key, column_name=source)
            for source in sources
        ]
    return TableColumnLineage(deps_by_column=deps)


def contract_metadata(contract, upstream_asset_key) -> dict:
    """Both of the above plus the headline counts, ready for `MaterializeResult`."""
    return {
        "dagster/column_schema": contract_table_schema(contract),
        "dagster/column_lineage": contract_column_lineage(contract, upstream_asset_key),
        "fingerprint": MetadataValue.text(contract.fingerprint()),
        "admitted": MetadataValue.int(len(contract.training_features())),
        "rejected": MetadataValue.int(len(contract.rejections())),
    }
