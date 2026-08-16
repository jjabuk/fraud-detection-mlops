"""Loading a pinned-schema CSV into BigQuery, and checking it stayed pinned.

Pure: builds SQL and compares schemas, with no client and no orchestrator. The assets in
`orchestration/assets/ingestion.py` do the loading and call these.
"""

from __future__ import annotations

from google.cloud import bigquery

__all__ = [
    "STAGING_SUFFIX",
    "build_cast_sql",
    "build_integrality_sql",
    "integer_columns",
    "relaxed_schema",
    "schema_matches_pinned",
]

# Vesta writes whole numbers with a decimal point -- `C1` is `1.0` in the file -- and
# BigQuery's CSV loader will not parse that into an INT64 column. Retyping those columns to
# FLOAT would make the load work and the schema wrong: they are counts.
#
# So the load lands in a staging table with the INTEGER columns widened to FLOAT64, and one
# query casts them back. `_assert_integral` verifies every value is whole before the cast,
# which also catches the opposite error: a column typed INTEGER that holds fractions would
# otherwise be silently rounded.
STAGING_SUFFIX = "_staging"


def relaxed_schema(pinned: list[bigquery.SchemaField]) -> list[bigquery.SchemaField]:
    """The pinned schema with INTEGER columns widened to FLOAT, for the load step only."""
    return [
        bigquery.SchemaField(
            field.name,
            "FLOAT64" if field.field_type in ("INTEGER", "INT64") else field.field_type,
            mode=field.mode,
        )
        for field in pinned
    ]


def integer_columns(pinned: list[bigquery.SchemaField]) -> list[str]:
    return [f.name for f in pinned if f.field_type in ("INTEGER", "INT64")]


def build_integrality_sql(staging_table: str, columns: list[str]) -> str:
    """Per INTEGER-typed column, how many values would lose precision on the cast.

    One scan answers for every column: a query per column would re-read the same bytes
    twenty-eight times.
    """
    counts = ",\n  ".join(
        f"COUNTIF(`{c}` IS NOT NULL AND `{c}` != TRUNC(`{c}`)) AS `{c}`" for c in columns
    )
    return f"SELECT\n  {counts}\nFROM `{staging_table}`"


def build_cast_sql(staging_table: str, pinned: list[bigquery.SchemaField]) -> str:
    """Projects the staging table back onto the pinned types, column by named column."""
    projection = ",\n  ".join(
        f"CAST(`{field.name}` AS INT64) AS `{field.name}`"
        if field.field_type in ("INTEGER", "INT64")
        else f"`{field.name}`"
        for field in pinned
    )
    return f"SELECT\n  {projection}\nFROM `{staging_table}`"



def schema_matches_pinned(table, pinned) -> bool:
    """Does the live table carry the types `schemas/*.json` pins?

    The pinned schema is an input to this load as much as the CSV is: retyping a column
    changes what lands in the table while leaving the row count and the source etag
    untouched. Comparing name -> type is enough; mode and description do not affect the cast.
    """
    live = {field.name: field.field_type for field in table.schema}
    equivalent = {"INT64": "INTEGER", "FLOAT64": "FLOAT", "BOOL": "BOOLEAN"}

    def canonical(field_type: str) -> str:
        return equivalent.get(field_type, field_type)

    for field in pinned:
        if field.name not in live:
            return False
        if canonical(live[field.name]) != canonical(field.field_type):
            return False
    return True
