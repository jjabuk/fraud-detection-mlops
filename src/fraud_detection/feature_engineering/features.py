"""The velocity features: one SQL statement.

The aggregates that make this a fraud model rather than a row classifier live here, not in
a Dagster asset, so the core SQL structure is centralized.

Every window is `RANGE BETWEEN … AND 1 PRECEDING`. `RANGE` frames on the `ORDER BY`
**value**, so `1 PRECEDING` means "`TransactionDT` strictly less than this row's" — it
excludes the current row *and* every peer sharing its timestamp. `LAG`, `LEAD` and `ROWS`
frames are banned, and the tests assert they are absent. See docs/point-in-time.md.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from fraud_detection.core.schema import (
    CARD_ENTITY_COLUMN,
    CLIENT_ENTITY_ANCHOR,
    CLIENT_ENTITY_COLUMN,
    CLIENT_ENTITY_COMPONENTS,
    DEVICE_ENTITY_COLUMN,
)

__all__ = [
    "CLIENT_UID_EXPRESSION",
    "C_COLUMNS",
    "FEATURE_ENGINEERING_SQL",
    "M_COLUMNS",
    "WINDOW_1H_SECONDS",
    "WINDOW_24H_SECONDS",
    "build_sql",
    "build_uid_aggregate_sql",

    "uid_aggregate_columns",
]



WINDOW_1H_SECONDS = 3_600
WINDOW_24H_SECONDS = 86_400

# card1 + addr1 + the day the card began. D1 is days since that day, so
# floor(TransactionDT / 86400) - D1 recovers it -- a constant across the client's history.
# NULL when a component is missing: filling with a sentinel merges unrelated clients into
# one group and measures *worse* than card1 alone (79.2% label purity against 84.8%).
CLIENT_UID_EXPRESSION = """IF(
      {addr_column} IS NULL OR {anchor_column} IS NULL,
      NULL,
      FORMAT(
        '%d_%d_%d',
        {card_entity_column},
        CAST({addr_column} AS INT64),
        DIV(CAST(TransactionDT AS INT64), 86400) - CAST({anchor_column} AS INT64)
      )
    )"""


# ---- the winners' uid aggregates, under a point-in-time window ------------------
#
# After Deotte, "XGB Fraud with Magic", and Yakovlev, "IEEE - uid detection" -- see
# ATTRIBUTION.md, which records that the published versions compute these over train u test
# and this one does not.
#
# Their gain came from aggregating over the reconstructed client: mean-encoded
# `C` and `M` columns, and `std` of the normalised `D` columns. They computed those over the
# whole group, which sees a client's future -- so the interesting question, and the one this
# project exists to answer, is what the same aggregates are worth when the window stops at
# the previous transaction.
#
# Every window below is `client_prior`: RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING,
# framing on the ORDER BY *value*, so the current row and every peer sharing its timestamp
# are excluded.
#
# `std(D1n)` deserves a note. D1n is the day the card began, so within a correctly
# reconstructed client it is *constant* and its std is 0. A non-zero value means the uid has
# merged two clients -- which is the source notebook's own consistency check, arriving here
# as a feature the model can use rather than a filter applied by hand.
C_COLUMNS = [f"C{i}" for i in range(1, 15)]

# M4 is excluded on purpose: it is a three-level string (M0/M1/M2), and a mean over labels
# is not a number. The other eight are booleans, so their mean is the client's historical
# rate of that flag.
M_COLUMNS = [f"M{i}" for i in (1, 2, 3, 5, 6, 7, 8, 9)]


def uid_aggregate_columns(
    derivations: Sequence = (),
    c_columns: Sequence[str] = (),
    m_columns: Sequence[str] = (),
) -> list[str]:
    """The names these aggregates produce, in the order the SQL emits them."""
    return (
        [f"client_{d.name.lower()}_std_prior" for d in derivations]
        + [f"client_{c.lower()}_mean_prior" for c in c_columns]
        + [f"client_{m.lower()}_mean_prior" for m in m_columns]
    )


def build_uid_aggregate_projection(
    derivations: Sequence = (),
    c_columns: Sequence[str] = (),
    m_columns: Sequence[str] = (),
) -> str:
    """The same names again, for the outer SELECT that reads them back out."""
    names = uid_aggregate_columns(derivations, c_columns, m_columns)
    return "".join(f",\n  {name}" for name in names)


def build_uid_aggregate_sql(
    derivations: Sequence = (),
    c_columns: Sequence[str] = (),
    m_columns: Sequence[str] = (),
) -> str:
    """The SELECT-list fragment computing them. Empty string when nothing is declared."""
    from fraud_detection.feature_engineering.derivations import DERIVATION_SQL

    lines = []
    for derivation in derivations:
        renderer = DERIVATION_SQL.get(derivation.tool)
        if renderer is None:
            continue
        expression = renderer(derivation.inputs, derivation.params)
        lines.append(
            f"    IF({{client_entity_column}} IS NULL, NULL, "
            f"STDDEV({expression}) OVER client_prior) "
            f"AS client_{derivation.name.lower()}_std_prior"
        )
    for column in c_columns:
        lines.append(
            f"    IF({{client_entity_column}} IS NULL, NULL, AVG({column}) OVER client_prior) "
            f"AS client_{column.lower()}_mean_prior"
        )
    for column in m_columns:
        lines.append(
            f"    IF({{client_entity_column}} IS NULL, NULL, "
            f"AVG(CAST({column} AS INT64)) OVER client_prior) "
            f"AS client_{column.lower()}_mean_prior"
        )
    # Leading comma, no trailing one: the fragment splices onto the end of an existing
    # SELECT list, and an empty declaration has to leave that list untouched.
    return ("".join(f",\n{line}" for line in lines)) if lines else ""


FEATURE_ENGINEERING_SQL = """
WITH with_client AS (
  SELECT
    *,
    {client_uid_expression} AS {client_entity_column}
  FROM {source_table}
),
entity_windows AS (
  SELECT
    TransactionID,
    TransactionDT,
    TransactionAmt,
    {card_entity_column} AS card_entity,
    {device_entity_column} AS device_entity,
    {client_entity_column} AS client_entity,
    isFraud,
    IF({client_entity_column} IS NULL, NULL, COUNT(*) OVER client_prior)
      AS client_txn_count_prior,
    IF({client_entity_column} IS NULL, NULL, COUNT(*) OVER client_24h)
      AS client_txn_count_24h,
    IF({client_entity_column} IS NULL, NULL, AVG(TransactionAmt) OVER client_prior)
      AS client_amt_avg_prior,
    IF({client_entity_column} IS NULL, NULL, MAX(TransactionDT) OVER client_prior)
      AS prev_txn_dt_client,
    COUNT(*) OVER card_1h AS card_txn_count_1h,
    COUNT(*) OVER card_24h AS card_txn_count_24h,
    AVG(TransactionAmt) OVER card_24h AS card_txn_amt_avg_24h,
    SUM(TransactionAmt) OVER card_24h AS card_txn_amt_sum_24h,
    MAX(TransactionDT) OVER card_prior AS prev_txn_dt_card,
    IF(
      {device_entity_column} IS NULL,
      NULL,
      COUNT(*) OVER device_24h
    ) AS device_txn_count_24h{uid_aggregate_expressions}
  FROM with_client
  WINDOW
    client_prior AS (
      PARTITION BY {client_entity_column} ORDER BY TransactionDT
      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ),
    client_24h AS (
      PARTITION BY {client_entity_column} ORDER BY TransactionDT
      RANGE BETWEEN {window_24h} PRECEDING AND 1 PRECEDING
    ),
    card_1h AS (
      PARTITION BY {card_entity_column} ORDER BY TransactionDT
      RANGE BETWEEN {window_1h} PRECEDING AND 1 PRECEDING
    ),
    card_24h AS (
      PARTITION BY {card_entity_column} ORDER BY TransactionDT
      RANGE BETWEEN {window_24h} PRECEDING AND 1 PRECEDING
    ),
    card_prior AS (
      PARTITION BY {card_entity_column} ORDER BY TransactionDT
      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ),
    device_24h AS (
      PARTITION BY {device_entity_column} ORDER BY TransactionDT
      RANGE BETWEEN {window_24h} PRECEDING AND 1 PRECEDING
    )
)
SELECT
  TransactionID,
  TransactionDT,
  TransactionAmt,
  card_entity AS {card_entity_column},
  device_entity AS {device_entity_column},
  client_entity AS {client_entity_column},
  isFraud,
  card_txn_count_1h,
  card_txn_count_24h,
  card_txn_amt_avg_24h,
  card_txn_amt_sum_24h,
  TransactionAmt - card_txn_amt_avg_24h AS card_amt_deviation_24h,
  TransactionDT - prev_txn_dt_card AS seconds_since_prev_txn_card,
  device_txn_count_24h,
  client_txn_count_prior,
  client_txn_count_24h,
  client_amt_avg_prior,
  TransactionAmt - client_amt_avg_prior AS client_amt_deviation_prior,
  TransactionDT - prev_txn_dt_client AS seconds_since_prev_txn_client{uid_aggregate_projection}
FROM entity_windows
"""

# The whole dialect difference, listed so it can be read and tested rather than trusted.
# Everything that matters -- the window frames, the null guards, the arithmetic -- is
# identical between engines.


def build_sql(
    *,
    source_table: str,
    destination_table: str | None = None,
    card_entity_column: str = CARD_ENTITY_COLUMN,
    device_entity_column: str = DEVICE_ENTITY_COLUMN,
    client_entity_column: str = CLIENT_ENTITY_COLUMN,
    window_1h: int = WINDOW_1H_SECONDS,
    window_24h: int = WINDOW_24H_SECONDS,
    derivations: Sequence = (),
    c_columns: Sequence[str] = (),
    m_columns: Sequence[str] = (),
) -> str:
    """The statement, in BigQuery dialect.

    ``destination_table`` wraps it in ``CREATE OR REPLACE TABLE``; without it the statement
    is a plain ``SELECT``, which is what a local run wants.

    ``derivations`` are the policy's declared derived columns. Only their `std` over the
    client's earlier transactions is emitted here -- the row-local values themselves were
    audited and rejected (PSI up to 3.09: `Dxn` is a calendar day number, so an early
    window and a late one barely overlap). The std is not a date and does not carry that
    problem.
    """
    aggregate_expressions = build_uid_aggregate_sql(derivations, c_columns, m_columns).format(
        client_entity_column=client_entity_column
    )
    body = FEATURE_ENGINEERING_SQL.format(
        source_table=source_table,
        card_entity_column=card_entity_column,
        device_entity_column=device_entity_column,
        client_entity_column=client_entity_column,
        client_uid_expression=CLIENT_UID_EXPRESSION.format(
            card_entity_column=card_entity_column,
            addr_column=CLIENT_ENTITY_COMPONENTS[1],
            anchor_column=CLIENT_ENTITY_ANCHOR,
        ),
        window_1h=window_1h,
        window_24h=window_24h,
        uid_aggregate_expressions=aggregate_expressions,
        uid_aggregate_projection=build_uid_aggregate_projection(
            derivations, c_columns, m_columns
        ),
    )
    if destination_table:
        return f"CREATE OR REPLACE TABLE `{destination_table}` AS{body}"
    return body





def uses_banned_window_functions(sql: str) -> list[str]:
    """Positional window functions, which would break the point-in-time guarantee.

    `LAG` returns the previous *row*, not the previous *instant*, so on two transactions
    sharing a timestamp it hands back a peer. Measured against the real dataset before the
    rule existed: 166 rows with a gap of zero.
    """
    return [
        name
        for name, pattern in (
            ("LAG", r"\bLAG\s*\("),
            ("LEAD", r"\bLEAD\s*\("),
            ("ROWS frame", r"\bROWS\s+BETWEEN\b"),
        )
        if re.search(pattern, sql, re.IGNORECASE)
    ]
