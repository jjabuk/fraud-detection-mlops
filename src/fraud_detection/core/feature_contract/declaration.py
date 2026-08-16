"""Where each column comes from when a real request arrives.

No audit can work this out. A check knows whether a column is *predictive*; only the
pipeline knows whether the caller can send it. Getting that distinction wrong is how a
request schema ends up demanding features nobody outside the system can compute, so the
declaration is derived here from the two tables that actually assemble the model input:

* ``features.transaction_features`` holds entity state — the card's and the device's
  history *before* this transaction. Retrieved at serving time.
* ``raw.ieee_train_joined`` holds properties of the transaction itself. Present in the
  request, because a transaction that has never been seen cannot be looked up anywhere.

See [docs/feature-engineering.md](../../../docs/feature-engineering.md) §2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fraud_detection.core.feature_contract.core import Source
from fraud_detection.core.schema import EXCLUDED_COLUMNS, FEATURE_COLUMNS

__all__ = ["RETRIEVED_COLUMNS", "declare_columns", "python_dtype", "retrieved_columns"]

RETRIEVED_COLUMNS = frozenset(FEATURE_COLUMNS)
"""The engineered velocity aggregates. Everything else in the model input is a property
of the transaction and therefore arrives in the request."""


def retrieved_columns() -> frozenset[str]:
    """`RETRIEVED_COLUMNS` plus the uid aggregates the declared derivations produce.

    Computed rather than pinned: the aggregates' names follow from
    `config/feature-admission.toml`, and a hand-kept copy would drift from it silently. Read
    lazily so importing this module does not depend on the policy file existing — the
    checks that need the file say so themselves.

    Getting this wrong has a specific, bad shape: an aggregate declared `request` would put
    `client_c1_mean_prior` in the serving schema, asking the caller to send a mean over
    their own history.
    """
    from fraud_detection.core.schema import uid_aggregate_feature_columns

    return RETRIEVED_COLUMNS | frozenset(uid_aggregate_feature_columns())

# Two vocabularies land here and both have to map correctly. The audit assets hold a
# DataFrame and know polars dtypes; anything reading `INFORMATION_SCHEMA` knows BigQuery
# type names. One table covers both, because the alternative -- defaulting the unknown to
# `float` -- is not a harmless fallback: polars calls a string column `String`, which is
# absent from a BigQuery-only table, so every categorical (`ProductCD`, `DeviceInfo`,
# `P_emaildomain`) would be declared `float` and `request_model()` would emit a request
# schema demanding a number where the caller sends text.
_TYPE_TO_PY = {
    # BigQuery
    "INTEGER": "int",
    "INT64": "int",
    "FLOAT": "float",
    "FLOAT64": "float",
    "NUMERIC": "float",
    "BIGNUMERIC": "float",
    "BOOLEAN": "bool",
    "BOOL": "bool",
    "STRING": "str",
    # polars
    "OBJECT": "str",
    "CATEGORY": "str",
    "CATEGORICAL": "str",
    # `STRING` is listed once, in the BigQuery block above -- polars uses the same spelling.
    "UTF8": "str",
    "STRING[PYTHON]": "str",
    "STRING[PYARROW]": "str",
    "INT8": "int",
    "INT16": "int",
    "INT32": "int",
    "UINT8": "int",
    "UINT16": "int",
    "UINT32": "int",
    "UINT64": "int",
    "FLOAT32": "float",
}


def python_dtype(type_name: str) -> str:
    """Map a BigQuery type name or a polars dtype onto the contract's dtype vocabulary.

    Nullable polars dtypes upper-case onto their non-null
    counterparts, which is why the comparison is case-insensitive rather than exact.
    """
    name = str(type_name).upper()
    if name == "BOOLEAN":  # polars `Boolean` and BigQuery BOOLEAN agree here
        return "bool"
    return _TYPE_TO_PY.get(name, "float")


def declare_columns(
    schema: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    retrieved: frozenset[str] | None = None,
    excluded: frozenset[str] = EXCLUDED_COLUMNS,
    derived: frozenset[str] = frozenset(),
) -> dict[str, tuple[str, str]]:
    """Build the ``declared`` mapping a contract is assembled against.

    ``schema`` maps column name to its type — either a BigQuery type name (from
    ``INFORMATION_SCHEMA``) or a polars dtype (from a loaded frame). Both are accepted
    because both callers exist; see :func:`python_dtype`.

    Columns in ``excluded`` are dropped rather than declared and rejected. The distinction
    matters: a rejected column is one a check ruled out and could be reinstated by a later
    audit; ``TransactionID`` and the label are not features at all and never will be, so
    recording them as rejections would put permanent noise in every contract diff.
    """
    items = schema.items() if isinstance(schema, Mapping) else schema
    retrieved = retrieved_columns() if retrieved is None else retrieved

    return {
        name: (_source_of(name, retrieved, derived), python_dtype(type_name))
        for name, type_name in items
        if name not in excluded
    }


def _source_of(name: str, retrieved: frozenset[str], derived: frozenset[str]) -> str:
    """Derived beats retrieved beats request.

    A derived column is computed inside the service from the other two, so it is never in
    the request schema -- asking a caller to send `D1n` would be asking them to run our
    arithmetic. It is not retrieved either: there is nothing to look up.
    """
    if name in derived:
        return Source.DERIVED
    return Source.RETRIEVED if name in retrieved else Source.REQUEST
