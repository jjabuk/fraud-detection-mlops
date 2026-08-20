"""Columns the contract defines rather than merely admits.

A derived column does not exist in `features.model_input`; the contract says how to compute
it, and **one implementation computes it everywhere** — for the audits, for training, and
for serving. A derivation applied in training and reimplemented in the serving container is
two definitions of one feature, and two definitions drift.

**Row-local only.** Every derivation here computes from one row and nothing else, which
makes it trivially point-in-time and trivially servable. Two things that look similar and
do not belong here:

* an **entity aggregate** (a mean over a client's past) needs a window over earlier rows —
  it lives in the feature-engineering SQL under `RANGE … 1 PRECEDING`;
* a **fitted transform** needs its parameters committed as an artefact, because a mapping
  refitted at serving time is a different mapping. `frequency_encode` below is one, and its
  counts live in `references/frequency-maps.json`.

## The D-normalisation

`Dxn = floor(TransactionDT / 86400) - Dx`. Each `D` column counts days since something began
for this card; subtracting it from the transaction's own day recovers the day it began. That
value is constant across a client's history, which is what makes it usable both as an
identity component and as a feature whose distribution does not drift with the calendar.

After Konstantin Yakovlev, "IEEE - uid detection" (Kaggle). What was taken and what is
this repository's: see ATTRIBUTION.md.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import polars as pl

__all__ = [
    "DERIVATIONS",
    "DERIVATION_SQL",
    "FREQUENCY_MAP_FILE",
    "DerivationError",
    "apply_derivations",
    "days_since_to_start_day",
    "days_since_to_start_day_sql",
    "frequency_encode",
    "load_frequency_maps",
    "one_hot",
]

SECONDS_PER_DAY = 86_400


class DerivationError(Exception):
    """Raised when a declared derivation cannot be computed from the frame it was given."""


def days_since_to_start_day(
    frame: pl.DataFrame,
    inputs: Sequence[str],
    params: Mapping | None = None,
) -> pl.Series:
    """`floor(time / 86400) - days_since` — the day the counter started.

    NaN in, NaN out. Filling the gap would invent a start day, and every row missing the
    counter would share one fictitious value — the same failure mode as sentinel-filling an
    entity key, which measured *worse* than not reconstructing at all.
    """
    params = params or {}
    (days_since,) = inputs
    time_column = params.get("time_column", "TransactionDT")

    day = (frame.get_column(time_column) / params.get("unit_seconds", SECONDS_PER_DAY)).floor()
    return (day - frame.get_column(days_since)).cast(pl.Float64)


def days_since_to_start_day_sql(inputs: Sequence[str], params: Mapping | None = None) -> str:
    """The same arithmetic as :func:`days_since_to_start_day`, as a SQL expression.

    Two renderings of one definition, kept adjacent so they cannot drift apart unnoticed,
    and asserted numerically equal in `tests/test_uid_aggregates.py`. The SQL one exists
    because `std(Dxn)` grouped by client is an **entity aggregate**: it needs a window over
    the client's earlier rows, which is a thing SQL does and a per-row Python function
    cannot.
    """
    params = params or {}
    (days_since,) = inputs
    time_column = params.get("time_column", "TransactionDT")
    unit = params.get("unit_seconds", SECONDS_PER_DAY)
    return f"(FLOOR({time_column} / {unit}) - {days_since})"


def one_hot(
    frame: pl.DataFrame,
    inputs: Sequence[str],
    params: Mapping | None = None,
) -> pl.Series:
    """1 when the column equals `level`, 0 for any other value, null when the source is null.

    **One indicator per declaration, with the level named in config.** A tool that expanded
    a column into whatever levels it found would infer a different set in training than at
    serving time, and the mismatch would be invisible — the serving frame would simply carry
    different columns. Naming levels explicitly also means a new category shows up as an
    indicator that is 0 everywhere rather than as a reshaped matrix.

    **Null stays null.** Mapping it to 0 would assert "not M0" about a row where nobody
    recorded M4, and M4 is null on 47.7% of rows. LightGBM routes nulls down whichever
    branch fits them, so keeping them distinct from "not this level" costs nothing.
    """
    params = params or {}
    (column,) = inputs
    level = params["level"]
    values = frame.get_column(column).cast(pl.String)
    # A Series, like every other tool in the registry.
    return (
        pl.select(
            pl.when(values.is_null()).then(None).otherwise((values == level).cast(pl.Int8))
        )
        .to_series()
        .alias(column)
    )


#: Counts fitted on the training split, committed so training and serving share one map.
FREQUENCY_MAP_FILE = "references/frequency-maps.json"


def load_frequency_maps(path: str | None = None) -> dict[str, dict[str, int]]:
    """Reads the fitted count tables. Cached by the caller if it matters.

    **The contract is the source.** The maps are fitted by the audit repository, on its
    training split, and travel inside `references/feature-contract.json` under the same
    fingerprint as the verdicts — so a model pinned to a contract is pinned to the mapping
    it was trained under, and a map that moves invalidates that pin instead of quietly
    changing what the model sees.

    `references/frequency-maps.json` is where they lived before the audits became their
    own repository. It is still read when the contract carries none, which keeps a
    contract stamped under the previous scheme loadable; an explicit ``path`` always wins,
    because a caller naming a file means it.
    """
    import json

    from fraud_detection.config import resolve_repo_path

    if path is None:
        from fraud_detection.contract import CONTRACT_FILE, FeatureContract

        contract_path = resolve_repo_path(CONTRACT_FILE)
        if contract_path.exists():
            fitted = FeatureContract.from_json(contract_path.read_text()).fitted_parameters
            maps = fitted.get("frequency_maps", {}).get("maps")
            if maps:
                return maps

    return json.loads(resolve_repo_path(path or FREQUENCY_MAP_FILE).read_text())["maps"]


def frequency_encode(
    frame: pl.DataFrame,
    inputs: Sequence[str],
    params: Mapping | None = None,
) -> pl.Series:
    """How often this value occurs, from counts fitted on the training split.

    The technique is from Yakovlev, "IEEE - Basic FE Part 1"; the training-split-only fit,
    the evidence-based column choice and the null-for-unseen rule are not. See
    ATTRIBUTION.md.

    A fitted transform, so its parameters are committed to
    `references/frequency-maps.json`. Counting the frame in hand would give training and
    scoring two different mappings; counting train and test together would be transductive,
    which the point-in-time rule forbids.

    **An unseen value maps to null, not 0.** Zero would claim the value occurs zero times,
    which is false — it occurs in the row being scored. Null says what is true: the training
    window has nothing to say about it.
    """
    params = params or {}
    (column,) = inputs
    # `is None`, not a truth test: an explicitly empty map means "there are no maps",
    # not "none supplied".
    maps = params.get("_maps")
    if maps is None:
        maps = load_frequency_maps(params.get("map_file"))
    table = maps.get(params.get("map_key", column))
    if table is None:
        raise DerivationError(
            f"no frequency map for {column!r} in {params.get('map_file') or FREQUENCY_MAP_FILE} "
            f"(have: {sorted(maps)[:8]}...) -- regenerate it with `uv run build-frequency-maps`"
        )
    values = frame.get_column(column).cast(pl.String)
    return values.replace_strict(table, default=None, return_dtype=pl.Int64).alias(column)


DERIVATIONS: dict[str, Callable[..., pl.Series]] = {
    "days_since_to_start_day": days_since_to_start_day,
    "one_hot": one_hot,
    "frequency_encode": frequency_encode,
}

# Only derivations a *window aggregate* needs get a SQL rendering — `std(Dxn)` is computed
# over a client's earlier rows, which a per-row Python function cannot do.
DERIVATION_SQL: dict[str, Callable[..., str]] = {
    "days_since_to_start_day": days_since_to_start_day_sql,
}


def apply_derivations(frame: pl.DataFrame, declared: Sequence) -> pl.DataFrame:
    """Add every declared derived column to a copy of ``frame``.

    ``declared`` is a sequence of `policy.Derivation`. Missing inputs raise rather than
    being skipped: a derivation that silently does not happen produces a model trained on
    fewer columns than the contract says it was, and the metrics would describe something
    that was never fitted.
    """
    if not len(declared):
        return frame

    out = frame
    for item in declared:
        tool = DERIVATIONS.get(item.tool)
        if tool is None:
            raise DerivationError(
                f"{item.name} declares tool '{item.tool}', which is not in DERIVATIONS "
                f"({sorted(DERIVATIONS)}) — a contract cannot define a column nothing can compute"
            )
        if missing := [c for c in item.inputs if c not in out.columns]:
            raise DerivationError(
                f"{item.name} needs {missing}, absent from the frame — the derivation and "
                "the table have drifted apart"
            )
        out = out.with_columns(tool(out, item.inputs, item.params).alias(item.name))

    return out
