"""One list of admitted columns, assembled from every check that ran.

Each audit — time consistency, distribution shift — produces a :class:`Fragment` saying
which columns it rejects and by what number. :class:`FeatureContract` merges the fragments
with a declaration of where each column comes from at serving time, and becomes the single
source of truth for three consumers that must never disagree:

* the feature list training reads,
* the request schema the serving API validates against,
* the columns the drift monitor iterates.

The contract is data, not code: serialize it, commit it, and let a change arrive as a
reviewable diff rather than as a silent re-derivation on the next run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

__all__ = [
    "Column",
    "ContractError",
    "FeatureContract",
    "Fragment",
    "Rejection",
    "Source",
    "assert_model_features_admitted",
    "from_admission_rules",
    "from_distribution_shift",
    "from_time_consistency",
]


class ContractError(Exception):
    """Raised when the contract and something downstream have gone out of step."""


class Source:
    """Where a column's value comes from when a real request arrives."""

    REQUEST = "request"
    """Carried by the incoming transaction. Belongs in the API request schema."""

    RETRIEVED = "retrieved"
    """Entity state read from the feature store. Never accepted from the caller —
    requiring that would move the hard half of the problem outside the system."""

    DERIVED = "derived"
    """Computed inside the service from the other two."""


@dataclass(frozen=True)
class Rejection:
    """Why one column is out, and the number that put it there."""

    column: str
    by: str
    value: float | None = None
    unit: str = "column"
    """``column``, or ``block:<name>`` when the decision was taken over a group. A column
    can pass on its own and still be rejected as part of a block it belongs to; recording
    which happened is what stops the decision being reversed later by someone reading the
    per-column number in isolation."""


@dataclass(frozen=True)
class Fragment:
    """One audit's contribution: what it rejected, under what settings, and how much to
    trust it.

    ``qualification`` is not decoration. An audit whose verdicts do not reproduce under a
    different window is noise, and the contract should carry the evidence either way.
    """

    check: str
    rejections: tuple[Rejection, ...] = ()
    params: dict = field(default_factory=dict)
    qualification: dict = field(default_factory=dict)
    tool: str = ""


@dataclass(frozen=True)
class Column:
    name: str
    source: str
    dtype: str = "float"
    admitted: bool = True
    rejected_by: str = ""
    rejected_value: float | None = None
    rejected_unit: str = ""

    admitted_by: str = ""
    """Non-empty only when policy overruled a check — ``"override"``. `rejected_by` keeps
    naming the check that objected, so the contract shows both what the audit found and
    that somebody decided otherwise. An override that looked like an ordinary admission
    would be an override nobody could audit."""

    override_reason: str = ""
    override_expires: str = ""


@dataclass
class FeatureContract:
    columns: tuple[Column, ...]
    data: dict = field(default_factory=dict)
    fragments: tuple[Fragment, ...] = ()
    entity: dict = field(default_factory=dict)
    admission_rules: dict = field(default_factory=dict)
    created_at: str = ""
    version: str = "1"

    # ---- assembly -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        declared: dict[str, tuple[str, str]],
        fragments: Iterable[Fragment],
        *,
        data: dict | None = None,
        entity: dict | None = None,
        admission_rules: dict | None = None,
        overrides: Iterable = (),
        version: str = "1",
    ) -> FeatureContract:
        """Merge ``fragments`` onto ``declared``.

        ``declared`` maps column name to ``(source, dtype)``. It cannot be inferred: no
        check knows whether a column arrives in the request or is looked up, and getting
        that wrong is precisely how a request schema ends up asking callers for features
        they cannot possibly have.

        A column is admitted unless some fragment rejects it. The **first** rejection wins
        and is recorded with the check that made it, so "why is this column missing" has
        one answer rather than a list.
        """
        fragments = tuple(fragments)
        unknown = {r.column for f in fragments for r in f.rejections} - set(declared)
        if unknown:
            raise ContractError(
                f"{len(unknown)} column(s) rejected by a check but absent from the "
                f"declaration, e.g. {sorted(unknown)[:5]} — the audit and the table have "
                "drifted apart"
            )

        first: dict[str, tuple[str, Rejection]] = {}
        for frag in fragments:
            for r in frag.rejections:
                first.setdefault(r.column, (frag.check, r))

        by_column = {o.column: o for o in overrides}
        if stray := sorted(set(by_column) - set(declared)):
            raise ContractError(
                f"override(s) name column(s) that are not declared: {stray[:5]} — an "
                "override for a column the table does not have is a typo, not a decision"
            )

        columns = []
        for name, (source, dtype) in declared.items():
            hit = first.get(name)
            override = by_column.get(name) if hit else None
            columns.append(
                Column(
                    name=name,
                    source=source,
                    dtype=dtype,
                    # An overridden column is admitted, and still records the check that
                    # objected. Both halves of that are the point.
                    admitted=hit is None or override is not None,
                    rejected_by=hit[0] if hit else "",
                    rejected_value=hit[1].value if hit else None,
                    rejected_unit=hit[1].unit if hit else "",
                    admitted_by="override" if override else "",
                    override_reason=override.reason if override else "",
                    override_expires=override.expires.isoformat() if override else "",
                )
            )

        return cls(
            columns=tuple(columns),
            data=dict(data or {}),
            fragments=fragments,
            entity=dict(entity or {}),
            admission_rules=dict(admission_rules or {}),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            version=version,
        )

    # ---- the three consumers --------------------------------------------------

    def training_features(self) -> list[str]:
        """Every admitted column, in declaration order."""
        return [c.name for c in self.columns if c.admitted]

    def request_fields(self) -> list[Column]:
        """Admitted columns the caller is expected to send."""
        return [c for c in self.columns if c.admitted and c.source == Source.REQUEST]

    def retrieved_fields(self) -> list[Column]:
        """Admitted columns the service looks up rather than accepts."""
        return [c for c in self.columns if c.admitted and c.source == Source.RETRIEVED]

    def monitored_columns(self) -> list[str]:
        """What the drift monitor iterates — the admitted set, and nothing else.

        A column that was never admitted is never monitored, and a column that was
        admitted cannot be forgotten. That equality is the whole point of routing all
        three consumers through one object.
        """
        return self.training_features()

    def request_model(self, name: str = "PredictRequest"):
        """Build the Pydantic request model from the admitted request-side columns.

        Every field is optional: this dataset is mostly missing values, and a schema that
        demanded them would reject traffic the model handles perfectly well.
        """
        from pydantic import create_model

        py = {"float": float, "int": int, "str": str, "bool": bool}
        fields = {c.name: (py.get(c.dtype, float) | None, None) for c in self.request_fields()}
        return create_model(name, **fields)

    # ---- provenance -----------------------------------------------------------

    def rejections(self) -> list[Column]:
        return [c for c in self.columns if not c.admitted]

    def overrides(self) -> list[Column]:
        """Columns admitted against a check's objection. Should be short and explicable."""
        return [c for c in self.columns if c.admitted_by == "override"]

    def fingerprint(self) -> str:
        """Hash of the admitted set, its sources, and the policy that produced it.

        Pin this next to a trained model. It changes when the contract changes in a way
        that affects the model, and not when the contract is merely regenerated.

        The rules are in the payload because an audit threshold can move without the admitted set
        moving — and then two models pinned to "the same" contract would have been evaluated
        under different admission rules. Note what this deliberately is *not*: a git SHA. That would
        change on a README typo and stay put when the contract is regenerated on new data,
        which is the failure both directions. It also could not be recomputed from the file,
        so `from_dict` would lose its ability to detect a hand-edited contract.
        """
        payload = {
            "columns": [(c.name, c.source, c.dtype) for c in self.columns if c.admitted],
            "policy": self.admission_rules,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint(),
            "data": self.data,
            "entity": self.entity,
            "policy": self.admission_rules,
            "summary": {
                "declared": len(self.columns),
                "admitted": len(self.training_features()),
                "rejected": len(self.rejections()),
                "overridden": len(self.overrides()),
                "request_fields": len(self.request_fields()),
                "retrieved_fields": len(self.retrieved_fields()),
            },
            "fragments": [
                {
                    "check": f.check,
                    "tool": f.tool,
                    "params": f.params,
                    "qualification": f.qualification,
                    "rejected": len(f.rejections),
                }
                for f in self.fragments
            ],
            "columns": [asdict(c) for c in self.columns],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureContract:
        contract = cls(
            columns=tuple(Column(**c) for c in d["columns"]),
            data=d.get("data", {}),
            entity=d.get("entity", {}),
            admission_rules=d.get("policy", d.get("admission_rules", {})),
            created_at=d.get("created_at", ""),
            version=d.get("version", "1"),
        )
        stored = d.get("fingerprint")
        if stored and stored != contract.fingerprint():
            raise ContractError(
                f"fingerprint mismatch: file says {stored}, contents hash to "
                f"{contract.fingerprint()} — the file was edited by hand"
            )
        return contract

    @classmethod
    def from_json(cls, text: str) -> FeatureContract:
        return cls.from_dict(json.loads(text))


# ---- adapters: audit output -> fragment ---------------------------------------


def from_time_consistency(
    report,
    *,
    blocks: dict[str, Sequence[str]] | None = None,
    params: dict | None = None,
    qualification: dict | None = None,
) -> Fragment:
    """Turn a ``time_consistency.scan`` report into a fragment.

    ``blocks`` maps a block name to its columns. When given, a block is rejected whole as
    soon as any member inverts — the reading the data forces: members of one missingness
    family degrade together, and a per-column threshold splits the family arbitrarily.
    """
    import polars as pl
    inverted = report.filter(pl.col("verdict") == "inverted")
    by_column = dict(zip(inverted.get_column("feature"), inverted.get_column("delta"), strict=True))

    rejections: list[Rejection] = []
    claimed: set[str] = set()

    for block_name, members in (blocks or {}).items():
        hits = [m for m in members if m in by_column]
        if not hits:
            continue
        worst = min(by_column[m] for m in hits)
        rejections += [
            Rejection(m, "time_consistency", float(worst), f"block:{block_name}") for m in members
        ]
        claimed.update(members)

    rejections += [
        Rejection(col, "time_consistency", float(delta), "column")
        for col, delta in by_column.items()
        if col not in claimed
    ]

    return Fragment(
        check="time_consistency",
        rejections=tuple(rejections),
        params=dict(params or {}),
        qualification=dict(qualification or {}),
        tool="fraud_detection.evaluation.time_consistency",
    )


def from_distribution_shift(
    psi_report,
    *,
    psi_threshold: float = 0.25,
    reject_degenerate: bool = False,
    params: dict | None = None,
    qualification: dict | None = None,
) -> Fragment:
    """Turn a ``Reference.psi`` report into a fragment.

    ``reject_degenerate`` covers the columns that are constant in the reference window and
    therefore unbinnable, so PSI is undefined rather than large. That is a different
    finding from "this column moved", and it is off by default: a column nothing varies in
    is uninformative, which is the redundancy check's business. Folding the two together
    would let a drift check quietly reject a third of the table for a reason that has
    nothing to do with drift.
    """
    rejections = [
        Rejection(row['column'], "distribution_shift", float(row['psi']), "column")
        for row in psi_report.iter_rows(named=True)
        if row['psi'] is not None and row['psi'] == row['psi'] and row['psi'] > psi_threshold  # NaN-safe
    ]
    if reject_degenerate:
        rejections += [
            Rejection(row['column'], "distribution_shift_degenerate", None, "column")
            for row in psi_report.iter_rows(named=True)
            if row['psi'] is None or row['psi'] != row['psi']
        ]

    return Fragment(
        check="distribution_shift",
        rejections=tuple(rejections),
        params={"psi_threshold": psi_threshold, **(params or {})},
        qualification=dict(qualification or {}),
        tool="fraud_detection.evaluation.distribution_shift",
    )


def from_segment_qualification(
    scored,
    *,
    segment: str,
    pooled_floor: float = 0.60,
    max_drop: float = 0.08,
    reject_unmeasurable: bool = False,
    reject: bool = True,
    params: dict | None = None,
    qualification: dict | None = None,
) -> Fragment:
    """Turn a `segment_qualification.qualify` table into a fragment.

    **Two thresholds, because one would be wrong.** A column is flagged when it clears
    ``pooled_floor`` — it was admitted on the strength of that number — *and* gives up more
    than ``max_drop`` inside the segment. A weak column that stays weak is not this audit's
    business, and a strong column that is somewhat weaker in a harder segment is not a
    defect either.

    ``reject`` defaults to **false**. Acting on these verdicts was measured and cost more
    than it recovered; the fragment reports `would_reject` instead. ``reject_unmeasurable``
    is separate and also off: a column constant inside the segment *cannot be scored*
    there, which is a different finding from *scored badly*.
    """
    pooled = {
        row["column"]: row["auc"]
        for row in scored.iter_rows(named=True)
        if row["segment"] == "__pooled__"
    }
    inside = {
        row["column"]: row
        for row in scored.iter_rows(named=True)
        if row["segment"] == segment
    }

    rejections, unmeasurable = [], []
    for column, pooled_auc in pooled.items():
        row = inside.get(column)
        if row is None or pooled_auc is None:
            continue
        if row["auc"] is None:
            unmeasurable.append(column)
            if reject_unmeasurable and pooled_auc > pooled_floor:
                rejections.append(
                    Rejection(column, "segment_qualification", None, f"segment:{segment}")
                )
            continue
        drop = pooled_auc - row["auc"]
        if pooled_auc > pooled_floor and drop > max_drop:
            rejections.append(
                Rejection(column, "segment_qualification", round(drop, 4), f"segment:{segment}")
            )

    if not reject:
        # Between-segment signal is spurious for within-group inference and genuinely
        # predictive here: `C` runs at a 0.1467 fraud rate against `W`'s 0.0198, so "this is
        # a C transaction" is real information about risk. The finding stays in
        # `qualification`; the rejections do not.
        rejections = []

    return Fragment(
        check="segment_qualification",
        rejections=tuple(rejections),
        params={
            "segment": segment,
            "pooled_floor": pooled_floor,
            "max_drop": max_drop,
            "reject_unmeasurable": reject_unmeasurable,
            "reject": reject,
            **(params or {}),
        },
        qualification={
            "columns_compared": len(pooled),
            "unmeasurable_in_segment": len(unmeasurable),
            "would_reject": len(rejections) if reject else len([
                c for c, a in pooled.items()
                if a is not None and (r := inside.get(c)) and r["auc"] is not None
                and a > pooled_floor and a - r["auc"] > max_drop
            ]),
            **(qualification or {}),
        },
        tool="fraud_detection.evaluation.segment_qualification",
    )


def from_admission_rules(blacklist) -> Fragment:
    """Turn the admission file's blacklist into a fragment.

    Separate from the audits on purpose. A check rejecting a column is evidence — it has a
    number attached and a later audit could reverse it. A blacklist entry is a decision
    taken for a reason no check can measure, and it should not be reversible by a better
    threshold. Recording them under one banner would make the contract unable to answer
    "was this column dropped because it failed, or because we chose to drop it".
    """
    return Fragment(
        check="admission_rules",
        rejections=tuple(
            Rejection(entry.column, "admission_rules", None, "column") for entry in blacklist
        ),
        params={"reasons": {entry.column: entry.reason for entry in blacklist}},
        tool="fraud_detection.feature_contract.admission",
    )


# ---- the gate ------------------------------------------------------------------


def assert_model_features_admitted(
    contract: FeatureContract, model_features: Sequence[str], *, fingerprint: str | None = None
) -> None:
    """Fail when a trained model and the contract disagree. Cheap enough for every run.

    Two directions, both real failures. A model using a column the contract rejected has
    silently reintroduced it. A model missing an admitted column was trained against a
    stale list, so its metrics describe a different feature set than the one that will be
    served.
    """
    admitted = set(contract.training_features())
    used = set(model_features)

    problems = []
    if forbidden := sorted(used - admitted):
        rejected_by = {c.name: c.rejected_by for c in contract.rejections()}
        detail = ", ".join(f"{c} (rejected by {rejected_by.get(c, 'not declared')})" for c in forbidden[:5])
        problems.append(f"{len(forbidden)} column(s) used but not admitted: {detail}")
    if missing := sorted(admitted - used):
        problems.append(f"{len(missing)} admitted column(s) unused: {missing[:5]}")
    if fingerprint is not None and fingerprint != contract.fingerprint():
        problems.append(f"fingerprint {fingerprint} does not match contract {contract.fingerprint()}")

    if problems:
        raise ContractError("; ".join(problems))
