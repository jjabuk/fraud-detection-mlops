"""The settings the audits run under, as data rather than as constants.

Thresholds live here rather than as constants across `assets/`, because the
contract recorded ``params: {}`` and you could not tell, from a contract, which numbers
produced it. Here they are one committed file, loaded into one object, and stamped into
every fragment the audits emit.

Pure layer on purpose: a notebook can load the same rules the pipeline runs under without
importing Dagster or constructing a cloud client.

Two things here are rules rather than measurement, and the contract keeps them visibly
apart from what the checks decided:

* a **blacklist** entry excludes a column for a reason no check can measure — PII, a legal
  constraint, an upstream deprecation. It becomes its own fragment.
* an **override** admits a column a check rejected. It carries a reason and an expiry, and
  a lapsed expiry fails the load: an override that nobody renews is an override that has
  quietly become permanent.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

__all__ = [
    "ADMISSION_FILE",
    "AdmissionError",
    "Blacklisted",
    "Derivation",
    "FeatureAdmissionRules",
    "Override",
    "load_admission_rules",
]

from fraud_detection.config import resolve_repo_path

ADMISSION_FILE = Path("config/feature-admission.toml")
"""Relative on purpose — resolved through `resolve_repo_path` at read time, so the
container's working directory still wins and a notebook still finds it."""


class AdmissionError(Exception):
    """Raised when the admission file is unusable — malformed, or carrying a lapsed override."""


@dataclass(frozen=True)
class Blacklisted:
    """A column excluded by decision rather than by measurement."""

    column: str
    reason: str


@dataclass(frozen=True)
class Derivation:
    """A column the contract defines rather than merely admits.

    Declared here rather than in code so that adding a candidate feature is a config diff
    somebody reviews — and so the derivation travels inside the contract, where the serving
    path can read it without importing the pipeline.
    """

    name: str
    tool: str
    inputs: tuple[str, ...]
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "tool": self.tool,
            "inputs": list(self.inputs),
            "params": self.params,
        }


@dataclass(frozen=True)
class Override:
    """An admission that overrules a check. Reason and expiry are both required."""

    column: str
    reason: str
    expires: date

    def lapsed(self, today: date | None = None) -> bool:
        return self.expires < (today or datetime.now(UTC).date())


@dataclass(frozen=True)
class FeatureAdmissionRules:
    time_consistency: dict = field(default_factory=dict)
    distribution_shift: dict = field(default_factory=dict)
    redundancy: dict = field(default_factory=dict)
    contract: dict = field(default_factory=dict)
    segment_qualification: dict = field(default_factory=dict)
    uid_aggregates: dict = field(default_factory=dict)
    blacklist: tuple[Blacklisted, ...] = ()
    overrides: tuple[Override, ...] = ()
    derivations: tuple[Derivation, ...] = ()
    version: str = "1"

    # ---- accessors the assets read -------------------------------------------------

    @property
    def time_windows(self) -> dict[str, tuple[float, float]]:
        return {
            "train": tuple(self.time_consistency.get("train_window", (0.0, 0.17))),
            "holdout": tuple(self.time_consistency.get("holdout_window", (0.83, 1.0))),
        }

    @property
    def psi_threshold(self) -> float:
        return float(self.distribution_shift.get("psi_threshold", 0.25))

    @property
    def segment_column(self) -> str:
        """Which column defines the segments the sixth audit judges within.

        Empty means the audit is off. It is a column name rather than a boolean because
        `ProductCD` is this dataset's answer and not the general one: the question is
        "which axis splits the population into groups with materially different base
        rates", and every deployment has a different answer.
        """
        return str(self.segment_qualification.get("segment_column", ""))

    @property
    def segment_enabled(self) -> bool:
        return bool(self.segment_qualification.get("enabled", False)) and bool(self.segment_column)

    @property
    def segment_pooled_floor(self) -> float:
        return float(self.segment_qualification.get("pooled_floor", 0.60))

    @property
    def segment_max_drop(self) -> float:
        return float(self.segment_qualification.get("max_drop", 0.08))

    @property
    def segment_reject(self) -> bool:
        """Whether the audit rejects, or only reports.

        Default **false**, on evidence: acting on its verdicts cost more than it
        recovered. The measurement is worth keeping; the inference is not.
        """
        return bool(self.segment_qualification.get("reject", False))

    @property
    def segment_reject_unmeasurable(self) -> bool:
        return bool(self.segment_qualification.get("reject_unmeasurable", False))

    @property
    def reject_degenerate(self) -> bool:
        return bool(self.distribution_shift.get("reject_degenerate", False))

    @property
    def min_admitted_features(self) -> int:
        return int(self.contract.get("min_admitted_features", 50))

    @property
    def max_staleness_days(self) -> int:
        return int(self.contract.get("max_staleness_days", 30))

    @property
    def uid_c_columns(self) -> tuple[str, ...]:
        return tuple(self.uid_aggregates.get("mean_encode_c", ()))

    @property
    def uid_m_columns(self) -> tuple[str, ...]:
        return tuple(self.uid_aggregates.get("mean_encode_m", ()))

    @property
    def uid_std_of_derived(self) -> tuple[str, ...]:
        """Which derived columns to take a std of, by name.

        A list rather than a flag, because not every derived column is a sensible thing to
        aggregate. `D1n` is the clearest case: it is a *component of the uid*, so its std
        within a group is identically zero by construction and the column carries no
        information at all.
        """
        return tuple(self.uid_aggregates.get("std_of_derived", ()))

    def enabled(self, check: str) -> bool:
        section = getattr(self, check, {})
        return bool(section.get("enabled", True))

    def override_for(self, column: str) -> Override | None:
        return next((o for o in self.overrides if o.column == column), None)

    # ---- provenance ----------------------------------------------------------------

    def as_dict(self) -> dict:
        """The rules as they go into the contract. Sorted, so the digest is stable."""
        return {
            "version": self.version,
            "time_consistency": self.time_consistency,
            "distribution_shift": self.distribution_shift,
            "redundancy": self.redundancy,
            "contract": self.contract,
            # In the digest deliberately: turning the sixth audit on, or moving its
            # thresholds, changes which columns reach the model, and a fingerprint that did
            # not move would describe a model trained under different rules.
            "segment_qualification": self.segment_qualification,
            "uid_aggregates": self.uid_aggregates,
            "blacklist": [{"column": b.column, "reason": b.reason} for b in self.blacklist],
            "derivations": [d.as_dict() for d in self.derivations],
            "overrides": [
                {"column": o.column, "reason": o.reason, "expires": o.expires.isoformat()}
                for o in self.overrides
            ],
        }

    def digest(self) -> str:
        """Short hash of the whole ruleset, so a contract can prove which rules made it."""
        payload = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_admission_rules(path: Path | str = ADMISSION_FILE, *, today: date | None = None) -> FeatureAdmissionRules:
    """Read and validate the admission file.

    A lapsed override raises rather than being skipped. Skipping would silently drop a
    column out of the admitted set on some future Tuesday, and the model would lose a
    feature with no diff anywhere to explain it.
    """
    # Resolved rather than taken literally: the container runs from /app and a notebook
    # from eda/notebooks/, and the same relative path has to work in both.
    path = resolve_repo_path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AdmissionError(f"no admission file at {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise AdmissionError(f"{path} is not valid TOML: {exc}") from exc

    blacklist = tuple(
        Blacklisted(column=_require(e, "column", path), reason=_require(e, "reason", path))
        for e in raw.get("blacklist", [])
    )

    overrides = []
    for entry in raw.get("override", []):
        expires = _require(entry, "expires", path)
        overrides.append(
            Override(
                column=_require(entry, "column", path),
                reason=_require(entry, "reason", path),
                expires=expires if isinstance(expires, date) else date.fromisoformat(str(expires)),
            )
        )

    if lapsed := [o for o in overrides if o.lapsed(today)]:
        detail = ", ".join(f"{o.column} (expired {o.expires})" for o in lapsed)
        raise AdmissionError(
            f"{len(lapsed)} override(s) in {path} have expired: {detail} — renew them with a "
            "current reason or delete them; an override nobody renews is a permanent one"
        )

    derivations = tuple(
        Derivation(
            name=_require(e, "name", path),
            tool=_require(e, "tool", path),
            inputs=tuple(_require(e, "inputs", path)),
            params=e.get("params", {}),
        )
        for e in raw.get("derive", [])
    )

    names = [d.name for d in derivations]
    if len(names) != len(set(names)):
        raise AdmissionError(f"duplicate derived column name(s) in {path}: {sorted(set(names))}")

    duplicates = {b.column for b in blacklist} & {o.column for o in overrides}
    if duplicates:
        raise AdmissionError(
            f"{sorted(duplicates)} appear in both blacklist and override in {path} — "
            "a column cannot be both excluded by admission rules and reinstated against a check"
        )

    return FeatureAdmissionRules(
        time_consistency=raw.get("time_consistency", {}),
        distribution_shift=raw.get("distribution_shift", {}),
        redundancy=raw.get("redundancy", {}),
        contract=raw.get("contract", {}),
        segment_qualification=raw.get("segment_qualification", {}),
        uid_aggregates=raw.get("uid_aggregates", {}),
        blacklist=blacklist,
        overrides=tuple(overrides),
        derivations=derivations,
        version=str(raw.get("meta", {}).get("version", "1")),
    )


def _require(entry: dict, key: str, path: Path):
    if key not in entry or entry[key] in ("", None):
        raise AdmissionError(f"entry {entry} in {path} is missing required field '{key}'")
    return entry[key]
