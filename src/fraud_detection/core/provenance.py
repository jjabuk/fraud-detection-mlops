"""Which code produced this artifact.

The model already travels with the fingerprint of the *contract* it was trained against,
and the scoring job compares it. That closes the data half of provenance: a model cannot
be scored against a feature set it was not fitted on. The code half was open — nothing on
a `model.pkl` said which commit trained it, so reproducing a promoted model meant guessing
from timestamps, which is the same class of mistake `promotion.py` exists to remove.

Two sources, in this order:

1. ``GIT_SHA`` in the environment. This is the one that matters in production. The image is
   tagged with the git SHA and the Dockerfile bakes the same value in as an ARG, so a
   container knows its own commit without carrying a `.git` directory it has no other use
   for.
2. ``git rev-parse HEAD`` in the working tree. This is the local-development path, and it
   also reports whether the tree was dirty — a SHA from a modified checkout does not
   identify the code that ran, and saying so is more useful than pretending it does.

Neither is available in some contexts (a source tarball, a test fixture), and that is not
an error worth failing a training run over. `UNKNOWN` is a legible answer; a crash is not.

Pure on purpose: no Dagster, no cloud clients. `assets/` calls it, and so can a notebook.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["UNKNOWN_SHA", "CodeVersion", "code_version", "describe_code_version"]

UNKNOWN_SHA = "unknown"

#: Read before shelling out to git. Set by the Dockerfile ARG and by CI.
GIT_SHA_ENV = "GIT_SHA"


@dataclass(frozen=True)
class CodeVersion:
    """The commit that produced an artifact, and whether it can be trusted to."""

    sha: str
    dirty: bool = False
    source: str = "unavailable"
    """Where the SHA came from: ``env``, ``git``, or ``unavailable``."""

    @property
    def known(self) -> bool:
        return self.sha != UNKNOWN_SHA

    @property
    def short(self) -> str:
        return self.sha[:12] if self.known else UNKNOWN_SHA

    def describe(self) -> str:
        """The string that goes into metrics.json and the registry.

        A dirty tree gets a `-dirty` suffix rather than a separate field, because every
        consumer of this value is something a human reads while asking "can I rebuild
        this?", and the answer for a dirty tree is no.
        """
        if not self.known:
            return UNKNOWN_SHA
        return f"{self.sha}-dirty" if self.dirty else self.sha


def _from_env() -> CodeVersion | None:
    sha = os.getenv(GIT_SHA_ENV, "").strip()
    if not sha:
        return None
    return CodeVersion(sha=sha, dirty=False, source="env")


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _from_git(root: Path) -> CodeVersion | None:
    sha = _run_git(["rev-parse", "HEAD"], cwd=root)
    if not sha:
        return None
    # `--porcelain` prints one line per modified path and nothing at all for a clean tree,
    # so emptiness is the whole test. Untracked files count: they are code the run could
    # have imported and the commit does not carry.
    status = _run_git(["status", "--porcelain"], cwd=root)
    return CodeVersion(sha=sha, dirty=bool(status), source="git")


@lru_cache(maxsize=1)
def code_version() -> CodeVersion:
    """The commit this process is running, resolved once."""
    from_env = _from_env()
    if from_env is not None:
        return from_env

    root = Path(__file__).resolve().parents[3]
    from_git = _from_git(root)
    if from_git is not None:
        return from_git

    return CodeVersion(sha=UNKNOWN_SHA, dirty=False, source="unavailable")


def describe_code_version() -> str:
    """Shorthand for the value stamped onto artifacts."""
    return code_version().describe()
