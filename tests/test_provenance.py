"""The commit that produced an artifact has to be recoverable from the artifact.

These tests are about the *reporting*, not about git. The resolution order is the part
worth pinning: a container has no `.git`, so if the environment variable ever stopped
winning, every production model would silently start reporting `unknown` and nothing would
fail.
"""

from __future__ import annotations

import json

from fraud_detection.core.promotion import parse_promotion_marker
from fraud_detection.core.provenance import (
    UNKNOWN_SHA,
    CodeVersion,
    _from_env,
    _from_git,
    code_version,
    describe_code_version,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_env_is_read_when_set(monkeypatch):
    monkeypatch.setenv("GIT_SHA", SHA)
    resolved = _from_env()
    assert resolved is not None
    assert resolved.sha == SHA
    assert resolved.source == "env"
    assert resolved.dirty is False


def test_blank_env_is_not_a_version(monkeypatch):
    """An unstamped image sets GIT_SHA to the empty string, which is not an answer."""
    monkeypatch.setenv("GIT_SHA", "   ")
    assert _from_env() is None


def test_env_wins_over_git(monkeypatch):
    """The container path. Without this the image would report the SHA of whatever tree
    happened to be mounted, or nothing at all."""
    monkeypatch.setenv("GIT_SHA", SHA)
    code_version.cache_clear()
    try:
        assert code_version().sha == SHA
        assert code_version().source == "env"
    finally:
        code_version.cache_clear()


def test_git_is_the_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("GIT_SHA", raising=False)
    code_version.cache_clear()
    try:
        resolved = code_version()
    finally:
        code_version.cache_clear()
    # The repository this test runs in is a git checkout, so this resolves. If it ever
    # runs from an exported tarball the honest answer is `unavailable`, and that is a
    # valid outcome rather than a failure.
    assert resolved.source in {"git", "unavailable"}
    if resolved.source == "git":
        assert len(resolved.sha) == 40


def test_non_repository_reports_unavailable(tmp_path):
    assert _from_git(tmp_path) is None


def test_dirty_tree_is_marked():
    """A SHA from a modified checkout does not identify the code that ran."""
    assert CodeVersion(sha=SHA, dirty=True, source="git").describe() == f"{SHA}-dirty"
    assert CodeVersion(sha=SHA, dirty=False, source="git").describe() == SHA


def test_unknown_describes_as_unknown():
    unknown = CodeVersion(sha=UNKNOWN_SHA)
    assert unknown.known is False
    assert unknown.describe() == UNKNOWN_SHA
    assert unknown.short == UNKNOWN_SHA


def test_short_is_twelve_characters():
    assert CodeVersion(sha=SHA).short == SHA[:12]


def test_describe_code_version_returns_a_string(monkeypatch):
    monkeypatch.setenv("GIT_SHA", SHA)
    code_version.cache_clear()
    try:
        assert describe_code_version() == SHA
    finally:
        code_version.cache_clear()


def test_marker_carries_code_version_through_to_the_scoring_path():
    """The end of the chain: what the gate writes is what the batch job can read."""
    marker = json.dumps(
        {
            "alias": "production",
            "artifact_prefix": "gs://bucket/lightgbm/run-1",
            "code_version": SHA,
            "contract_fingerprint": "6ea8eb8b936c9e57",
        }
    )
    promoted = parse_promotion_marker(marker)
    assert promoted.code_version == SHA
    assert promoted.contract_fingerprint == "6ea8eb8b936c9e57"


def test_marker_without_provenance_still_parses():
    """Markers written before this existed must not become unreadable."""
    marker = json.dumps({"artifact_prefix": "gs://bucket/lightgbm/run-0"})
    promoted = parse_promotion_marker(marker)
    assert promoted.code_version == ""
    assert promoted.contract_fingerprint == ""
