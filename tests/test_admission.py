"""The policy file, and the two things in it that are decisions rather than measurements.

A blacklist entry and an override are the places where a human overrules the audit. Both
are legitimate and both are how this kind of system rots, so the rules around them are
tested rather than trusted: an override carries a reason and an expiry, a lapsed expiry
fails loudly, and neither ever reaches the contract disguised as an ordinary admission.
"""

from __future__ import annotations

from datetime import date

import pytest

from fraud_detection.core.feature_contract import (
    ContractError,
    FeatureContract,
    Fragment,
    Rejection,
    Source,
    from_admission_rules,
    load_admission_rules,
)
from fraud_detection.core.feature_contract.admission import ADMISSION_FILE, AdmissionError, Override

DECLARED = {
    "TransactionAmt": (Source.REQUEST, "float"),
    "V322": (Source.REQUEST, "float"),
    "V335": (Source.REQUEST, "float"),
}

REJECTS_V335 = Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.104),))


def write(tmp_path, body: str):
    path = tmp_path / "feature-admission.toml"
    path.write_text(body)
    return path


# ---- the committed policy ------------------------------------------------------


def test_the_committed_policy_loads():
    # It is read by three assets and two checks; a syntax error in it takes the pipeline
    # down, so the fact that it parses is worth one test.
    rules = load_admission_rules(ADMISSION_FILE)

    assert rules.psi_threshold > 0
    assert rules.max_staleness_days > 0
    assert rules.min_admitted_features > 0


def test_the_policy_digest_changes_with_the_policy(tmp_path):
    a = load_admission_rules(write(tmp_path, "[distribution_shift]\npsi_threshold = 0.25\n"))
    b = load_admission_rules(write(tmp_path, "[distribution_shift]\npsi_threshold = 0.30\n"))

    assert a.digest() != b.digest()


# ---- overrides -----------------------------------------------------------------


def test_a_lapsed_override_fails_the_load(tmp_path):
    # Skipping it silently would drop a column out of the admitted set on some future
    # Tuesday, and the model would lose a feature with no diff anywhere to explain it.
    path = write(
        tmp_path,
        '[[override]]\ncolumn = "V335"\nreason = "measured per-column"\nexpires = 2020-01-01\n',
    )

    with pytest.raises(AdmissionError, match="expired"):
        load_admission_rules(path)


def test_an_override_without_a_reason_is_rejected(tmp_path):
    path = write(tmp_path, '[[override]]\ncolumn = "V335"\nexpires = 2099-01-01\n')

    with pytest.raises(AdmissionError, match="reason"):
        load_admission_rules(path)


def test_a_column_cannot_be_both_blacklisted_and_overridden(tmp_path):
    path = write(
        tmp_path,
        '[[blacklist]]\ncolumn = "V335"\nreason = "pii"\n\n'
        '[[override]]\ncolumn = "V335"\nreason = "keep"\nexpires = 2099-01-01\n',
    )

    with pytest.raises(AdmissionError, match="both"):
        load_admission_rules(path)


def test_an_override_admits_the_column_and_still_records_the_objection():
    override = Override("V335", "per-column delta is +0.004", date(2099, 1, 1))
    c = FeatureContract.build(DECLARED, [REJECTS_V335], overrides=[override])

    assert "V335" in c.training_features()

    column = next(col for col in c.columns if col.name == "V335")
    assert column.admitted is True
    assert column.admitted_by == "override"
    # The check that objected is still named. An override that erased the objection would
    # be an override nobody could audit later.
    assert column.rejected_by == "time_consistency"
    assert column.override_reason == "per-column delta is +0.004"
    assert c.overrides() == [column]


def test_an_override_for_an_undeclared_column_is_a_typo_not_a_decision():
    override = Override("V999", "reinstate", date(2099, 1, 1))

    with pytest.raises(ContractError, match="not declared"):
        FeatureContract.build(DECLARED, [REJECTS_V335], overrides=[override])


def test_an_override_survives_a_serialization_round_trip():
    override = Override("V335", "measured", date(2099, 1, 1))
    c = FeatureContract.build(DECLARED, [REJECTS_V335], overrides=[override])

    back = FeatureContract.from_json(c.to_json())

    assert [col.name for col in back.overrides()] == ["V335"]
    assert back.fingerprint() == c.fingerprint()


# ---- the policy fragment -------------------------------------------------------


def test_a_blacklisted_column_is_rejected_under_its_own_check_name(tmp_path):
    rules = load_admission_rules(
        write(tmp_path, '[[blacklist]]\ncolumn = "V322"\nreason = "carries PII"\n')
    )
    c = FeatureContract.build(DECLARED, [from_admission_rules(rules.blacklist)])

    column = next(col for col in c.columns if col.name == "V322")
    # Not "redundancy", not "distribution_shift". A decision and a measurement must not
    # look alike in the contract: one is reversible by a better threshold, the other is not.
    assert column.rejected_by == "admission_rules"
    assert column.admitted is False


def test_policy_wins_the_attribution_when_a_check_also_rejects():
    # Fragment order decides who owns a rejection. Policy goes first so "we chose to
    # exclude this" is not displaced in the record by a check that happened to agree.
    admission_fragment = from_admission_rules([type("E", (), {"column": "V335", "reason": "pii"})()])
    c = FeatureContract.build(DECLARED, [admission_fragment, REJECTS_V335])

    assert next(col for col in c.columns if col.name == "V335").rejected_by == "admission_rules"


# ---- the fingerprint covers the rules, not just the outcome --------------------


def test_the_fingerprint_moves_when_a_threshold_moves_even_if_the_columns_do_not():
    # The failure this prevents: a threshold changes, the admitted set happens not to, and
    # two models pinned to "the same" contract were trained under different rules.
    a = FeatureContract.build(DECLARED, [], admission_rules={"distribution_shift": {"psi_threshold": 0.25}})
    b = FeatureContract.build(DECLARED, [], admission_rules={"distribution_shift": {"psi_threshold": 0.30}})

    assert a.training_features() == b.training_features()
    assert a.fingerprint() != b.fingerprint()


# ---- the uid-aggregate readmission, 2026-08-16 ------------------------------------
#
# Nineteen overrides taken as one decision, argued at length in the policy file itself.
# Pinned here because the whole point of an override is that it is visible: a family of
# features silently dropping back out of the contract on an unrelated edit is exactly the
# failure this block exists to prevent.

UID_OVERRIDES = {
    "client_amt_deviation_prior",
    "client_d2n_std_prior",
    "client_d5n_std_prior",
    "client_d10n_std_prior",
    "client_d11n_std_prior",
    "client_d15n_std_prior",
    "client_c3_mean_prior",
    "client_c4_mean_prior",
    "client_c7_mean_prior",
    "client_c8_mean_prior",
    "client_c9_mean_prior",
    "client_c10_mean_prior",
    "client_c11_mean_prior",
    "client_c12_mean_prior",
    "client_c13_mean_prior",
    "client_c14_mean_prior",
    "client_m2_mean_prior",
    "client_m3_mean_prior",
    "client_m5_mean_prior",
}

# Deliberately still rejected: distribution_shift is a different check with a different
# reproducibility record (1.00 against time_consistency's 0.32), so overruling it is a
# separate decision needing its own evidence. Folding them in would also make the delta
# from this experiment unattributable.
STILL_REJECTED_BY_PSI = {
    "client_txn_count_prior",
    "seconds_since_prev_txn_client",
    "client_d3n_std_prior",
    "client_m8_mean_prior",
}


def test_the_uid_override_was_retired_on_evidence():
    """It was applied, measured, and withdrawn: 0.8954 with, 0.8951 without, over five
    seeds — a tenth of the resolution. The names are kept here so a future readmission is a
    deliberate act rather than a drift back."""
    rules = load_admission_rules()
    overridden = {o.column for o in rules.overrides}
    assert not (UID_OVERRIDES & overridden), sorted(UID_OVERRIDES & overridden)


def test_the_psi_rejected_uid_columns_are_left_alone():
    """One change, one measurement. This test is what keeps that true."""
    rules = load_admission_rules()
    overridden = {o.column for o in rules.overrides}
    assert not (STILL_REJECTED_BY_PSI & overridden), sorted(
        STILL_REJECTED_BY_PSI & overridden
    )


def test_any_override_carries_a_reason_and_a_future_expiry():
    """The rule, not the instance: whatever is overridden must justify itself and lapse."""
    rules = load_admission_rules()
    for override in rules.overrides:
        assert len(override.reason) > 40, override.column
        assert override.expires > date(2026, 8, 16), override.column
