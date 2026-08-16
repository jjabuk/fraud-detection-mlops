"""Columns the contract defines rather than admits.

The property that matters is not that the arithmetic is right — it is one subtraction. It
is that **one function** computes it for the audit, for training, and for serving. A
derivation applied in training and reimplemented in the service is two definitions of one
feature, and two definitions drift; that is the failure the contract exists to prevent.
"""

from __future__ import annotations

import polars as pl
import pytest

from fraud_detection.core.feature_contract import FeatureContract, Source, load_admission_rules
from fraud_detection.core.feature_contract.admission import Derivation
from fraud_detection.core.feature_contract.declaration import declare_columns
from fraud_detection.feature_engineering.derivations import (
    DERIVATIONS,
    DerivationError,
    apply_derivations,
    days_since_to_start_day,
)


def frame() -> pl.DataFrame:
    # Two transactions of one card, six days apart. D1 counts days since the card began,
    # so both rows must recover the *same* start day -- that constancy is what makes the
    # normalised column usable as an identity component and as a non-drifting feature.
    return pl.DataFrame(
        {
            "TransactionDT": [86_400 * 10, 86_400 * 16],
            "D1": [3.0, 9.0],
            "D15": [1.0, None],
        }
    )


D1N = Derivation(name="D1n", tool="days_since_to_start_day", inputs=("D1",))
D15N = Derivation(name="D15n", tool="days_since_to_start_day", inputs=("D15",))


def test_the_normalised_column_is_constant_across_one_card_history():
    derived = apply_derivations(frame(), [D1N])

    assert derived.get_column("D1n").to_list() == [7.0, 7.0]


def test_a_missing_counter_yields_no_start_day():
    # Filling it would invent one, and every row missing the counter would share a single
    # fictitious value -- the same failure as sentinel-filling an entity key, which
    # measured worse than not reconstructing at all.
    derived = apply_derivations(frame(), [D15N])

    assert derived.get_column("D15n").is_null().to_list() == [False, True]


def test_the_source_frame_is_not_mutated():
    original = frame()
    apply_derivations(original, [D1N])

    assert "D1n" not in original.columns


def test_an_unknown_tool_fails_rather_than_being_skipped():
    unknown = Derivation(name="X", tool="not_a_tool", inputs=("D1",))

    with pytest.raises(DerivationError, match="not in DERIVATIONS"):
        apply_derivations(frame(), [unknown])


def test_a_missing_input_fails_rather_than_being_skipped():
    # A derivation that silently does not happen produces a model trained on fewer columns
    # than the contract says, and the published metrics describe something never fitted.
    missing = Derivation(name="D99n", tool="days_since_to_start_day", inputs=("D99",))

    with pytest.raises(DerivationError, match="drifted apart"):
        apply_derivations(frame(), [missing])


# ---- how a derived column reaches the contract ---------------------------------


def test_a_derived_column_is_declared_derived_not_requested():
    # The caller cannot be asked to send D1n: that would be asking them to run our
    # arithmetic. And there is nothing to look up, so it is not retrieved either.
    declared = declare_columns(
        {"TransactionAmt": "float64", "D1": "float64", "D1n": "float64"},
        derived=frozenset({"D1n"}),
    )

    assert declared["D1n"][0] == Source.DERIVED
    assert declared["TransactionAmt"][0] == Source.REQUEST


def test_a_derived_column_stays_out_of_the_request_schema():
    contract = FeatureContract.build(
        {
            "TransactionAmt": (Source.REQUEST, "float"),
            "D1": (Source.REQUEST, "float"),
            "D1n": (Source.DERIVED, "float"),
        },
        [],
    )

    assert "D1n" in contract.training_features()
    assert "D1n" not in contract.request_model().model_fields
    # The input it is computed from still has to arrive.
    assert "D1" in contract.request_model().model_fields


# ---- the committed policy ------------------------------------------------------


def test_the_committed_derivations_are_computable():
    # Every declared derivation names a tool that exists and inputs the model input has.
    # A contract that defines a column nothing can compute is worse than one that omits it.
    for derivation in load_admission_rules().derivations:
        assert derivation.tool in DERIVATIONS, derivation.name


def test_the_winners_seven_d_columns_are_declared():
    # Yakovlev's uid-detection notebook normalises D1, D2, D3, D5, D10, D11 and D15. This
    # is the point-in-time-computable half of the winning solution's feature work, and the
    # audits now judge all seven on the same terms as any raw column.
    # Scoped to the D-normalisation tool. The equality this used to assert over *every*
    # declared name broke the moment a second family of derivations was declared, which
    # made it a test of "nothing else exists" rather than of "these seven do".
    declared = {
        d.name for d in load_admission_rules().derivations if d.tool == "days_since_to_start_day"
    }

    assert declared == {"D1n", "D2n", "D3n", "D5n", "D10n", "D11n", "D15n"}


def test_the_derivation_is_the_one_the_client_uid_already_used():
    # features.CLIENT_UID_EXPRESSION computes floor(TransactionDT/86400) - D1 in SQL. The
    # Python derivation must agree with it, or the entity the aggregates are grouped by and
    # the feature the model sees would be two different quantities.
    day = (frame().get_column("TransactionDT") / 86_400).floor()
    expected = day - frame().get_column("D1")

    assert days_since_to_start_day(frame(), ("D1",), {}).to_list() == expected.to_list()


# ---- encodings, 2026-08-16 --------------------------------------------------------

import json as _json
from pathlib import Path

import polars as _pl
import pytest as _pytest

from fraud_detection.feature_engineering.derivations import (
    DerivationError as _DerivationError,
)
from fraud_detection.feature_engineering.derivations import (
    frequency_encode,
    one_hot,
)


def test_one_hot_marks_the_level_and_nothing_else():
    frame = _pl.DataFrame({"ProductCD": ["W", "C", "W", "S"]})
    out = one_hot(frame, ["ProductCD"], {"level": "W"})
    assert out.to_list() == [1, 0, 1, 0]


def test_one_hot_keeps_null_as_null():
    """M4 is null on 47.7% of rows. Mapping that to 0 would assert "not M0" about rows
    where nobody recorded M4 at all — the conflation would be most of the column."""
    frame = _pl.DataFrame({"M4": ["M0", None, "M2"]})
    assert one_hot(frame, ["M4"], {"level": "M0"}).to_list() == [1, None, 0]


def test_one_hot_of_an_unknown_level_is_all_zero_not_an_error():
    """A category that vanishes from the data must not break the run: the declared column
    still exists, it is simply 0. That is the point of pinning levels in config."""
    frame = _pl.DataFrame({"card6": ["debit", "credit"]})
    assert one_hot(frame, ["card6"], {"level": "charge card"}).to_list() == [0, 0]


def test_frequency_encode_reads_the_pinned_map_not_the_frame():
    """The whole reason this is a fitted transform. If it counted the frame, these two
    frames would encode the same value differently."""
    maps = {"card1": {"1000": 500, "2000": 7}}
    small = _pl.DataFrame({"card1": ["1000", "2000"]})
    large = _pl.DataFrame({"card1": ["1000"] * 50 + ["2000"]})

    assert frequency_encode(small, ["card1"], {"_maps": maps}).to_list() == [500, 7]
    assert frequency_encode(large, ["card1"], {"_maps": maps}).to_list()[:2] == [500, 500]


def test_an_unseen_value_encodes_as_null_not_zero():
    """Zero would claim the value occurs zero times, which is false — it occurs in the row
    being scored. Null says what is true: the training window has nothing to say."""
    maps = {"card1": {"1000": 500}}
    frame = _pl.DataFrame({"card1": ["1000", "9999", None]})
    assert frequency_encode(frame, ["card1"], {"_maps": maps}).to_list() == [500, None, None]


def test_a_missing_map_fails_loudly():
    with _pytest.raises(_DerivationError, match="no frequency map"):
        frequency_encode(_pl.DataFrame({"card1": ["1"]}), ["card1"], {"_maps": {}})


def test_the_committed_map_is_fitted_on_the_training_split_only():
    """A map fitted on train ∪ test would be transductive — the thing the noise-band entry
    and adversarial-drift.md both refuse."""
    payload = _json.loads(Path("references/frequency-maps.json").read_text())
    assert payload["fitted_on"]["split"] == "train"
    assert payload["fitted_on"]["rows"] == 442_905
    assert payload["min_count"] >= 2
    assert set(payload["maps"]) == {"addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31"}


def test_the_frequency_columns_were_chosen_on_evidence_not_cardinality():
    """card1 has 12,485 levels and no frequency signal: fraud rate across its frequency
    bands runs 0.85 / 0.76 / 1.19 / 0.98 times the base, with no direction. It was in the
    map until that was measured. High cardinality is what makes this encoding *possible*,
    not what makes it worth having."""
    payload = _json.loads(Path("references/frequency-maps.json").read_text())
    assert "card1" not in payload["maps"]
    assert "card2" not in payload["maps"]
