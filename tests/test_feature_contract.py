"""The contract is the one place three consumers agree. These tests are that guarantee."""

from __future__ import annotations

import json

import pytest

from fraud_detection.contract import (
    ContractError,
    FeatureContract,
    Fragment,
    Rejection,
    Source,
    assert_model_features_admitted,
    fragment_from_dict,
    read_fragments,
)

DECLARED = {
    "TransactionAmt": (Source.REQUEST, "float"),
    "ProductCD": (Source.REQUEST, "str"),
    "V322": (Source.REQUEST, "float"),
    "V323": (Source.REQUEST, "float"),
    "V335": (Source.REQUEST, "float"),
    "card_txn_count_1h": (Source.RETRIEVED, "float"),
}


def contract(*fragments, **kw) -> FeatureContract:
    return FeatureContract.build(DECLARED, fragments, **kw)


def test_everything_is_admitted_when_no_check_objects():
    c = contract()
    assert c.training_features() == list(DECLARED)
    assert c.rejections() == []


def test_a_rejected_column_leaves_the_training_list():
    c = contract(Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.104),)))

    assert "V335" not in c.training_features()
    assert [r.name for r in c.rejections()] == ["V335"]
    assert c.rejections()[0].rejected_by == "time_consistency"
    assert c.rejections()[0].rejected_value == -0.104


def test_the_first_check_to_reject_a_column_owns_the_reason():
    """Two checks can both dislike a column. "Why is this missing" needs one answer."""
    c = contract(
        Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1),)),
        Fragment("distribution_shift", (Rejection("V335", "distribution_shift", 0.4),)),
    )
    assert c.rejections()[0].rejected_by == "time_consistency"


def test_a_check_rejecting_an_undeclared_column_is_an_error():
    """The audit ran on a different table than the one being served."""
    with pytest.raises(ContractError, match="absent from the declaration"):
        contract(Fragment("time_consistency", (Rejection("V999", "time_consistency", -0.1),)))


def test_the_three_consumers_agree():
    c = contract(Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1),)))

    assert set(c.monitored_columns()) == set(c.training_features())
    assert {f.name for f in c.request_fields()} | {f.name for f in c.retrieved_fields()} == set(
        c.training_features()
    )


def test_retrieved_columns_never_enter_the_request_schema():
    """A caller cannot know an entity's history, so asking for it makes the schema a lie."""
    model = contract().request_model()

    assert "TransactionAmt" in model.model_fields
    assert "card_txn_count_1h" not in model.model_fields


def test_request_fields_are_optional_because_the_data_is_mostly_missing():
    model = contract().request_model()
    assert model().TransactionAmt is None
    assert model(TransactionAmt=12.5).TransactionAmt == 12.5


def test_a_rejected_column_disappears_from_the_request_schema_too():
    c = contract(Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1),)))
    assert "V335" not in c.request_model().model_fields


# ---- provenance ---------------------------------------------------------------


def test_fingerprint_tracks_the_admitted_set_and_not_the_timestamp():
    a, b = contract(), contract()
    assert a.fingerprint() == b.fingerprint()

    rejected = contract(Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1),)))
    assert rejected.fingerprint() != a.fingerprint()


def test_json_round_trip_preserves_the_decisions():
    c = contract(
        Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1, "block:V322-V339"),)),
        data={"table": "features.model_input", "rows": 590_540},
        entity={"columns": ["card1", "addr1"], "anchor": "D1"},
    )
    back = FeatureContract.from_json(c.to_json())

    assert back.training_features() == c.training_features()
    assert back.fingerprint() == c.fingerprint()
    assert back.entity == c.entity
    assert back.rejections()[0].rejected_unit == "block:V322-V339"


def test_round_trip_preserves_the_policy_the_fingerprint_is_computed_over():
    """The round trip above passes with an empty policy, which is how this got through.

    `fingerprint()` hashes the admitted set **and** the admission rules, on the argument
    that a threshold moving without the column list moving still has to produce a new
    fingerprint. But `to_dict` writes that field as "policy" while `from_dict` read it
    back from "admission_rules", so every contract loaded from disk recomputed its hash
    over `{}`.

    The consequence was not subtle: every contract ever written failed its own integrity
    check with "the file was edited by hand", and the fingerprint stamped onto a trained
    model could never be reconciled with the contract on disk. The existing round-trip
    test missed it because its contract carries no policy, so both sides hashed `{}` and
    agreed.
    """
    policy = {"distribution_shift": {"psi_threshold": 0.5}, "time_consistency": {"reject_by_block": True}}
    c = contract(admission_rules=policy)

    back = FeatureContract.from_json(c.to_json())

    assert back.admission_rules == policy
    assert back.fingerprint() == c.fingerprint()


def test_a_policy_change_alone_moves_the_fingerprint():
    """The property the policy is in the payload for, checked through the file."""
    loose = contract(admission_rules={"distribution_shift": {"psi_threshold": 0.5}})
    strict = contract(admission_rules={"distribution_shift": {"psi_threshold": 0.25}})

    assert loose.training_features() == strict.training_features()
    assert loose.fingerprint() != strict.fingerprint()
    # And it must survive serialization, or the distinction exists only in memory.
    assert FeatureContract.from_json(loose.to_json()).fingerprint() != (
        FeatureContract.from_json(strict.to_json()).fingerprint()
    )


def test_hand_editing_the_file_is_caught():
    payload = contract().to_dict()
    payload["columns"] = [c for c in payload["columns"] if c["name"] != "V322"]

    with pytest.raises(ContractError, match="fingerprint mismatch"):
        FeatureContract.from_dict(payload)


# ---- fragments, as the audits write them ---------------------------------------

# The audits live in analysis/ and are R. What crosses the boundary is a JSON
# fragment per check, so these are tests of a file format rather than of a
# statistic -- the statistics have their own tests, in testthat, next to them.


def _fragment_file(tmp_path, check, rejections, tool="fraudaudit::test"):
    payload = {
        "check": check,
        "tool": tool,
        "params": {"threshold": 0.25},
        "qualification": {"columns_scanned": 3},
        "rejections": rejections,
    }
    (tmp_path / f"{check}.json").write_text(json.dumps(payload))
    return payload


def test_a_fragment_round_trips_from_the_shape_r_writes():
    frag = fragment_from_dict(
        {
            "check": "time_consistency",
            "tool": "fraudaudit::time_consistency_scan",
            "params": {"margin": 0.02},
            "qualification": {"features_scanned": 377},
            "rejections": [{"column": "V1", "check": "time_consistency", "value": -0.11,
                            "unit": "column"}],
        }
    )
    assert frag.check == "time_consistency"
    assert frag.rejections[0].column == "V1"
    assert frag.rejections[0].value == pytest.approx(-0.11)
    assert frag.params["margin"] == 0.02


def test_a_block_rejection_keeps_the_unit_that_says_so():
    # A column can pass on its own and still be rejected as part of a block. The
    # unit is what stops that being reversed later by someone reading the
    # per-column number in isolation.
    frag = fragment_from_dict(
        {
            "check": "time_consistency",
            "tool": "fraudaudit::time_consistency_scan",
            "rejections": [{"column": "V13", "value": -0.07, "unit": "block:V12-V34"}],
        }
    )
    assert frag.rejections[0].unit == "block:V12-V34"


def test_a_missing_number_is_none_however_r_spelled_it():
    # jsonlite's default renders a missing number as the string "NA", which is
    # valid JSON and a type error here. Accepted rather than crashing at stamping
    # time, long after the run that produced it.
    frag = fragment_from_dict(
        {
            "check": "redundancy",
            "tool": "fraudaudit::redundancy_scan",
            "rejections": [
                {"column": "V1", "value": None, "unit": "column"},
                {"column": "V2", "value": "NA", "unit": "column"},
            ],
        }
    )
    assert all(r.value is None for r in frag.rejections)


def test_a_fragment_that_does_not_name_its_tool_is_refused():
    # A verdict whose origin is not recorded cannot be reproduced, and the
    # contract would be asserting something nobody can check.
    with pytest.raises(ContractError, match="tool"):
        fragment_from_dict({"check": "time_consistency", "rejections": []})


def test_reading_fragments_applies_the_precedence_given(tmp_path):
    _fragment_file(tmp_path, "time_consistency", [{"column": "a", "value": -0.1}])
    _fragment_file(tmp_path, "redundancy", [{"column": "a", "value": 0.95}])
    order = ("redundancy", "time_consistency")
    assert [f.check for f in read_fragments(tmp_path, order)] == list(order)


def test_a_missing_fragment_is_an_error_not_an_omission(tmp_path):
    # A contract assembled from three of four audits is not a weaker contract, it
    # is a different one, and it would carry no sign of which check never ran.
    _fragment_file(tmp_path, "time_consistency", [])
    with pytest.raises(ContractError, match="redundancy"):
        read_fragments(tmp_path, ("time_consistency", "redundancy"))


def test_fragments_carry_their_qualification_into_the_contract():
    frag = fragment_from_dict(
        {"check": "time_consistency", "tool": "fraudaudit::x", "rejections": [],
         "qualification": {"reproduced_share": 0.35}}
    )
    contract = FeatureContract.build(DECLARED, [frag])
    assert contract.fragments[0].qualification["reproduced_share"] == 0.35


# ---- the gate -----------------------------------------------------------------


def test_gate_passes_when_the_model_matches_the_contract():
    c = contract()
    assert_model_features_admitted(c, c.training_features(), fingerprint=c.fingerprint())


def test_gate_catches_a_model_using_a_rejected_column():
    c = contract(Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1),)))

    with pytest.raises(ContractError, match="used but not admitted"):
        assert_model_features_admitted(c, [*c.training_features(), "V335"])


def test_gate_names_the_check_that_rejected_the_column():
    c = contract(Fragment("time_consistency", (Rejection("V335", "time_consistency", -0.1),)))

    with pytest.raises(ContractError, match="rejected by time_consistency"):
        assert_model_features_admitted(c, [*c.training_features(), "V335"])


def test_gate_catches_a_model_trained_on_a_stale_list():
    c = contract()
    with pytest.raises(ContractError, match="unused"):
        assert_model_features_admitted(c, c.training_features()[:-1])


def test_gate_catches_a_contract_swapped_under_the_model():
    c = contract()
    with pytest.raises(ContractError, match="does not match"):
        assert_model_features_admitted(c, c.training_features(), fingerprint="deadbeefdeadbeef")


# ---- declaration: the dtype vocabularies ---------------------------------------


def test_pandas_string_columns_are_declared_str_not_float():
    # The failure this catches is silent and lands in the serving API: pandas calls a
    # string column `object`, and a BigQuery-only type table has no entry for it. The old
    # `float` fallback therefore typed ProductCD and DeviceInfo as numbers, and
    # request_model() would have emitted a schema rejecting the text a caller actually
    # sends.
    from fraud_detection.contract.declaration import declare_columns

    declared = declare_columns(
        {
            "ProductCD": "object",
            "DeviceInfo": "category",
            "TransactionAmt": "float64",
            "card1": "int64",
            "card_seen": "bool",
        },
        excluded=frozenset(),
    )

    assert declared["ProductCD"] == (Source.REQUEST, "str")
    assert declared["DeviceInfo"] == (Source.REQUEST, "str")
    assert declared["TransactionAmt"] == (Source.REQUEST, "float")
    assert declared["card1"] == (Source.REQUEST, "int")
    assert declared["card_seen"] == (Source.REQUEST, "bool")


def test_bigquery_type_names_still_map():
    from fraud_detection.contract.declaration import declare_columns

    declared = declare_columns(
        {"ProductCD": "STRING", "TransactionAmt": "FLOAT64", "n": "INTEGER"},
        excluded=frozenset(),
    )

    assert [d[1] for d in declared.values()] == ["str", "float", "int"]


def test_engineered_columns_are_retrieved_not_requested():
    # A request schema that asked the caller for card_txn_count_24h would move the hard
    # half of the problem outside the system boundary.
    from fraud_detection.contract.declaration import declare_columns

    declared = declare_columns({"card_txn_count_24h": "float64", "TransactionAmt": "float64"})

    assert declared["card_txn_count_24h"][0] == Source.RETRIEVED
    assert declared["TransactionAmt"][0] == Source.REQUEST


def test_request_model_types_follow_the_declaration():
    c = contract()
    model = c.request_model()

    fields = model.model_fields
    assert fields["ProductCD"].annotation == (str | None)
    assert fields["TransactionAmt"].annotation == (float | None)
    # Retrieved columns are never in the request schema.
    assert "card_txn_count_1h" not in fields


def test_the_policy_section_moves_the_fingerprint():
    """Turning the sixth audit on changes which columns reach the model, so a fingerprint
    that did not move would describe a model trained under different rules."""
    from fraud_detection.contract.admission import FeatureAdmissionRules

    off = FeatureAdmissionRules(segment_qualification={"enabled": False})
    on = FeatureAdmissionRules(segment_qualification={"enabled": True, "segment_column": "ProductCD"})

    assert off.as_dict() != on.as_dict()
    assert on.segment_enabled and not off.segment_enabled
