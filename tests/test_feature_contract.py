"""The contract is the one place three consumers agree. These tests are that guarantee."""

from __future__ import annotations

import polars as pl
import pytest

from fraud_detection.core.feature_contract import (
    ContractError,
    FeatureContract,
    Fragment,
    Rejection,
    Source,
    assert_model_features_admitted,
    from_distribution_shift,
    from_time_consistency,
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


# ---- adapters -----------------------------------------------------------------


def scan_report(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["feature", "verdict", "delta"], orient="row")


def test_time_consistency_adapter_takes_only_the_inversions():
    frag = from_time_consistency(
        scan_report([("V335", "inverted", -0.104), ("V322", "pass", -0.077), ("V323", "weak", -0.001)])
    )
    assert [r.column for r in frag.rejections] == ["V335"]
    assert frag.rejections[0].unit == "column"


def test_a_block_falls_whole_when_any_member_inverts():
    """V322 passes on its own. It is still out: its family degrades together, and a
    per-column threshold splits that family at an arbitrary point."""
    frag = from_time_consistency(
        scan_report([("V335", "inverted", -0.104), ("V322", "pass", -0.077), ("V323", "pass", -0.075)]),
        blocks={"V322-V339": ["V322", "V323", "V335"]},
    )

    assert {r.column for r in frag.rejections} == {"V322", "V323", "V335"}
    assert all(r.unit == "block:V322-V339" for r in frag.rejections)
    assert all(r.value == -0.104 for r in frag.rejections)  # the block's worst


def test_a_block_with_no_inversions_is_left_alone():
    frag = from_time_consistency(
        scan_report([("V322", "pass", -0.01)]), blocks={"V322-V339": ["V322", "V323"]}
    )
    assert frag.rejections == ()


def test_distribution_shift_adapter_uses_the_psi_threshold():
    psi = pl.DataFrame({"column": ["V335", "V322", "V323"], "psi": [0.40, 0.12, float("nan")]})
    frag = from_distribution_shift(psi, psi_threshold=0.25, reject_degenerate=True)

    assert {r.column for r in frag.rejections} == {"V335", "V323"}
    assert frag.params["psi_threshold"] == 0.25


def test_an_unmeasurable_column_is_rejected_with_no_value():
    psi = pl.DataFrame({"column": ["V323"], "psi": [float("nan")]})
    frag = from_distribution_shift(psi, reject_degenerate=True)

    assert frag.rejections[0].by == "distribution_shift_degenerate"
    assert frag.rejections[0].value is None


def test_a_constant_column_is_not_drift_and_is_left_alone_by_default():
    # PSI is undefined for a column that never varies in the reference window, which is a
    # different finding from "this column moved". On this dataset it covers 158 of 444
    # columns -- a third of the table rejected by a drift check for a reason that is not
    # drift. Off unless the policy asks for it.
    psi = pl.DataFrame({"column": ["V323"], "psi": [float("nan")]})

    assert from_distribution_shift(psi).rejections == ()


def test_a_naive_threshold_filter_would_reject_every_degenerate_column():
    """Why the adapter exists, rather than `filter(psi > threshold)` at each call site.

    Polars orders NaN **above** every number, so the obvious filter treats "PSI is
    undefined here" as "this column moved more than anything else". Written out because it
    already happened: an audit qualification that filtered by hand reported 210 rejections
    where the contract had 45, and the 165 extra were exactly the degenerate columns —
    the outcome `reject_degenerate = false` is there to prevent.
    """
    psi = pl.DataFrame({"column": ["shifted", "stable", "degenerate"], "psi": [0.9, 0.1, float("nan")]})

    naive = psi.filter(pl.col("psi") > 0.5)["column"].to_list()
    assert naive == ["shifted", "degenerate"], "polars NaN ordering changed; the guard below is why"

    admitted_out = [r.column for r in from_distribution_shift(psi, psi_threshold=0.5).rejections]
    assert admitted_out == ["shifted"]


def test_fragments_carry_their_qualification_into_the_contract():
    c = contract(
        Fragment("time_consistency", (), qualification={"verdict_stability_0.25": 0.87}),
    )
    assert c.to_dict()["fragments"][0]["qualification"]["verdict_stability_0.25"] == 0.87


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
    from fraud_detection.core.feature_contract.declaration import declare_columns

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
    from fraud_detection.core.feature_contract.declaration import declare_columns

    declared = declare_columns(
        {"ProductCD": "STRING", "TransactionAmt": "FLOAT64", "n": "INTEGER"},
        excluded=frozenset(),
    )

    assert [d[1] for d in declared.values()] == ["str", "float", "int"]


def test_engineered_columns_are_retrieved_not_requested():
    # A request schema that asked the caller for card_txn_count_24h would move the hard
    # half of the problem outside the system boundary.
    from fraud_detection.core.feature_contract.declaration import declare_columns

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


# ---- the sixth audit's fragment, 2026-08-16 ---------------------------------------

import polars as _pl

from fraud_detection.core.feature_contract import from_segment_qualification


def _scored(rows):
    return _pl.DataFrame(
        [{"column": c, "segment": s, "auc": a} for c, s, a in rows],
        schema={"column": _pl.String, "segment": _pl.String, "auc": _pl.Float64},
    )


def test_a_column_that_collapses_inside_the_segment_is_rejected():
    """card3 scored 0.656 pooled and 0.501 within W — a coin flip on 77% of traffic."""
    fragment = from_segment_qualification(
        _scored([("card3", "__pooled__", 0.656), ("card3", "W", 0.501)]),
        segment="W",
    )

    assert [r.column for r in fragment.rejections] == ["card3"]
    assert fragment.rejections[0].by == "segment_qualification"
    assert fragment.rejections[0].unit == "segment:W"


def test_a_weak_column_that_stays_weak_is_not_this_audits_business():
    """Below the pooled floor the column was never admitted on a pooled score, so a drop
    says nothing about how it got in."""
    fragment = from_segment_qualification(
        _scored([("noise", "__pooled__", 0.52), ("noise", "W", 0.50)]),
        segment="W",
    )

    assert fragment.rejections == ()


def test_a_strong_column_that_merely_dips_survives():
    """W has a seven-times lower base rate, so some decay is the problem being harder."""
    fragment = from_segment_qualification(
        _scored([("D5", "__pooled__", 0.746), ("D5", "W", 0.709)]),
        segment="W",
    )

    assert fragment.rejections == ()


def test_unmeasurable_is_counted_and_not_rejected_by_default():
    """`cannot be scored` is a different finding from `scored badly`, and collapsing them
    is the mistake `reject_degenerate = false` exists to avoid on the drift side."""
    scored = _scored([("product_proxy", "__pooled__", 0.66), ("product_proxy", "W", None)])

    lenient = from_segment_qualification(scored, segment="W")
    strict = from_segment_qualification(scored, segment="W", reject_unmeasurable=True)

    assert lenient.rejections == ()
    assert lenient.qualification["unmeasurable_in_segment"] == 1
    assert [r.column for r in strict.rejections] == ["product_proxy"]


def test_the_policy_section_moves_the_fingerprint():
    """Turning the sixth audit on changes which columns reach the model, so a fingerprint
    that did not move would describe a model trained under different rules."""
    from fraud_detection.core.feature_contract.admission import FeatureAdmissionRules

    off = FeatureAdmissionRules(segment_qualification={"enabled": False})
    on = FeatureAdmissionRules(segment_qualification={"enabled": True, "segment_column": "ProductCD"})

    assert off.as_dict() != on.as_dict()
    assert on.segment_enabled and not off.segment_enabled
