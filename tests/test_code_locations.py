"""The seam between the two code locations, checked rather than described.

    feature_platform  --[ model_input, feature_contract ]-->  model_factory

Two properties make that a boundary rather than a diagram:

1. The crossing is **narrow and enumerated**. If a third artefact starts crossing, this
   test fails and the boundary moves as a decision instead of by accident.
2. Neither location imports the other. That is what would let the model factory move to
   its own repository, its own image, or its own deploy cadence without a code change —
   the claim `docs/code-structure.md` makes, made falsifiable.
"""

from __future__ import annotations

import ast
import pathlib

from fraud_detection.orchestration.definitions.feature_platform import defs as feature_platform
from fraud_detection.orchestration.definitions.model_factory import defs as model_factory

LOCATIONS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "fraud_detection"
    / "orchestration"
    / "definitions"
)

# Everything the model factory consumes and does not build. The whole seam, in one list.
SEAM = {"fraud_detection/model_input", "fraud_detection/feature_contract"}


def keys(defs) -> set[str]:
    return {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}


def executable_keys(defs) -> set[str]:
    graph = defs.resolve_asset_graph()
    return {k.to_user_string() for k in graph.get_all_asset_keys() if graph.get(k).is_executable}


def test_the_seam_is_exactly_two_artefacts():
    crossing = keys(model_factory) - executable_keys(model_factory)

    assert crossing == SEAM


def test_the_platform_actually_builds_what_the_factory_declares():
    # An external asset whose key matches nothing upstream is a dangling reference: the
    # lineage graph would show a source with no producer, and a typo in a key would look
    # exactly like a boundary.
    assert SEAM <= executable_keys(feature_platform)


def test_the_factory_builds_nothing_the_platform_builds():
    # Overlap would mean two locations racing to write one artefact.
    assert executable_keys(model_factory) & executable_keys(feature_platform) == set()


def test_neither_location_imports_the_other():
    paths = [LOCATIONS / f"{stem}.py" for stem in ("feature_platform", "model_factory")]
    # Guard against the loop silently emptying if the modules move again: this test passed
    # for as long as LOCATIONS pointed at a directory that no longer existed.
    assert all(p.is_file() for p in paths), f"code location modules not found under {LOCATIONS}"

    for path in paths:
        other = "model_factory" if path.stem == "feature_platform" else "feature_platform"
        for node in ast.walk(ast.parse(path.read_text())):
            module = getattr(node, "module", None) or ""
            names = [a.name for a in getattr(node, "names", [])]
            assert other not in module, f"{path.name} imports {other}"
            assert all(other not in n for n in names), f"{path.name} imports {other}"


def test_the_contract_gate_lives_with_the_model_that_it_gates():
    # The check compares a trained model against the contract, so it belongs to whoever
    # trains. Putting it in the platform would mean the platform importing the model.
    check_names = {
        key.name for check in model_factory.asset_checks or [] for key in check.check_keys
    }

    assert "model_features_admitted_check" in check_names


# ---- the process is visible in the graph ---------------------------------------

# Each location's groups, in the order the work happens. The Dagster UI groups assets by
# these, so the graph reads as the process rather than as a pile of tables.
FEATURE_PLATFORM_GROUPS = {
    "raw_ingestion",
    "feature_store",
    "feature_validation",
}

MODEL_FACTORY_GROUPS = {
    "dataset_preparation", 
    "model_training", 
    "model_registry",
}


def groups_of(defs) -> set[str]:
    graph = defs.resolve_asset_graph()
    return {
        graph.get(key).group_name
        for key in graph.get_all_asset_keys()
        if graph.get(key).is_executable
    }


def test_the_feature_platform_reads_as_a_process():
    assert groups_of(feature_platform) == FEATURE_PLATFORM_GROUPS


def test_the_model_factory_reads_as_a_process():
    assert groups_of(model_factory) == MODEL_FACTORY_GROUPS


def test_the_two_locations_share_no_group():
    # A group spanning the seam would make the boundary invisible in the one place people
    # actually look at it.
    assert FEATURE_PLATFORM_GROUPS & MODEL_FACTORY_GROUPS == set()
