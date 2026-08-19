"""The rule that keeps the code usable outside Dagster.

The pure layer is what a notebook imports and what the serving path will reuse. The moment
something in it reaches back into `assets`, importing it drags in Dagster and a cloud SDK,
and the "experiment locally, run in the cloud" split stops being true.

These tests read the import graph directly, so the rule is checked rather than described.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "fraud_detection"

PURE = ("config.py", "schema.py", "contract", "features", "training", "registry")
ORCHESTRATION = {"orchestration", "resources", "tools"}
FORBIDDEN_IN_PURE = {"dagster", "google"}


def modules(*roots: str) -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for root in roots:
        target = SRC / root
        found += [target] if target.is_file() else sorted(target.rglob("*.py"))
    return [p for p in found if "__pycache__" not in str(p)]


def imports_of(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """(top-level external packages, first-class internal modules)."""
    external, internal = set(), set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            external |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "fraud_detection":
                internal.add(parts[1] if len(parts) > 1 else "")
            else:
                external.add(parts[0])
    return external, internal


PURE_MODULES = modules(*PURE)


def label(path: pathlib.Path) -> str:
    return str(path.relative_to(SRC))


@pytest.mark.parametrize("module", PURE_MODULES, ids=label)
def test_pure_modules_do_not_import_the_orchestrator(module):
    external, _ = imports_of(module)
    assert not (external & FORBIDDEN_IN_PURE), (
        f"{label(module)} imports {sorted(external & FORBIDDEN_IN_PURE)}. "
        "Anything a notebook imports must not pull in Dagster or a cloud SDK."
    )


@pytest.mark.parametrize("module", PURE_MODULES, ids=label)
def test_pure_modules_do_not_import_from_assets(module):
    _, internal = imports_of(module)
    assert not (internal & ORCHESTRATION), (
        f"{label(module)} imports from {sorted(internal & ORCHESTRATION)}. "
        "The dependency runs the other way: assets import the pure layer, never the "
        "reverse. Move the shared name into fraud_detection/schema.py."
    )


def test_the_package_root_stays_empty():
    """`import fraud_detection.evaluation.x` must not load the whole Dagster graph.

    It used to: the root `__init__` imported `definitions`, which cost 3 seconds and 3,500
    modules before a notebook could call a pandas function.
    """
    external, internal = imports_of(SRC / "__init__.py")
    assert not external and not internal


def test_pure_layer_covers_what_a_notebook_needs():
    """A guard against the layers being satisfied by being empty.

    The list shrank when the audits moved to `analysis/`. What the rule protects is
    unchanged and is now narrower and sharper: the entity key, the modelling recipe
    and the contract are the three things an analysis has to be able to call
    against the same implementation the pipeline runs.
    """
    names = {label(m) for m in PURE_MODULES}
    for expected in (
        "schema.py",
        "features/entity.py",
        "training/model.py",
        "contract/core.py",
    ):
        assert expected in names, f"{expected} missing from the pure layer"


def test_assets_are_allowed_to_import_the_pure_layer():
    """The rule is one-directional, not a ban on coupling."""
    reaching_in = {
        label(m) for m in modules("orchestration/assets") if imports_of(m)[1] & {"training", "evaluation", "core", "feature_engineering"}
    }
    assert reaching_in, "no asset uses the pure layer; the split would be decoration"


def test_tools_may_reach_outside_and_the_pure_layer_may_not():
    """The criterion that puts a module in `tools/` rather than in its concern.

    A command that opens a BigQuery client is not pure-layer material however
    much it is *about* features or training: importing it would drag a cloud SDK
    into every notebook that touched the package it lived in. `tools/` is where
    those go, and the direction is one-way.
    """
    reaching = {
        label(m) for m in modules("tools") if imports_of(m)[0] & FORBIDDEN_IN_PURE
    }
    assert reaching, "tools/ holds nothing that reaches outside — the split earns nothing"

    for module in PURE_MODULES:
        _, internal = imports_of(module)
        assert "tools" not in internal, (
            f"{label(module)} imports from tools/. The dependency runs the other way: "
            "tools compose the pure layer, never the reverse."
        )
