"""Catalog affordances: the labels that make the asset graph readable in the UI.

None of this changes what runs. It changes what somebody browsing the graph can tell at a
glance — which engine an asset uses, which code location owns it, and whether it is stale
with respect to the code that produced it.

`CODE_VERSION` is the repository's git SHA rather than a per-asset hash. That is the right
granularity *here* specifically because `config/*.toml` is read at import time by nearly
every asset: a commit that only edits a threshold really can change what any of them
produces. In a repository where assets had independent inputs it would over-invalidate, and
a per-asset version would be worth the bookkeeping.
"""

from __future__ import annotations

from fraud_detection.core.provenance import code_version

#: Marks assets stale in the graph when the code that produces them moves.
CODE_VERSION = code_version().short

# Dagster renders a known `kind` as an icon. These are the four things this pipeline
# actually computes with.
BIGQUERY = {"bigquery"}
LIGHTGBM = {"lightgbm"}
GCS = {"gcs"}
VERTEX = {"vertexai"}

# Owners are the code location responsible, not a person: this is a single-operator project
# and naming an individual on thirty assets would be noise. A team handle is what the field
# is for, and the locations are the real boundary of responsibility.
FEATURE_PLATFORM = ["team:feature-platform"]
MODEL_FACTORY = ["team:model-factory"]
INFERENCE = ["team:inference"]
