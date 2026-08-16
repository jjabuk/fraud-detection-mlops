"""Two code locations, one seam.

Deliberately empty. Importing this package must not load either asset graph — Dagster
loads each location's module directly (see `dagster/workspace.yaml`), and a re-export here
would put Dagster and the cloud SDK back into the import path of anything that touches the
package.

* `feature_platform` — raw data in, admitted features out. Owns `features.model_input` and
  `references/feature-contract.json`.
* `model_factory` — those two artefacts in, a validated model out.

The boundary between them is documented in `docs/code-structure.md`.
"""
