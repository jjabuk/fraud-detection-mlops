# `references/` — pinned artefacts

Facts the pipeline reads but does not re-derive on every run, each committed so that a
change to one arrives as a reviewable diff rather than as a silent recomputation.

Two kinds live here and the difference matters:

- **Cited** — copied in from an external source, with provenance. Nothing in this repo produces them; they change only when a human decides to re-read the source.
- **Produced** — written by a Dagster asset. They are committed anyway, because the decision they encode is one somebody should approve.

| File                               | Written by                                   | Read by                                                                      | Committed                                                           |
| ---------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------- | --- |
| `column-groups-v.json`             | cited                                        | nobody — hand-transcribed from Deotte's EDA notebook                         | `redundancy_report`, `feature_contract`, `tests/test_redundancy.py` | yes |
| `column-groups-id.json`            | cited                                        | nobody — same source                                                         | **nothing** (see below)                                             | yes |
| `feature-contract.json`            | the `feature_contract` asset                 | `lightgbm_model`, both contract asset checks, and eventually the serving API | yes, once it exists                                                 |

## The contract

`feature-contract.json` is the seam. It is the answer to "which columns is the model
allowed to use", and it has three consumers that must never disagree:

- `contract.training_features()` — the list `lightgbm_model` trains on,
- `contract.request_model()` — the Pydantic schema the serving API will validate against,
  built from the columns marked `source: request`,
- `contract.monitored_columns()` — what the drift monitor iterates.

It is assembled by the `feature_contract` asset in [`assets/feature_audit.py`](../src/fraud_detection/assets/feature_audit.py), which fans in
from the three audit reports, and it is defined by [`feature_contract/core.py`](../src/fraud_detection/feature_contract/core.py). The
`fingerprint` field hashes the admitted set, so a hand-edited file fails to load and a
model can be pinned to the exact contract it was trained against.

### Regenerating it

The contract is produced by materializing the audits. Do this when the data changes, or
after editing [`config/feature-admission.toml`](../config/feature-admission.toml):

```bash
uv run dagster asset materialize -m fraud_detection.orchestration.definitions.feature_platform --select time_consistency_report,distribution_shift_report,redundancy_report,feature_contract
```

That needs GCP credentials and `features.model_input` populated; the time-consistency scan
is ~4 minutes on eight cores. Commit the resulting file — the diff _is_ the decision.

`feature_contract` alone re-assembles from the existing report tables in seconds, which is
what you want after a policy edit that does not change how the reports were computed.


