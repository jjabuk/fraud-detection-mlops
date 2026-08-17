# Pinned artefacts

Facts the pipeline reads but does not re-derive on every run, committed so that a change
arrives as a reviewable diff rather than as a silent recomputation.

Two kinds live here. **Cited** artefacts are copied in from an external source with
provenance; nothing in this repo produces them, and they change only when someone decides to
re-read the source. **Produced** artefacts are written by code in this repo and committed
anyway, because the decision they encode is one somebody should approve.

| File | Kind | Written by | Read by |
| --- | --- | --- | --- |
| `column-groups-v.json` | cited | hand-transcribed from a published EDA notebook | `redundancy_report` via `config/orchestration.toml`, `tests/test_redundancy.py` |
| `column-groups-id.json` | cited | same source | nothing currently |
| `feature-contract.json` | produced | the `feature_contract` asset | `lightgbm_model`, both contract asset checks, the scoring job's fingerprint check |
| `frequency-maps.json` | produced | `uv run build-frequency-maps` | `feature_engineering/derivations.py` |

`column-groups-id.json` is kept because the identity-block partition is part of the same cited
source and dropping half of it would make the citation harder to check than keeping it.

## The contract

`feature-contract.json` is the seam. It answers which columns a model may use, and it has
three consumers that must not disagree:

- `contract.training_features()`, the list `lightgbm_model` trains on,
- `contract.request_model()`, the Pydantic schema the serving API validates against, built
  from the columns marked `source: request`,
- `contract.monitored_columns()`, what the drift logic iterates.

It is assembled by the `feature_contract` asset in
[`assets/feature_audit.py`](../src/fraud_detection/orchestration/assets/feature_audit.py),
which fans in from the audit reports, and defined by
[`core/feature_contract/`](../src/fraud_detection/core/feature_contract/). The `fingerprint`
field hashes the admitted set, so a hand-edited file fails its integrity check and a model can
be pinned to the exact contract it was trained against.

### Regenerating it

Do this when the data changes, or after editing
[`config/feature-admission.toml`](../config/feature-admission.toml):

```bash
uv run dagster asset materialize \
  -m fraud_detection.orchestration.definitions.feature_platform \
  --select time_consistency_report,distribution_shift_report,redundancy_report,feature_contract
```

That needs GCP credentials and a populated `features.model_input`; the time-consistency scan
takes about four minutes on eight cores. Commit the result, since the diff is the decision.

After a policy edit that does not change how the reports were computed, materializing
`feature_contract` alone reassembles from the existing report tables in seconds.
