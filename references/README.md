# Pinned artefacts

Facts the pipeline reads but does not re-derive on every run, committed so that a change
arrives as a reviewable diff rather than as a silent recomputation.

Two kinds live here. **Cited** artefacts are copied in from an external source with
provenance; nothing in this repo produces them, and they change only when someone decides to
re-read the source. **Produced** artefacts are written by code in this repo and committed
anyway, because the decision they encode is one somebody should approve.

| File | Kind | Written by | Read by |
| --- | --- | --- | --- |
| `column-groups-v.json` | cited | hand-transcribed from a published EDA notebook | the R audits: block-level rejection in `time_consistency`, and `audit_pinned_blocks`, which checks the borrowed partition against this data |
| `column-groups-id.json` | cited | same source | **nothing** |
| `feature-contract.json` | produced | `uv run stamp-contract`, from the audit fragments | `lightgbm_model`, both contract asset checks, the scoring job's fingerprint check |
| `frequency-maps.json` | produced | `uv run build-frequency-maps` | `features/derivations.py` |

`column-groups-id.json` has no consumer and is kept only as provenance: it is the transcribed
half of a cited source whose other half *is* used and audited. Two things make that weaker
than it sounds, and both are stated here rather than discovered later. The citation that
matters is the URL in [ATTRIBUTION.md](../ATTRIBUTION.md), not the transcription. And the
fact it records — which `id_*` columns are labels rather than numbers — is independently
implemented in [`training/data.py`](../src/fraud_detection/training/data.py) and
[`config/feature-admission.toml`](../config/feature-admission.toml), so this is a second
copy that can drift from them with nothing to notice.

Kept for now on the argument that provenance for something you decided *not* to use is
still honest bookkeeping. Delete it the moment that stops being the reason.

## The contract

`feature-contract.json` is the seam. It answers which columns a model may use, and it has
three consumers that must not disagree:

- `contract.training_features()`, the list `lightgbm_model` trains on,
- `contract.request_model()`, the Pydantic schema the serving API validates against, built
  from the columns marked `source: request`,
- `contract.monitored_columns()`, the set a drift monitor would iterate.

It is stamped by [`contract/stamp.py`](../src/fraud_detection/contract/stamp.py) from the
fragments the R audits write, and defined by
[`contract/`](../src/fraud_detection/contract/). The `fingerprint` field hashes the admitted
set and the policy, so a hand-edited file fails its integrity check and a model can be pinned
to the exact contract it was trained against.

### Regenerating it

Do this when the data changes, or after editing
[`config/feature-admission.toml`](../config/feature-admission.toml):

```bash
uv run dagster asset materialize \
  -m fraud_detection.orchestration.definitions.feature_platform \
  --select "fraud_detection/model_input+"     # rebuild, then re-export the audit frame

cd analysis && Rscript -e 'targets::tar_make()' && quarto render && cd ..
uv run stamp-contract
```

The first step needs GCP credentials and a populated `features.model_input`; the audits
themselves need neither, only the exported parquet. Commit the result, since the diff is the
decision.

After a policy edit that does not change how the audits were computed, `uv run stamp-contract`
alone reassembles from the existing fragments in seconds. `uv run stamp-contract --check` is
what CI runs: it stamps into memory and compares fingerprints, so a contract that was never
re-stamped fails the build.
