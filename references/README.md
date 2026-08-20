# Pinned artefacts

Facts the pipeline reads but does not re-derive on every run, committed so that a change
arrives as a reviewable diff rather than as a silent recomputation.

Two kinds live here. **Cited** artefacts are copied in from an external source with
provenance; nothing in this repo produces them, and they change only when someone decides to
re-read the source. **Produced** artefacts are written by code in this repo and committed
anyway, because the decision they encode is one somebody should approve.

| File | Kind | Written by | Read by |
| --- | --- | --- | --- |
| `column-groups-v.json` | cited | hand-transcribed from a published EDA notebook | the [audit repository](https://github.com/jjabuk/ieee-cis-fraud-detection-eda), which keeps its own copy: it describes the dataset rather than this warehouse, so both sides read one without either owning it |
| `column-groups-id.json` | cited | same source | **nothing** |
| `feature-contract.json` | produced | `uv run stamp-contract`, from fragments the [audit repository](https://github.com/jjabuk/ieee-cis-fraud-detection-eda) wrote | `lightgbm_model`, both contract asset checks, the scoring job's fingerprint check, and `features/derivations.py` for the fitted parameters |
| `frequency-maps.json` | superseded | `uv run build-frequency-maps` | `features/derivations.py`, and only when the contract carries none |

`frequency-maps.json` is where the fitted counts lived before the audits became their own
repository. They are fitted there now, on its training split, and travel inside
`feature-contract.json` under the same fingerprint as the verdicts — so a map that moves
invalidates every model pinned to that contract instead of quietly changing what the model
sees. This file is read only when the contract carries no `fitted_parameters`, which keeps
a contract stamped under the previous scheme loadable. `uv run build-frequency-maps` still
works and still needs the warehouse; the audit repository's
`scripts/build-audit-frame.R` does the same fit from the CSVs and needs nothing.

`column-groups-id.json` has no consumer and is kept only as provenance: it is the transcribed
half of a cited source whose other half *is* used and audited. Two things make that weaker
than it sounds, and both are stated here rather than discovered later. The citation that
matters is the URL in [ATTRIBUTION.md](https://github.com/jjabuk/ieee-cis-fraud-detection-eda/blob/main/ATTRIBUTION.md), not the transcription. And the
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

It also carries `derivations` — how each derived column is computed — and
`fitted_parameters`, what the fitted ones learned, both executed by
[`features/derivations.py`](../src/fraud_detection/features/derivations.py).

It is stamped by [`contract/stamp.py`](../src/fraud_detection/contract/stamp.py) from
fragments produced in [`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda), and defined by
[`contract/`](../src/fraud_detection/contract/). The `fingerprint` field hashes the admitted
set, the policy and the fitted parameters, so a hand-edited file fails its integrity check
and a model can be pinned to the exact contract it was trained against.

### Regenerating it

Do this when the data changes, or after editing
[`config/feature-admission.toml`](../config/feature-admission.toml). The verdicts come from
[`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda), cloned beside this repository; re-run the audits there
per its README, then:

```bash
uv run stamp-contract    # reads ../ieee-cis-fraud-detection-eda/out/, writes this directory
```

Commit the result, since the diff is the decision. After a policy edit that does not change
how the verdicts were computed, `stamp-contract` reassembles from the existing fragments in
seconds.

`uv run stamp-contract --check` compares a fresh stamp against the committed file and is a
local check: CI cannot run it, because the fragments are not in this repository. CI verifies
instead that the committed contract's stored fingerprint still matches a hash of its own
contents, which is the failure this side can have on its own.
