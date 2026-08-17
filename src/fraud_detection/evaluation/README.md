# Feature audits

The published solutions to the IEEE-CIS competition describe checks their authors ran by hand,
once. Each one here is implemented as tested code whose verdict becomes a decision the training
pipeline and the serving schema read from one place.

Nothing in this package imports Dagster or a cloud SDK, so every audit is callable from a
notebook against the same implementation the pipeline runs.

| Audit | Question it answers | Verdict feeds |
| --- | --- | --- |
| [`time_consistency.py`](time_consistency.py) | Does a feature still rank the same way months later? | contract fragment |
| [`distribution_shift.py`](distribution_shift.py) | Did a column move, and can a model tell the periods apart? | contract fragment, drift logic |
| [`redundancy.py`](redundancy.py) | Which columns carry the same signal as their neighbours? | contract fragment |
| [`segment_qualification.py`](segment_qualification.py) | Does a column that scores well pooled still score inside the dominant segment? | contract fragment, report-only |
| [`entity_purity.py`](entity_purity.py) | Who is the customer the data does not name, and is the reconstruction real? | validation split, gate segmentation |
| [`selection.py`](selection.py) | What survives PCA, and what does a greedy search keep? | measurement only |

The first four write fragments that
[`assets/feature_audit.py`](../orchestration/assets/feature_audit.py) merges into
[`references/feature-contract.json`](../../../references/feature-contract.json).
`segment_qualification` records its verdicts without applying them, for the reason in
[MEASUREMENTS.md](../../../docs/MEASUREMENTS.md): applying them cost 0.0325 PR-AUC and moved
the segment it was protecting by 0.0005.

## Audit, not gate

None of these runs on every training job. They run when the data changes, and their output is a
committed artefact; the checks downstream are cheap assertions against that artefact.

The reason is not cost. Time consistency over 377 columns takes 4.5 minutes on eight cores,
which no pipeline would notice. It is that the answer is a property of the data, so
re-deriving it per run recomputes a constant and gives the run one more way to fail.

## Deliberately not implemented

| Technique | Why not |
| --- | --- |
| Recursive feature elimination | The same claim as forward selection at higher cost. One implementation of "does this column pay for itself on one holdout" is enough. |
| Permutation importance | Ranks features, does not reject them. Interpretation output rather than an admission check. |
| Ensembling (CAT + LGB + XGB), neural nets | Three times the serving cost for a leaderboard delta, on a leaderboard this project is not entering. |
| Per-column descriptive profiling | A report a human reads once. Nothing downstream consumes it, which is what makes it EDA rather than a pipeline asset. |

Forward selection and PCA were on this list until they were measured; `selection.py` is the
result, and it stays measurement-only because nothing downstream consumes its verdicts.

## Provenance

The V-block partition in
[`references/column-groups-v.json`](../../../references/column-groups-v.json) is human judgement
taken from a published notebook, pinned rather than re-derived, with attribution in the file.
Everything downstream of it, including representative selection and the audit of whether the
partition holds, is this repository's own.
