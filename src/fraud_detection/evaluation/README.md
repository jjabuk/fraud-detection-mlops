# Evaluation

The published solutions to the IEEE-CIS competition describe a set of checks their authors
ran by hand, once. Each one here is implemented as code with tests, and each produces a
`Fragment` for [the feature contract](../feature_contract/) — so an
audit's result becomes a decision the training pipeline and the serving schema both read
from one place.

Code: [this directory](./)

| Technique | Question it answers | Verdict feeds |
| --- | --- | --- |
| [time-consistency](docs/time-consistency.md) | Does a feature still rank the same way months later? | contract |
| [distribution-shift](docs/distribution-shift.md) | Did a column move, and can a model tell the periods apart? | contract + drift monitor |
| [entity-purity](docs/entity-purity.md) | Who is the customer the data does not name, and is the reconstruction real? | validation split + gate segmentation |
| [redundancy](docs/redundancy.md) | Which columns carry the same signal as their neighbours? | contract |
| [selection](docs/selection.md) | What survives PCA, and what does a greedy search actually keep? | contract |

## Audit, not gate

None of these runs on every training job. They run when the **data** changes, and their
output is a committed artefact. The gate downstream is a set of cheap assertions against
that artefact — see [ROADMAP](../../../docs/ROADMAP.md).

The distinction is not about cost. Time consistency over 377 columns takes 4.5 minutes on
eight cores, which no pipeline would notice. It is that the answer is a property of the
data, so re-deriving it per run recomputes a constant and makes the run less reliable for
nothing.

## What is deliberately not implemented

| Technique | Why not |
| --- | --- |
| Recursive feature elimination | Same claim as forward selection, at higher cost. One implementation of "does this column pay for itself on one holdout" is enough. |
| Permutation importance | Ranks features; does not reject them. Interpretation output, not an admission check. |
| Ensembling (CAT + LGB + XGB), neural nets | Three times the serving cost for a leaderboard delta, on a leaderboard this project is not entering. |
| Per-column descriptive profiling | A report a human reads once. Nothing downstream consumes it, which is what makes it EDA rather than a pipeline asset. |

Two of these — forward selection and PCA — were on this list until they were measured. See
[selection.md](docs/selection.md) for what changed and what the measurements are worth.

## Provenance

The V-block partition in [`references/column-groups-v.json`](../../../references/column-groups-v.json)
is human judgement taken from a published notebook, pinned rather than re-derived, with
attribution in the file. Everything downstream of it — representative selection, the audit
of whether the partition holds, and every measurement in these documents — is this
repository's own. See [redundancy.md](docs/redundancy.md).
