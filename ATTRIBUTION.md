# Attribution

Where the ideas in this repository came from, what was taken, and what was written here.

The IEEE-CIS Fraud Detection competition produced a large body of public analysis, and this
project builds on several pieces of it. This file exists so that "built on" can be checked
rather than assumed: every borrowed idea is listed with its source, what specifically was
used, and what this repository does differently.

**No source code was copied.** Every implementation here is original. What was taken is
*findings* — a normalisation formula, a grouping of columns, the observation that a client
can be reconstructed — and in each case the implementation, the correctness constraints and
the evaluation are this repository's.

---

## Chris Deotte

### "EDA for Columns V and ID"
<https://www.kaggle.com/code/cdeotte/eda-for-columns-v-and-id>

**Taken:** the partition of the anonymised `V1`–`V339` block into correlated sub-groups
within each missingness family, and the observation that some `id_*` columns carry a small
label set rather than a number.

**How:** hand-transcribed into [`references/column-groups-v.json`](references/column-groups-v.json)
and [`references/column-groups-id.json`](references/column-groups-id.json), each carrying a
`provenance` block naming the author, the work, the URL and the retrieval date. The
groupings are **human judgement read off correlation heatmaps and are taken as given, not
re-derived.**

**What is this repository's:** everything downstream. Selecting a representative per group
([`evaluation/redundancy.py`](src/fraud_detection/evaluation/redundancy.py)), and — more to
the point — `audit_partition`, which asks whether each group is more tightly bound inside
than to its neighbours. That audit scores the borrowed partition at **0.78 over 93
measurable groups**, and the number is recorded in the feature contract. Using someone
else's grouping without checking whether it holds on your data would be the borrowing this
file is meant to prevent.

### "XGB Fraud with Magic — [0.9600]"
<https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600>

**Taken:** the central finding of the competition — that a client can be reconstructed from
`card1 + addr1 + D1n`, and that aggregates over that reconstructed client are the strongest
features available.

**What is this repository's, and it is the substantive difference:** the published solution
computes those aggregates as **full-group statistics over train ∪ test**, which is
legitimate in a competition where the whole test set is handed over at once. This pipeline
computes them under `RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`, so a row sees its
entity's past and never its future — see [docs/point-in-time.md](docs/point-in-time.md).
That is a different feature with the same name, and the difference is most of the gap
between this project's score and the published one.

Implementation: [`feature_engineering/features.py`](src/fraud_detection/feature_engineering/features.py),
`build_uid_aggregate_sql`.

## Konstantin Yakovlev

### "IEEE — uid detection"
<https://www.kaggle.com/code/kyakovlev/ieee-uid-detection>

**Taken:** the D-normalisation, `Dxn = floor(TransactionDT / 86400) − Dx`, applied to
`D1, D2, D3, D5, D10, D11, D15`. Each `D` column counts days since something began for a
card; subtracting it from the transaction's own day recovers the day it began, which is
constant across that client's history and therefore usable as an identity component.

**What is this repository's:** the derivations are *declared in the feature contract* rather
than computed inline, so each of the seven is judged by the same audits as any raw column —
which is not something the source notebook needed. All seven were then **rejected**: six by
the drift audit and one by time consistency. `Dxn` is a calendar day number, so an early
window and a late window are close to disjoint by construction, and a model trained on it
learns "clients who first appeared in months 1–3 look like this".

That is not a disagreement with the source. Yakovlev used `Dxn` as a *component of the uid*,
where being calendar-anchored is the point, not as a model feature. This repository uses it
the same way and additionally measured what happens if you do the other thing.

Implementation: [`feature_engineering/derivations.py`](src/fraud_detection/feature_engineering/derivations.py),
`days_since_to_start_day`.

### "IEEE — Basic FE Part 1"
<https://www.kaggle.com/code/kyakovlev/ieee-basic-fe-part-1>

**Taken:** frequency encoding as the right tool for high-cardinality categoricals — replace
a category with how often it occurs.

**What is this repository's:**

- the counts are **fitted on the training split only** and committed to
  [`references/frequency-maps.json`](references/frequency-maps.json). The published kernels
  count over train and test together; that is transductive and unavailable to a system
  scoring a period nobody has seen;
- the columns were chosen **on measured evidence** rather than on cardinality. Rarity has to
  predict fraud for the encoding to be worth anything, and it does not for every
  high-cardinality column — `card1` and `card2` were declared and then removed when their
  fraud-rate-by-frequency profile turned out to be flat;
- an unseen value encodes as **null, not zero**, because zero would assert the value occurs
  zero times, which is false in the row being scored.

Implementation: [`feature_engineering/derivations.py`](src/fraud_detection/feature_engineering/derivations.py),
`frequency_encode`, and [`cli/build_frequency_maps.py`](src/fraud_detection/cli/build_frequency_maps.py).

## FraudSquad — 1st place solution

<https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111308>

**Taken:** context and confirmation, not implementation. The write-up establishes that the
competition's decisive move was client reconstruction plus aggregation, which is the finding
this project chose to reimplement causally. Read to understand what the ceiling is made of;
no code or artefact derives from it.

## Competition organisers — Vesta Corporation

<https://www.kaggle.com/competitions/ieee-fraud-detection>

The dataset, its column semantics (`C*` counting, `D*` timedeltas, `M*` match flags, `V*`
Vesta-engineered), and the label definition. Discussion posts by the organisers are the
source for what the anonymised blocks mean; the data itself is **not committed to this
repository** and must be fetched under the competition's own terms.

---

## Standard techniques, used without attribution to any individual

These are textbook or widely-published methods with no single owner, implemented from the
definition rather than from anyone's code: population stability index, adversarial
validation, isotonic and Platt calibration, precision-recall and ROC analysis, stratified
cross-validation, SHAP.

## What is original here

Listed because a reader deserves to know which parts are the contribution rather than the
inheritance:

- the **feature contract** — audits producing fragments that merge into one committed,
  fingerprinted artefact, with the fingerprint checked at scoring time;
- the **six audits** and, in particular, that each carries its own reproducibility score, so
  a check that does not reproduce is visible as such;
- **point-in-time enforcement** as a tested property rather than an intention;
- the **promotion gate** and its five checks;
- the **noise-band measurement** and the discipline built on it;
- the orchestration, infrastructure, serving surface and supply-chain work in full.

## Licence

This repository is MIT-licensed (see [LICENSE](LICENSE)). The cited notebooks are published
under Kaggle's own terms and are linked rather than reproduced. The dataset is subject to
the competition rules and is not redistributed here.
