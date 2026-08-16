# Model card — `fraud-lightgbm`

> What this model family is for, what it is not for, where it is structurally weakest, and
> what would have to be true before it decided anything about a person.
>
> **This card carries no run-specific numbers.** Retraining changes them nightly, and a card
> that goes stale is worse than one that never claimed to be current. Where a number is
> needed, §3 says which artefact to read it from.
>
> Architecture: [architecture.md](architecture.md) · Leakage guarantee:
> [point-in-time.md](point-in-time.md) · Numbers worth keeping: [MEASUREMENTS.md](MEASUREMENTS.md)

---

## 1. What it is

| | |
| --- | --- |
| Task | Binary classification — is this card-not-present payment fraudulent? |
| Output | A calibrated probability in [0, 1], plus a raw ranking score |
| Algorithm | LightGBM, gradient-boosted trees, class-weighted |
| Calibration | Chosen per run between Platt and isotonic, on cross-validated log loss subject to a ranking budget |
| Features | Whatever the feature contract admits — see §3 |
| Training data | IEEE-CIS (Vesta) e-commerce transactions, ~590k rows, ~3.5% fraud |
| Deployment | Batch — a Cloud Run Job that scores a period and exits |

**This is a research and demonstration system built on a public competition dataset.** It has never scored a
live payment and nothing here has been reviewed by a compliance function. §6 states what
would have to change first.

## 2. Intended use

**In scope.** Ranking historical or batched card-not-present transactions by fraud
likelihood, so a review queue can be ordered and a block decision taken against a stated
false-positive budget.

| Not for | Why |
| --- | --- |
| Real-time authorisation | The velocity features are windowed aggregates over an entity's prior history and there is no online retrieval path. A live decision would be taken on features that were never retrieved. |
| Any decision about a person rather than a transaction | The model scores an event. It carries no notion of a customer's standing and supports no account closure, credit decision, or report to a third party. |
| Adverse action without human review | The output is an input to a decision, not the decision. See §6. |
| Anything outside e-commerce card-not-present | Card-present, ATM, transfer and BNPL are different problems that happen to share a label name. |
| A different period without re-validation | See §4. |

## 3. Where the current numbers live

Every performance figure is a property of one trained artifact, not of this document.

| Question | Read |
| --- | --- |
| Which model is in production, and under which contract and commit | `gs://<project>-models/promoted/production.json` |
| What that model scored | `metrics.json` beside the artifact — overall, per client segment, per product |
| Which columns it was allowed to see, and why the others were rejected | [`references/feature-contract.json`](../references/feature-contract.json) |
| How large a difference has to be before it means anything | [MEASUREMENTS.md](MEASUREMENTS.md) — the noise band, measured over five seeds |

**Read any performance number against that noise band.** A single training run does not
resolve small differences, and treating one as an improvement is the most common way to
mislead yourself with this pipeline.

### The decision rule, which is a stated policy rather than a fitted value

The threshold is not `argmax F1`, which prices a missed fraud and a declined customer
identically. It comes from a **false-positive budget** — the share of legitimate
transactions the operation will tolerate blocking — set in `config/training.toml`, and is
cross-checked against an explicit cost model:

| Error | Assumed cost |
| --- | --- |
| Missed fraud | The full transaction amount — the money is gone and charged back |
| Blocked legitimate transaction | A flat review-and-goodwill figure, not the sale, which is usually recoverable |

Both are assumptions. They are written down here and in
[`training/threshold.py`](../src/fraud_detection/training/threshold.py) so that disagreeing
with them means disagreeing with a stated number.

**A consequence worth stating plainly:** at a tight false-positive budget this model misses a
large share of fraud, and misses more of it by value than by count — the transactions it
lets through are larger than the ones it catches. The budget is the binding constraint, not
the model.

## 4. Where it is structurally weakest

These do not change between runs, because they are properties of the data and the design.

**Cold entities.** The dataset has no customer identifier, so a client is reconstructed as
`card1 + addr1 + (day − D1)`. Roughly two thirds of test rows sit on a client never seen in
training, and every per-client feature is weaker there. The promotion gate checks this
segment on every run — measured on the *client*, not the card, because nearly every test row
sits on a card that was seen and the per-card version of the check would ask almost nothing.

**Reconstructed identity is an inference.** That key groups transactions that *behave* like
one client; it is not an identity an issuer would recognise, and it has never been validated
against ground truth because the dataset contains none.

**Uneven across products.** `ProductCD` splits the population into groups with very
different fraud rates, and model quality varies sharply between them — worst on the segment
holding most of the traffic. A pooled metric hides this completely, which is why the gate
carries a per-segment check and the contract carries a per-segment audit.

**Anonymised features cannot be explained.** `V*`, `C*`, `D*` and `M*` are anonymised by the
provider, and several of the strongest drivers are among them. An explanation built on them
is mathematically faithful and operationally weak: it can say which column moved the score
and not what that column represents to a customer asking why their payment was declined.
That would be blocking in any deployment subject to an explanation duty.

**Fairness has not been evaluated.** No disparate-impact analysis has been run. The dataset
carries no protected attributes, so the standard measurements cannot be computed directly —
but that is an explanation, not a defence. Two proxies are present and unexamined: email
domain (which correlates with age and country) and coarse address codes. Block rates across
those strata should be measured before any deployment.

## 5. When a model stops being valid

| Trigger | What happens |
| --- | --- |
| The feature contract fingerprint changes | The scoring job compares the model's stamp against the contract on disk and **fails** rather than scoring. |
| A training run newer than the promotion marker | Either the gate rejected it or the gate has not run. Scoring with either model would be a guess, so the job fails. |
| Drift on a scored feature | The PSI logic and the prediction log exist; nothing runs the check on a cadence. Drift would be found by someone looking — see §7. |
| Scoring far past the training window | Untested. The model has never been evaluated further out than the test window. |

## 6. Human oversight

**As built there is none, and that is the honest statement.** The system writes `block` or
`allow` into a log. Nothing consumes that column, no reviewer sees it, and no appeal path
exists — because nothing is in production and no real payment has ever been affected.

Preconditions before this model could take a decision touching a person — not improvements:

1. **A reviewer between the score and the decline.** A meaningful share of blocks are
   legitimate customers at any usable budget; that is tolerable only if a person can
   overturn it quickly.
2. **An appeal path, and a record of its outcomes.** Overturned blocks are the highest-value
   label the system could collect, and it collects nothing.
3. **The fairness measurement in §4**, run and published.
4. **A retention policy** on `inference.prediction_logs`, which does not exist.
5. **A named accountable owner other than the model's author.** Today the person who built
   the gate would be the person approving an override of it, which is not oversight.

## 7. What the automation covers

| Covered | How |
| --- | --- |
| A worse model cannot be promoted | Five gate checks, each failing the run |
| A model cannot be scored against the wrong feature set | Contract fingerprint stamped at fit time, compared at scoring time |
| The scoring path cannot pick an unapproved model | It reads the promotion marker, never the newest artifact |
| Which commit produced an artifact | Git SHA in `metrics.json`, the marker, the registry entry and every scored row |
| No future data reaches a training row | `RANGE … 1 PRECEDING` frames; positional functions banned and asserted absent by tests |

| Not covered | Consequence |
| --- | --- |
| Scheduled drift monitoring | Drift would be found by someone looking, not by an alert |
| Alerting | A failed run is visible in Dagster and nowhere else |
| Automated rollout | Promotion writes a marker; deploying it stays a deliberate `tofu apply` |
| Fairness monitoring | Not measured once, therefore not monitored |

## 8. Data

| | |
| --- | --- |
| Source | [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection), Vesta Corporation |
| Personal data | None directly identifying: email **domain** (not address), coarse address codes, device and browser strings |
| Licence | Kaggle competition terms, research use. Not committed here; fetched per [setup.md](setup.md). |
| Retention | Nothing deletes on a schedule. A deployment on real payment data would need a retention policy on `inference.prediction_logs` before it wrote its first row. |
| Labels | `isFraud` as provided. The label's own latency — a chargeback arrives months after the transaction — is modelled as a deliberate unlabelled gap between train and validation. |
