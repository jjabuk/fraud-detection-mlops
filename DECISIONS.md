# Decision Log

Architectural and mathematical "why"s, one entry per decision, newest first. Written at
the moment the decision is made — not reconstructed later.

Target architecture: [docs/architecture.md](docs/architecture.md) ·
Numbers: [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) ·
Where borrowed ideas came from: [ATTRIBUTION.md](https://github.com/jjabuk/ieee-cis-fraud-detection-eda/blob/main/ATTRIBUTION.md)

Entries are kept even when superseded. A decision that was later reversed is more
informative than one that was never written down, and the reversal is recorded in place.

---

## 2026-08-19 — The entity key is a column, and the split it drives was never reproducible

**Decision.** `features.model_input` now carries `client_uid`, the identifier the
feature-engineering statement already computes. Python reads that column instead of
rebuilding the key from `card1`, `addr1` and `D1`. `Anchor` and `EntityKey` are deleted;
`features/entity.py` is three set operations.

**Why it was duplicated.** `CLIENT_UID_EXPRESSION` builds the identifier in SQL, uses it to
window every client aggregate, and the final `SELECT` emitted it into
`features.transaction_features` — but `MODEL_INPUT_SQL` selects `j.*` plus an explicit
feature list, and the key is not a feature, so it fell off the end. Python then rebuilt it
in polars because training and the gate both need it. Two implementations of one
identifier: a BigQuery `FORMAT` with `_` separators and null-guarded on two columns, and a
polars `concat_str` with `|` separators and null-guarded on all parts. They agreed. Nothing
made them agree.

Carrying the column was always the smaller change. It is one line in the statement, and
`schema.EXCLUDED_COLUMNS` already listed `client_uid` so no audit could mistake it for a
feature.

**What this exposed.** Writing the test for the new, smaller `entity_split` found that it
had never been deterministic:

```
same seed, four calls, identical splits: False
```

`polars.Series.unique()` is hash-based and does not promise an order between calls, so
permuting its result under a fixed seed produced **a different train/holdout split on every
run**. Every model this repository has trained was fitted on a different split from the one
the previous run measured, and nothing showed it: the metrics move by less than seed noise,
which is exactly the band the promotion gate treats as "unchanged".

`unique(maintain_order = TRUE)` would have fixed the symptom and inherited a worse one — the
order would then depend on the frame's row order, which a multi-threaded scan does not
promise either. The entities are sorted, so the split depends on the *set* of entities and
the seed and on nothing else.

This is the third instance of the same failure in one week, after the Arrow row ordering and
the redundancy representative. All three were seeded, none were reproducible, and none were
visible in any output a person reads. The pattern is worth naming: **a seed makes a
computation repeatable only if everything upstream of it has a defined order.** Hash-based
uniqueness, multi-threaded scans and dictionary iteration all break that quietly.

**What did not move.** The question of whether this is the *right* key is still measured in
the audit repository, against a permuted null. What is new is that the analysis now checks its
recommendation against the column the pipeline builds, comparing the two as partitions
rather than as strings — so the statement and the analysis can no longer describe different
entities without saying so.

---

## 2026-08-19 — The audits move to R, and out of this repository

**Decision.** The audits that decide which columns reach the model are rewritten as
statistics and moved to a separate repository,
[`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda). `evaluation/*.py` is deleted — about 1,340 lines.
The contract is now produced by `uv run stamp-contract` from the fragments that repository
writes. The reasoning, the validation against the implementation it replaced, and what the
new statistics found are in
[its decision log](https://github.com/jjabuk/ieee-cis-fraud-detection-eda/blob/main/DECISIONS.md).

**What this side kept.** The entity key, because it is a transformation the training job
runs and the gate's cold-entity segment depends on, not a question about the data. The
fingerprint, because `FeatureContract.from_dict` refuses a file whose stored hash disagrees
with its contents, and that detector should not depend on two JSON serialisers agreeing
forever. One writer, one hash.

**What this side lost.** `stamp-contract --check` was a CI step: it re-stamped from the
fragments on disk and failed a pull request that changed an audit without re-stamping. The
fragments are no longer in this repository, so the check is now local only. CI verifies
what it still can — that the committed contract's stored fingerprint matches a hash of its
own contents, which catches a hand-edited or truncated file.

---

## 2026-08-16 — The uid override is retired, on the measurement that was taken to justify it

**Decision.** The nineteen `client_*` aggregates readmitted by policy override are removed.
The contract goes from 224 admitted columns to 205 and admits nothing the audits rejected.

**Why it existed.** `time_consistency` rejected all nineteen. The argument for overruling it
was that a lifetime aggregate's single-feature AUC compared across an early and a late window
measures the aggregate changing meaning as the entity ages, not a signal inverting — and that
`time_consistency` reproduces at 0.32 where `distribution_shift`, which did not reject them,
reproduces at 1.00.

**Why it goes.** The argument still looks correct and it is not worth nineteen columns.
Seed-averaged over five seeds at equal footing: **0.8954 ROC-AUC with the override, 0.8951
without** — a difference of 0.0002, a tenth of the resolution of a five-seed mean. The
retrained model confirmed it: 0.5210 PR-AUC against 0.5229, inside the noise.

**What this is really a decision about.** An override is where a human overrules an audit,
and the only thing that makes that legitimate is being willing to withdraw it. This one was
applied on an argument, measured, and withdrawn on the measurement. The reasoning is kept as
a comment in `config/feature-admission.toml` rather than deleted, because a record of an
override that was tried and did not pay is worth more than a clean file.


## 2026-08-16 — The sixth audit is kept and its verdict discarded

**Decision.** `segment_qualification` — does a column carry signal _within_ the dominant
segment, or only between segments — runs in **report-only** mode. It records what it would
reject and rejects nothing.

**Why it was built.** `ProductCD` splits the population into groups with very different
fraud rates, and columns exist that look predictive pooled and are constants inside the
segment carrying most of the traffic — a coin flip on three quarters of the volume, credited
as signal by every pooled check. The per-column figures are in
[`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda).

**Why the verdict was discarded the same day.** Applying its 26 rejections cost **0.0325
PR-AUC** globally — about 3.5 sd — and moved `W` from 0.2130 to **0.2125**. The asset check
refused the model on the existing floor.

The premise was wrong. Between-segment signal is spurious for _within-group inference_ and
genuinely predictive here, because the segment predicts the outcome: "this is a C
transaction" is real information about risk. Simpson's paradox is a paradox only when you
wanted the within-group answer.

**What survives.** The measurement. The fragment still records what it would have rejected
and how much of the table is unmeasurable inside the dominant segment, which is a real
coverage finding. Discarding the verdict did not require discarding the evidence.

## 2026-08-16 — One-hot encoding introduced against recommendation, in order to measure it

**Decision.** One-hot indicators are declared for the five low-cardinality categoricals,
one indicator per level, with levels pinned in config.

**Why, given the recommendation was against it.** `training/data.py` had asserted for months
that one-hot is the wrong tool here — LightGBM splits on categoricals natively, which is
strictly more expressive than a per-level indicator. That was an assertion in a comment, and
this project's standard is that assertions get measured. Declaring them puts the question
through the same audits as any other column.

Frequency encoding was introduced alongside, on the opposite reasoning: it is the only
option where one-hot is hopeless (`DeviceInfo` has 1,786 levels), and the columns were
chosen on measured evidence — rarity has to predict fraud, and it does not for `card1`
(0.85 / 0.76 / 1.19 / 0.98 times base rate across frequency bands) or `card2`.

## 2026-08-16 — Point-in-time correctness is kept, at a cost that is still unmeasured

**Decision.** The uid aggregates stay under `RANGE … 1 PRECEDING`. The transductive
formulation — full-group statistics over train ∪ test, as the published solutions compute
them — is not adopted, and not implemented as an option.

**Why.** It is the property the whole project exists to demonstrate. A pipeline that
abandons it to chase a leaderboard has nothing left to say.

**What this costs is recorded as unknown.** The gap to the published solutions is real but
has never been attributed: measuring it needs the transductive variant computed and
compared, which has not been done. The README states it as an open question rather than as
a result.

## 2026-08-16 — The noise band is measured before results are reported

**Decision.** `uv run noise-band` refits the chosen configuration five times with nothing
moving but the LightGBM seed. **sd 0.0029 ROC-AUC, sd 0.0065 PR-AUC.** A difference between
two single fits therefore carries sd 0.0041 and 0.0092.

**Consequence, adopted as a rule.** A single fit resolves nothing below ~0.004 ROC-AUC and a
single submission nothing below ~0.005. Differences under that are recorded as
**unmeasured**, never as _no effect_.

**What it retracted immediately.** Three conclusions, including one recorded the same
evening: a feature change credited with "+0.0032 ROC-AUC" was 0.78 sd, and its headline
number turned out to be the maximum of five draws. The dominant source is early stopping,
not the model — `best_iteration` ranged over 431–1581 across the five seeds.

## 2026-08-16 — The BQML baseline is pinned, not recomputed before every gate

**Decision.** `validation_gate` reads `baseline_pr_auc` from config instead of depending on
the `bqml_baseline` asset.

**Why.** It is one `CREATE MODEL` statement over fixed columns on a fixed split and returned
0.0635 on every run it was asked for. Rebuilding it before each gate spent ~90 seconds of
BigQuery re-deriving a constant. The number is still what "above baseline" means, and a
change to it is now a reviewable diff. Re-derive with
`dagster asset materialize --select fraud_detection/bqml_baseline` when the split moves.

## 2026-08-16 — Documentation carries no numbers that a rerun invalidates

**Decision.** The model card describes the model _family_ and its structural properties, and
carries no run-specific figures. README states the thesis and the resolution limit, not the
current score. Both point at the promotion marker and `metrics.json` for live numbers.

**Why.** The card had already gone stale — it described 184 features under a contract that
had been superseded twice. A document that claims to be current and is not is worse than one
that never claimed it.

## 2026-08-11 — Client entity reconstructed; it is correct, and it barely helps

**Decision.** A client id is reconstructed as `card1 + addr1 + (day − D1)` and five
point-in-time aggregates are computed over it. It is kept, but recorded as a **negative
result**: the marginal gain is +0.0012 test PR-AUC, and none of the five features reach the
top 25 by SHAP.

**Why it was attempted.** Vesta's labelling rule propagates a chargeback forward across
every later transaction linked by account, email or billing address, so the client — not
the card, not the transaction — is the natural unit of this problem. The competition was
won on exactly this lever.

**The reconstruction is real, and that is measurable.** `D1` is days since the client's
first transaction, so `day − D1` is their account-start day. It groups the table far more
purely by label than `card1` or `card1 + addr1` do, which is the right test precisely
because fraud propagates within a client. The measurement, and the null it is judged
against, are in [`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda).

**And it still bought almost nothing.**

|                                    | Test PR-AUC | Uncalibrated |
| ---------------------------------- | ----------- | ------------ |
| 7 velocity features (card, device) | 0.5017      | 0.5144       |
| + 5 client features                | **0.5029**  | 0.5177       |

Reproducible rather than noise — LightGBM was made deterministic first, specifically so a
gain this small could be distinguished from thread scheduling. Two reasons it is small:

1. **At scoring time there is usually no client history.** 33.7% of transactions are that
   client's first, and a further 11.3% have no reconstructable id. For **45% of rows** the
   client features are structurally empty.
2. **Vesta already shipped client aggregates, in the C block.** The host describes
   `C1–C14` as counts of entities associated with the card. `C13`, `C14`, `C1` and `C5` are
   all top-10 SHAP drivers, and `C13` is the strongest feature in the model. Our
   reconstruction is a sparser, weaker version of something the dataset already contains.

**Why the competition's version was worth more.** Winning solutions computed client
aggregates over train and test together and post-processed by averaging each client's
predictions. Both use information from a client's _future_ transactions. That is
transductive — legitimate for a leaderboard, lookahead bias in production, and forbidden
here. The gap between +0.0012 and what the leaderboard saw is the price of that rule, and
it is worth knowing rather than assuming.

**Rejected.** Filling missing `addr1`/`D1` with a sentinel — measured at 79.4% purity on
that subset, _worse than `card1` alone_, because it merges unrelated clients. The id is
NULL there and every client aggregate is null-guarded, the same conclusion the device
feature reached.

**Rejected.** `client_uid` as a categorical feature — 217,735 levels would let the model
memorise clients, and every memorised client is one it can never meet again. Excluded, with
a test pinning the exclusion.

**Open.** Whether five features earning +0.0012 justify a third online lookup key in the
serving path. That is a Step 6 cost, not a Step 5 one, and it belongs to the
feature-selection task in Step 8.

---

## 2026-08-10 — Split boundaries computed exactly, not approximately

**Decision.** The train/val/test cut points use `PERCENTILE_CONT` on `TransactionDT`, not
`APPROX_QUANTILES`.

**Why.** Re-materializing the split over byte-identical input moved rows between
validation and test — 88,534 validation rows on one run, 88,590 on the next. That is
`APPROX_QUANTILES` doing what its name says. The consequence is not a wrong split; it is
that two runs are not measuring the same thing, and this project compares runs constantly:
baseline against candidate, calibrated against uncalibrated, one gate execution against
the next. A metric difference of a few thousandths — the size of several findings recorded
in this log — is inside the noise an approximate boundary introduces.

At 590,540 rows the exact computation is not a cost worth optimising away.

**How it was found.** By noticing that a row count printed in one run's logs differed from
the same count in another, while auditing whether Step 4 had delivered what it claimed.

---

## 2026-08-10 — Entity aggregates are NULL when the entity is missing

**Decision.** `device_txn_count_24h` evaluates to `NULL` when `DeviceInfo` is null, rather
than to a count. Any future aggregate over a nullable entity carries the same guard.

**Why.** SQL puts every NULL into one window partition. `DeviceInfo` is absent on 471,874
of 590,540 rows — 80% — so the unguarded aggregate was counting how many _other
device-less_ transactions had happened in the preceding 24 hours. On four fifths of the
dataset the column was a proxy for global transaction volume wearing a device feature's
name:

| Rows                       | Mean    | Max   |
| -------------------------- | ------- | ----- |
| With a device (118,666)    | 270.8   | 1,574 |
| Without a device (471,874) | 2,650.4 | 6,506 |

An order of magnitude apart, no nulls, nothing to trip over. A model will happily learn it,
and what it learns is a volume trend that will not hold when volume changes.

**Why NULL rather than 0.** Zero is a claim: this device made no transactions. There is no
device. LightGBM routes missing values down their own branch, which is the honest encoding;
zero would put device-less rows in the same bucket as brand-new devices, which are a
genuinely different and genuinely interesting population.

**Why `card1` needs no such guard.** It has zero nulls across the dataset, which is the
reason it was chosen as the card entity in the first place (2026-08-09). The asymmetry is
in the data, not in the care taken.

**How it was found.** By counting rows in the materialized table while auditing what Step 4
had actually delivered — not by review, and not by any test. The point-in-time tests all
passed, because the leakage guarantee was never violated: every value was computed from
strictly earlier rows. It was simply computed over the wrong partition. A test now asserts
the guard is in the query.

---

## 2026-08-10 — Isotonic calibration, and the tie bug it exposed

**Decision.** Raw LightGBM scores are mapped to probabilities by **isotonic regression**,
selected over Platt scaling by cross-validated log loss on validation: **0.08371 vs
0.08590**. Model artifacts carry the fitted calibrator, so nothing downstream reads a raw
score.

**Why cross-fitted selection.** Isotonic is far more flexible than a one-parameter
sigmoid, so comparing both on the data they were fitted to would hand isotonic the win by
construction. Held-out folds ask the question that matters — which mapping generalises.

**Why calibrate at all, when it costs ranking.** Calibrated test PR-AUC is **0.4992**
against **0.5091** uncalibrated. Isotonic is a step function; it creates ties and ties cost
a little ranking. That trade is accepted because the threshold, the cost model and every
number an analyst reads are all on the probability scale, and a ranking score placed on
that scale is simply wrong. Expected calibration error on test is **0.0045**.

**What the ties then broke.** The same step function put hundreds of rows on identical
probabilities, and the threshold routine took the (1 − budget) quantile of legitimate
scores — which is equivalent to "lowest threshold meeting the budget" only when scores are
distinct. The quantile landed on a repeated value and `>=` admitted the whole tied group:
a 1% budget realising **1.07%**, on validation, the split it was fitted to. The routine now
searches realised rates over distinct candidate scores. Validation FPR is 0.0090.

**How it was caught.** By the validation gate, on its first real run, failing the build.
Not by the unit test — that asserted the realised rate was within 0.015 of the budget, a
tolerance loose enough to swallow exactly this. The test now asserts `<= budget` across
four budgets and carries a tied-score fixture.

---

## 2026-08-10 — Decision threshold from a false-positive budget, with a cost model beside it

**Decision.** The blocking threshold is the lowest score whose false-positive rate still
fits a stated budget of **1% of legitimate transactions**, measured on validation. A
cost-minimising threshold is computed alongside it and reported, but does not set policy.

**Why not `argmax F1`.** F1 asserts that a missed fraud and a blocked customer are equally
bad. No payments business believes that: one is a chargeback, the other is a person whose
card was declined at a checkout, and the second has a cost the model cannot see. A budget
states how much of that invisible cost the operation will tolerate, and the threshold
follows from it arithmetically — the (1 − budget) quantile of legitimate scores.

**Why the lowest qualifying threshold.** Among thresholds that respect the budget, the
lowest blocks the most fraud. Spending less of an agreed budget is not prudence, it is
recall left on the table.

**The cost model, stated so it can be argued with.** A missed fraud costs the full
transaction amount — the money is gone and charged back. A blocked legitimate transaction
costs a flat **5.0** in review and goodwill, not its amount: the sale is usually
recoverable, the handling is not. Both numbers are assumptions. Reporting the
cost-minimising threshold next to the budget-driven one puts a price on the friction
policy, which is the number worth showing whoever set the budget.

**Rejected.** Letting the cost model set the threshold directly — it would optimise
against two invented constants and present the result as objective. The budget is at least
honestly a policy input.

---

## 2026-08-10 — Promotion thresholds enforced by the validation gate

**Decision.** A model reaches the registry only after clearing four checks, all of them
failing the Dagster run rather than warning:

| Check                            | Threshold                             | Why this one                                                                                                                                                                             |
| -------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PR-AUC above baseline            | ≥ 1.10 × BQML baseline                | A margin, not noise. A model that cannot manage 10% is not worth a second system to operate.                                                                                             |
| No cold-card regression          | unseen-card PR-AUC ≥ baseline overall | 98.6% of test rows sit on a card seen in training, so an aggregate score can look excellent while the model is useless on a genuinely new customer — who is what a fraudster looks like. |
| Calibration sane                 | ECE ≤ 0.02                            | The threshold and the cost model are both read off the probability scale; a systematically overconfident model would block the wrong volume while every ranking metric stayed healthy.   |
| Threshold survives out of sample | **test** FPR ≤ 1.25 × budget          | Checked on test, not validation. On validation the budget holds by construction, so checking it there asks nothing.                                                                      |

First run: 0.4992 vs 0.0684 × 1.1, cold-card 0.6129, ECE 0.0045, test FPR 0.0111 against a
0.01 budget.

**Why the drift allowance.** A threshold fitted to one period and carried into the next
will not land on the same rate; a gate forbidding that would reject every model forever.
A quarter of the budget is wide enough to absorb ordinary period-to-period movement and
narrow enough that a policy which stops holding gets caught.

**Why the baseline is read from storage rather than pinned.** The gate compares against
whatever the BQML baseline last scored, published to a fixed path in the model bucket. A
hardcoded number would drift from the baseline it claims to describe the moment the
features change.

**Why promotion is a marker, not a registry entry — for now.** `Model.upload` requires a
serving container image and LightGBM has no prebuilt Vertex container. Pointing the entry
at the prebuilt sklearn image would register a model that cannot serve, and the registry
would repeat that untruth to whatever deploys from it. The gate therefore writes its
verdict to `promoted/production.json` beside the artifact, and Step 6 replays it into the
registry once the FastAPI image exists. The verdict is the deliverable; the registry entry
is how it gets published.

---

## 2026-08-10 — Vertex AI Experiments is the only tracker; MLflow removed

**Decision.** Every training run is recorded in Vertex AI Experiments, through an
`ExperimentTracker` resource that exposes a single `log_run` method. MLflow is removed
entirely — dependency, resource and local store. The 2026-08-09 two-track entry is
superseded in its tracking half; its registry half stands.

**Why.** The two-track design rested on the local tracker being free and offline for
throwaway runs. The offline half never held in this architecture: training reads from
BigQuery, so any run needs the network regardless. The cost half is negligible in both
directions. What was actually left was a second place to look for a number, and a standing
opportunity for the two to disagree about what was run.

MLflow 3.x putting its file-store backend into maintenance mode sharpened the question
rather than creating it. Continuing would have meant carrying a local database file and its
schema in order to obtain a capability the managed tracker already provides — which is the
same argument that sent the registry to Vertex in the first place, applied consistently.

**Why a role name with exactly one implementation.** Assets should not know which tracker
they are talking to: `bqml_baseline` calls `log_run` and nothing else, which is what lets
the test suite substitute a recorder and never reach Vertex. But no second implementation
is written until something needs one. An abstraction with one implementation is just a
name; an abstraction with a second one added speculatively is a liability that has to be
kept working.

**What it cost.** MLflow is the open-source market standard and it is now absent from the
project. That is a deliberate trade, taken because a tracker nobody consults is worth less
than one that is the single record of what ran.

**Consequences in infrastructure.** `aiplatform.googleapis.com` and
`roles/aiplatform.user` are now declared in `iaac/`. Vertex refuses to reopen a finished
run, so run names carry the Dagster run id as a suffix — the same fix the Dataproc batch
id needed, for the same reason.

---

## 2026-08-10 — Model input assembled from two tables, not one widened feature table

**Decision.** The training matrix is `raw.ieee_train_joined` joined to
`features.transaction_features` on `TransactionID`. The feature table keeps only entity
state and is not widened to carry the raw predictors. Full argument in
[docs/feature-engineering.md](docs/feature-engineering.md).

**Why.** A transaction arriving at `POST /predict` has never been seen before, so its own
`V*` and `id_*` values cannot be retrieved from anywhere — they are in the request. Only
the card's and device's prior history can be looked up, because it predates the request.
Serving input is therefore `request fields + retrieved entity state`, and training has to
be assembled the same way or the two paths compute different things from the same data.

**Rejected.** Widening the feature table to hold the raw predictors — tidier to query, and
it would fill the feature store with columns production can never retrieve, which is the
exact shape of a training-serving skew bug. Pushing the velocity aggregates into the raw
joined table — the same asymmetry mirrored, and it would make ingestion and feature
engineering stop being independently re-runnable.

---

## 2026-08-10 — Positional window functions banned; the point-in-time rule is written down

**Decision.** `LAG`, `LEAD` and `ROWS` frames may not appear in feature engineering.
`seconds_since_prev_txn_card` is rebuilt as `MAX(TransactionDT)` over a
`RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` frame. The whole argument moves into
[docs/point-in-time.md](docs/point-in-time.md), which the code and the architecture
document both point at.

**Why.** The 2026-08-09 `RANGE` entry claimed the guarantee excludes "every peer sharing
its exact timestamp". That was true of six features and false of the seventh:
`seconds_since_prev_txn_card` used `LAG`, which is positional and therefore returns a peer
when two same-card transactions share a `TransactionDT`. Measured against the full dataset
after materialization: **166 rows** carried a gap of 0, each one told that something else
was happening on its card at that exact instant — the single strongest fraud pattern in
this data, and precisely what every other frame exists to hide. After the fix the count is
0 and the minimum gap is 1.

**What this changes about how the rule is stated.** Previously it was "use `RANGE … 1
PRECEDING`", which describes frames and says nothing about functions that take no frame.
The rule is now about the property rather than the syntax: nothing positional, because
position is not time. That is a broader ban than strictly necessary — `LAG` over a
guaranteed-unique ordering key would be safe — but the dataset has no such key, and a rule
with an exception is a rule that gets applied wrongly.

**How it was found, and what that implies.** Not by review, and not by the tests: the unit
tests assert on the shape of the generated SQL, and the SQL had exactly the shape its
author intended. It surfaced from querying the materialized output. The lesson is recorded
in the document as a rule of its own — every feature ships with an empirical check, a pair
of independently computed numbers that must agree, not only a structural assertion.

---

## 2026-08-10 — Spark leaves the pipeline; the identity join is BigQuery SQL

**Decision.** PySpark and Dataproc Serverless come off the critical path. `transaction` +
`identity` is joined by a single BigQuery query that also computes `null_count_V_block`.
`pyspark` and `google-cloud-dataproc` are dropped as dependencies; `iaac/dataproc.tf`,
`iaac/network.tf`, the Dataproc and Compute APIs and the `dataproc.*` IAM roles are
deleted. The 2026-08-09 entry scoping PySpark to this join is superseded.

**Why.** The work Spark was doing is a left join on a key plus a count of nulls. That is
ordinary SQL, and the earlier entry's justification — the transformation is "awkward to
express in SQL" — does not survive comparison with the query that replaced it. What the
Spark path cost was concrete and recurring: a JVM in the dev loop, a job that could not be
unit-tested without one, a multi-minute batch cold start per iteration, a VPC and subnet
existing solely to run it, a GCS staging bucket, and two project-level IAM roles.

The V-block reduction was the one part with a plausible claim on distributed compute, and
it does not hold either. Selecting columns out of a static anonymized block is a one-off
analysis whose output is a pinned list of column names — not a computation the pipeline
repeats on every run.

**What this costs, stated plainly.** PySpark is a market-standard tool this project was
partly meant to demonstrate, and it is now absent from the critical path. That is a real
loss, accepted deliberately rather than overlooked. Spark keeps its place in Step 8 as the
second feature-engineering backend: an optional, measured benchmark against BigQuery SQL
that blocks nothing. A result of the form "Spark loses at 590k rows, here are the numbers"
belongs there anyway — it was never a good reason to route the pipeline through a JVM.

**Rejected.** Keeping Dataproc for the portfolio line alone — building infrastructure in
order to be seen using it is exactly the failure mode this log exists to catch. Spark on
GKE instead of Dataproc — strictly more operational surface for the same output, and the
JVM does not disappear; it moves into a container image.

---

## 2026-08-10 — Online feature retrieval reopened; the Feature Store decision is superseded

**Decision.** How the serving path obtains per-entity velocity features becomes an explicit
open design question, settled in Step 6 by measurement rather than by argument. The
2026-08-09 entry "Feature store is a BigQuery table, not Vertex AI Feature Store" is
superseded.

**Why the earlier entry was wrong.** It rejected the managed Feature Store on the grounds
that this system has "no per-entity online serving requirement". The feature logic says
otherwise: `card_txn_count_1h`, `card_txn_count_24h`, `card_txn_amt_avg_24h` and
`device_txn_count_24h` are windowed aggregates partitioned by card and by device. None of
them can be derived from a single transaction payload — answering one request means
reading that entity's recent history. The requirement was there all along; the decision
was taken before a serving contract existed to expose it.

A second premise also failed. The managed Feature Store is a layer _over_ BigQuery — a
FeatureView defined on the existing table and synced to an online store — so adopting it
would not displace the feature table as the source of truth. The cost objection survives
(the online store provisions nodes and does not scale to zero); the rewrite-cost objection
does not.

**How it gets settled.** A BigQuery point lookup ships as the default path, with p50/p99
measured. Vertex AI Feature Store online serving is stood up as a timeboxed spike behind
the same retriever interface and measured identically, then torn down. Both numbers land
in this file. A rejection backed by measurement is worth more than the reasoned one it
replaces — which is precisely how this entry came to exist.

**Rejected.** Having the caller supply precomputed features — that moves the hard part of
the problem outside the system boundary and turns the request schema into a fiction.
Self-managed Redis or Bigtable as the online store — the most work of the available
options and the least to show for it, since the point being demonstrated is
online/offline consistency, not cache operations.

---

## 2026-08-10 — BigQuery ML as baseline and ablation, never as the production model

**Decision.** BigQuery ML provides the trivial baseline wired end to end in Step 5, and
reappears as one row of the Step 8 ablation table. The production model stays LightGBM.

**Why.** A one-statement `CREATE MODEL` baseline costs almost nothing and gives PR-AUC a
reference point from the first run, before any later gain can be credited to itself. Two
BQML capabilities are worth exercising on their own account: the `TRANSFORM` clause, which
bakes preprocessing into the model so training-serving skew becomes structurally
impossible rather than merely monitored, and direct registration of a BQML model into
Vertex AI Model Registry, which tests the seam between the data layer and the registry
before a real model depends on it.

**Rejected.** BQML as the production model. Calibration (isotonic vs Platt), a threshold
derived from a false-positive budget and a cost matrix, and SHAP on our own terms are the
substance of Module 2, and BQML constrains all three. The managed one-statement path is
more useful here as a measured comparison than as a candidate.

---

## 2026-08-10 — `V1–V339` reduced by NaN-group + correlation, not by variance threshold

**Decision.** The anonymized V-block is reduced in two stages: group the columns by their
missing-value pattern, then prune within each group by correlation, keeping one
representative per correlated cluster. `null_count_V_block` is retained as a meta-feature.
The `VarianceThresholdSelector` currently in `scripts/join_identity.py` is replaced.

**Why the variance filter was the wrong instrument.** Three separate problems, and the
first is the one that matters:

1. _Variance does not measure redundancy._ The actual property of this block is severe
   within-group collinearity. Two perfectly correlated columns both have high variance and
   both survive the filter, so the stated goal — reduction — is not what the filter
   achieves.
2. _The threshold is not scale-invariant._ At `0.01` on unscaled columns whose ranges
   differ by orders of magnitude, the filter removes little beyond near-constant columns.
   A binary column tops out at variance 0.25; a column valued in the thousands is in the
   millions. One threshold does not mean the same thing to both.
3. _`fillna(0)` before measuring destroys the block's most informative structure._ In
   IEEE-CIS the **missing-value pattern** is what defines the natural groups within
   `V1–V339`. Imputing zero conflates "absent" with "zero" in columns where zero is a
   legitimate value, and collapses that structure to the single scalar
   `null_count_V_block`.

**Why this alternative.** NaN-pattern grouping followed by within-group correlation
pruning is the approach public work on this competition converged on, and it attacks the
collinearity directly. Exact grouping boundaries and the correlation cutoff are
implementation details to fix against the data, and the numbers land here once measured.

**Where it runs.** In the audit repository, not as a pipeline stage. The V-block is
static and anonymized, so column selection over it is done once and its output is a pinned
list of column names the join then selects — the same reasoning that keeps the raw schema
pinned in Terraform rather than re-inferred per run. `null_count_V_block` stays in the
join query, computed on raw nulls before anything imputes them.

**Rejected.** PCA over the V-block — comparable reduction, but the components are not
interpretable, and Module 2 commits to SHAP explanations that a fraud analyst can act on.
Dropping the block wholesale — cheap, and discards real signal.

---

## 2026-08-09 — Portable data processing, managed model lifecycle

**Decision.** The open-source/managed boundary runs between the two halves of the
platform. Data processing (Spark, Dagster, the transformation logic) stays portable and
vendor-neutral. Model lifecycle — registry, serving, monitoring — is handed to managed
cloud services.

**Why.** These two halves have opposite economics. Transformation logic is where the
domain knowledge lives, it is the largest body of code, and it is the part most likely to
outlive any single cloud account — so it is worth keeping free of vendor semantics.
Registry, serving and monitoring are undifferentiated infrastructure: self-hosting them
means running a tracking server, a metadata database and its backups in order to obtain
capabilities the platform already sells, and that operational debt compounds quietly
while delivering nothing a user of the system can see.

**Rejected.** Fully open source — maximum portability, but the maintenance burden lands
on exactly the components that produce no differentiated value. Fully cloud-native —
fastest to build, but the transformation logic becomes rewrite-on-migration, which is the
expensive kind of lock-in.

---

## 2026-08-09 — Vertex AI Model Registry as the source of truth, MLflow local

> **Superseded 2026-08-10.** The MLflow half of this entry is gone; Vertex AI Experiments
> is now the only tracker. The registry half stands. Left as written; the correction is
> the entry at the top of this file.

**Decision.** Vertex AI Experiments + Model Registry is the production path for tracking
and model registration. MLflow stays, but only as a local file-store tracker for the
iteration loop.

**Why.** Direct application of the boundary above: the registry is the system of record
for what is deployed, and hosting it ourselves buys nothing. Keeping MLflow locally costs
nothing, works offline, and means the dozens of experiments that are not worth keeping
never touch a billable service. The training code logs through a thin wrapper, so which
backend receives a run is a configuration concern rather than a rewrite.

**Rejected.** Self-hosted MLflow as the registry of record — real hosting cost and
maintenance for a capability the platform already provides. Vertex-only — every
throwaway experiment would then require network access and spend.

---

## 2026-08-09 — PySpark scoped to the identity join, not to feature engineering

> **Superseded 2026-08-10.** The claim below that the join is "awkward to express in SQL"
> did not hold. Left as written; the correction is the 2026-08-10 entry at the top.

**Decision.** PySpark on Dataproc Serverless implements the `transaction` + `identity`
join and the handling of the anonymized `V1–V339` block. The existing BigQuery SQL
feature engineering stays as it is. A second PySpark backend for the same feature logic
is added later purely as a benchmark.

**Why.** At ~590k rows Spark is objectively unnecessary — BigQuery handles this volume
without leaving SQL, and rewriting working, leakage-safe SQL to introduce a distributed
engine would trade correctness risk for nothing. The identity join is different: ~430
columns with heavy null structure, where the transformation is awkward to express in SQL
and benefits from a real programming model. The dual-backend benchmark then answers the
question the architecture actually raises — at what volume does the choice flip — with a
measurement instead of an assumption.

**Rejected.** Rewriting feature engineering in PySpark — discards correct code and
misrepresents the scale the system operates at. Skipping Spark entirely — leaves no
tested path for the case where data volume grows by two orders of magnitude.

---

## 2026-08-09 — Identity join before training, not after

**Decision.** Ingest and join the identity table (Step 4) before building the training
pipeline and the serving contract.

**Why.** Identity adds roughly 40 columns to the model input. Doing it after Step 5/6
would force a rework of the Pydantic request schema, feature retrieval, and their tests.
Ordering it first costs nothing and avoids that churn.

---

## 2026-08-09 — Causal inference stays out of scope

**Decision.** Causal inference, uplift modelling and dynamic pricing are not part of this
project.

**Why.** Fraud detection is a predictive classification problem: the target is observed,
and the model is scored against it. Causal methods estimate the effect of an intervention
that was never randomized here, which requires different assumptions, a different
validation strategy, and data this dataset does not contain. Sharing a pipeline between
the two would mean a feature store and a validation harness that serve neither well.

---

## 2026-08-09 — Feature store is a BigQuery table, not Vertex AI Feature Store

> **Superseded 2026-08-10.** The premise below — that there is no per-entity online
> serving requirement — is contradicted by the velocity features this project already
> computes. Left as written; the correction is the 2026-08-10 entry above.

**Decision.** The feature store is a versioned BigQuery table produced by a Dagster asset.

**Why.** One consumer, no per-entity online serving requirement. The managed service adds
cost and operational surface for capabilities this system does not use. Revisit if online
per-entity lookup ever becomes a real requirement.

---

## 2026-08-09 — `RANGE … 1 PRECEDING` for point-in-time correctness

**Decision.** Every windowed aggregate uses a `RANGE` frame ending at `1 PRECEDING`,
ordered by `TransactionDT`.

**Why.** `RANGE` frames on the `ORDER BY` **value**, not on row position the way `ROWS`
does. So `1 PRECEDING` means "`TransactionDT` strictly less than the current row's",
which excludes the current row _and_ every peer sharing its exact timestamp. A `ROWS`
frame would let simultaneous transactions see each other — the exact leakage failure mode
this step exists to prevent.

---

## 2026-08-09 — `card1` as the card entity proxy

**Decision.** Velocity aggregates are grouped by `card1`.

**Why.** It is the most granular fully populated card field in the IEEE-CIS schema
(`card2`–`card6` carry nulls) and the conventional identifier used against this dataset.
A tighter entity would combine several card and address fields, but that is a modeling
refinement — the point-in-time guarantee holds regardless of which columns define the
entity.

---

## 2026-08-09 — Feature table rebuilt with `CREATE OR REPLACE`

**Decision.** The feature asset rebuilds its destination table from scratch on every run.

**Why.** No incremental state means nothing can drift out of sync with the raw table,
which itself is `WRITE_TRUNCATE`-loaded. Re-materializing is always safe. The dataset is
static and small enough that a full rebuild is cheap; revisit if it stops being either.

---

## 2026-08-08 — Raw table schema pinned in Terraform

**Decision.** `ieee_train_transaction_raw` is managed as a Terraform resource with an
explicit schema file, generated and audited via the `raw_transaction_bq_schema` Dagster
asset. Everything else is created by pipeline code.

**Why.** A schema change then surfaces as a reviewable `tofu plan` diff instead of being
silently re-inferred on the next ingestion run. Deliberately no `ignore_changes` on
schema — that would defeat the entire purpose.

---

## 2026-08-07 — Kaggle → GCS staging as a manually materialized asset

**Decision.** `raw_transaction_kaggle_to_gcs` is not wired as a dependency of the
validation and load assets. It is materialized by hand, rarely.

**Why.** The dataset is static. Making it an upstream dependency would re-download
hundreds of megabytes on every ingestion run to produce a byte-identical file.

---

## 2026-08-05 — Separate dev and prod IAM profiles

**Decision.** The service account's roles are selected by environment. `prod` gets
`bigquery.dataViewer` on `raw`; `dev` gets `dataEditor`.

**Why.** Nothing in production has a reason to write to the raw dataset — only the
ingestion pipeline does, and that runs in dev. Roles are added when a resource that needs
them actually exists (`run.invoker` is deliberately absent until Cloud Run is deployed),
so the IAM surface stays honest rather than aspirational.

---

## 2026-08-05 — Dagster instead of Airflow

**Decision.** Dagster with Software-Defined Assets as the orchestrator.

**Why.** Asset-based orchestration makes pipelines testable locally without standing up a
heavy metadata database. The unit of reasoning is the data artifact, which matches how a
feature store and a model registry are actually thought about — a task-graph orchestrator
forces you to reason about the artifacts indirectly, through the tasks that happen to
produce them.

**Rejected.** Managed Airflow (Cloud Composer) — a market standard with less local
iteration friction removed and a running cost even when idle. Orchestration sits on the
portable side of the open-source/managed boundary deliberately: it is coupled to the
transformation logic, so it should move with it.

---

## 2026-08-05 — OpenTofu for infrastructure

**Decision.** All GCP resources defined in OpenTofu; nothing created by hand in the
console.

**Why.** The claim being made by this project is platform engineering. Clicked
infrastructure cannot be reviewed, reproduced, or torn down cleanly.

---

## Open — to be decided

| Question                                                                                                                    | Decide by |
| --------------------------------------------------------------------------------------------------------------------------- | --------- |
| Online feature retrieval: BigQuery point lookup vs Vertex AI Feature Store — decided on measured p50/p99, not on preference | Step 6    |
