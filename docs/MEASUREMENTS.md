# Measurements

What is true now, on what data, and how far it can be trusted. How each number was arrived
at, including the conclusions that had to be retracted, is in git history and
[DECISIONS.md](../DECISIONS.md).

Related: [architecture.md](architecture.md), [point-in-time.md](point-in-time.md),
[model-card.md](model-card.md).

## Resolution

Read this before any comparison below.

| | sd | difference between two single fits |
| --- | ---: | ---: |
| ROC-AUC | 0.0029 | 0.0041 |
| PR-AUC | 0.0065 | 0.0092 |

Five seeds, same configuration, nothing moving but the LightGBM seed
([`uv run noise-band`](../src/fraud_detection/cli/noise_band.py)). The dominant source is
early stopping: `best_iteration` ranged over 431–1581 across the five runs.

Transfer to the leaderboard adds a second, larger source of noise. Three submissions:

| Contract | Features | Local ROC-AUC | Public LB | offset |
| --- | ---: | ---: | ---: | ---: |
| `6ea8eb8b` | 184 | 0.8960 | 0.898170 | +0.00217 |
| `bdb97707` | 225 | 0.8963 | 0.896944 | +0.00064 |
| `401d407a` | 224 | 0.8946 | 0.898204 | +0.00360 |

The offset varies more than either measurement it connects: 0.00296 against a local spread of
0.0017 and a leaderboard spread of 0.00126. Sorted by local score ascending, the leaderboard
scores run descending, three out of three. Three points cannot establish an inverse
relationship (a perfect inversion happens by chance one time in six), but they do show that
inside this band the local score carries no usable information about the leaderboard. All
three models are the same model.

A single fit resolves nothing below ~0.009 PR-AUC or ~0.004 ROC-AUC, and a single submission
nothing below ~0.005 ROC-AUC. Smaller differences are recorded as unmeasured, not as an
absence of effect.

## The cost of point-in-time correctness, measured 2026-08-16

The published solutions compute their uid aggregates as full-group
statistics over the whole dataset, including rows later than the one being scored. This
pipeline computes the same aggregates under `RANGE … 1 PRECEDING`. The two differ by one
window frame and nothing else, so swapping `AND 1 PRECEDING` for `AND UNBOUNDED FOLLOWING`,
rebuilding the feature table and retraining with the same configuration and seeds isolates
the difference:

| Formulation | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| point-in-time, a row sees its entity's past only | 0.8957 | 0.5219 |
| transductive, the aggregate sees the row's future | 0.9127 | 0.5847 |
| difference | +0.0169 | +0.0628 |

Three seeds each. The gap is roughly seven times the resolution of a three-seed comparison.

This corrects a claim this file used to make. The earlier text said the window frame accounted
for most of the distance to the winning solutions, which took the competition at 0.945 on the
private leaderboard. From 0.8957 that distance is about 0.049 ROC-AUC, and the frame accounts
for 0.0169 of it, roughly a third. The rest is everything else those solutions do (larger
feature sets, ensembling, seed averaging, competition tuning) and remains unattributed here.

The transductive variant is built by an offline script into its own table and is never read by
the production path.

## What calibration costs, on a run where isotonic won

Calibration is chosen per run between Platt and isotonic on cross-validated log loss (see
[model-card.md](model-card.md) §3). On a run where isotonic won by 0.0007 log loss, it
collapsed 59,054 test scores into 93 distinct values — a step function's tie problem — costing
0.0211 PR-AUC against the uncalibrated ranking. That is why the submission carries the raw
score while only decisions and `prediction_logs` read the calibrated one
([architecture.md §5](architecture.md)).

The pinned model below selected Platt instead, so this cost is not currently being paid; it
recurs whenever a future run selects isotonic.

## Dataset and splits

| | |
| --- | --- |
| Rows, `train_transaction` | 590,540 |
| Fraud rate | ~3.5% |
| Columns declared to the contract | 502 |
| Features reaching the model | 205, contract `36b7acba59944bac` |

Time axis on `TransactionDT`. Train ends at 0.75, validation runs 0.85–0.90, test 0.90–1.0.

| | rows |
| --- | ---: |
| train | 442,905 (15,563 fraud, `scale_pos_weight` 27.5) |
| gap, unassigned | 59,054 |
| val | 29,527 |
| test | 59,054 |

The 10% gap is intentional. A chargeback arrives months after the transaction, so at
deployment time the most recent period is never finished being labelled, and validating on a
window flush against training measures a situation nobody ever has. The gap's *width* is still
unjustified by measurement: the attempt to justify it turned out to be tracking population
composition rather than elapsed time.

## Current model, `lightgbm/2af70926`

LightGBM, 205 features under contract `36b7acba59944bac`, Platt calibration, deterministic.

| Metric | Value |
| --- | --- |
| Test PR-AUC | 0.5210 |
| Expected calibration error | 0.0077 |
| Threshold at a 1% false-positive budget | 0.2273 |
| Recall / precision at threshold | 0.456 / 0.603 |

Retiring the nineteen overridden columns was measured at −0.0002 ROC-AUC before it was done.
The retrained model came in at 0.5210 PR-AUC against the previous contract's 0.5229, a
difference of 0.0019, well inside the 0.0092 that separates two single fits.

### Where the model is uneven

By reconstructed client (`card1 + addr1 + (day − D1)`):

| Segment | Rows | Share | PR-AUC |
| --- | ---: | ---: | ---: |
| Client seen in training | 20,405 | 34.6% | 0.5671 |
| Client not seen in training | 38,649 | 65.4% | 0.5371 |

By product, which is the larger unevenness and the one that went unnoticed longest:

| `ProductCD` | Rows | Base rate | PR-AUC | Lift |
| --- | ---: | ---: | ---: | ---: |
| R | 2,938 | 0.0466 | 0.8207 | 17.6 |
| C | 6,481 | 0.1467 | 0.7220 | 4.9 |
| S | 2,437 | 0.0464 | 0.7064 | 15.2 |
| H | 1,772 | 0.0632 | 0.4255 | 6.7 |
| W | 45,426 | 0.0198 | 0.1976 | 10.0 |

W is 77% of scored rows and its PR-AUC is a quarter of R's, while its lift over its own base
rate is mid-table. The raw number and the lift disagree, which is why the gate checks this
segment relative to its base rate rather than against an absolute floor.

At the operating threshold the model misses 56% of frauds by count and 67% by value: missed
frauds average 183 against 117 for caught ones.

## The promotion gate

Five checks, each failing the Dagster run. Results for `lightgbm/2af70926`:

| Check | Threshold | Result |
| --- | --- | --- |
| PR-AUC above baseline | ≥ 1.10 × 0.0635 | 0.5210 |
| No cold-entity regression | unseen-client PR-AUC ≥ baseline | 0.5319 |
| Calibration sane | ECE ≤ 0.02 | 0.0077 |
| Threshold survives out of sample | test FPR ≤ 1.25 × budget | 0.0117 |
| Dominant segment not regressed | lift ≥ 5.0 on the largest product | 9.95 on W |

The baseline is pinned in `config/orchestration.toml`. The BQML model that produced it is one
`CREATE MODEL` statement over a fixed split and returns the same number every run.

Two of these five checks could not fail in an earlier version: one compared a metric against
a floor of `0.0`, and the other read a metric key training never wrote, so
`dict.get(key, 0.0)` turned the missing value into a pass. Both were rewritten; the table
above is the gate as it runs now.

## The feature contract

502 columns declared, 205 admitted, 297 rejected, no overrides. Four audits write fragments,
each carrying its own reproducibility evidence:

| Audit | Rejects | Reproducibility | Reading |
| --- | ---: | --- | --- |
| distribution_shift | 46 | 1.00 at doubled bin count | strongest |
| redundancy | 210 | 0.78 over 93 measurable groups | sound |
| time_consistency | 134 | 0.32 at a 1.5× wider window | weakest, and it rejects the most |
| segment_qualification | 0 (would reject 26) | report-only | see below |

The rejection counts sum to more than 297 because a column can be rejected by more than one
check.

`time_consistency` being both the least reproducible and the most prolific is why its
verdicts on nineteen uid aggregates were once readmitted by policy override. That override has
since been retired, on the measurement below.

`segment_qualification` reports rather than rejects: applying its verdicts cost 0.0325 PR-AUC
and gained 0.0005 on the segment it was protecting. It still records that 262 of 502 columns
are unmeasurable inside a segment holding 74% of rows, which is a real coverage finding.

### Does the contract select on signal? Measured 2026-08-16

Four column sets of the same size, so the comparison is about which columns rather than how
many. Five seeds each; the resolution of a five-seed mean is ±0.0018 ROC-AUC.

| Column set | Columns | ROC-AUC | vs admitted | |
| --- | ---: | ---: | ---: | --- |
| admitted, the contract | 224 | 0.8954 | | |
| admitted minus the 19 overridden | 205 | 0.8951 | −0.0002 | 0.1× resolution |
| random draw from everything | 224 | 0.8890 | −0.0063 | 3.5× |
| draw from what the audits rejected | 224 | 0.8674 | −0.0280 | 15.6× |

Admitted beats random and random beats rejected, which is the ordering expected if the checks
work. The margin against the rejected set is the largest effect measured anywhere in this
project.

That is evidence the contract previously lacked. Until this run it had been defended on
serving surface and monitoring surface only, because no measurement supported a quality claim.

The override does not earn its place. Nineteen columns were readmitted on the argument that
`time_consistency` measures an aggregate changing meaning rather than a signal inverting;
the effect is −0.0002 ROC-AUC. The argument still looks correct, but it is not worth nineteen
columns, and retiring it leaves a smaller contract that performs identically.

### Superseded

Earlier ablations on a different split reported the contract costing 0.010–0.013 PR-AUC. Every
one of those differences sat inside the noise band. The table above replaces them.

## Open

| | |
| --- | --- |
| Embargo width | the measurement tracked population composition; needs the base-rate control from the EDA notebook on decay |
| Why the model is weak on `ProductCD == "W"` | features, label or base rate, not separated |
| ROC-AUC decay under a controlled population | not measured |
