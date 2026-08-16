# Measurements

> What is true now, on what data, and how far it can be trusted. Numbers, not narrative.
>
> How these numbers were arrived at, including the conclusions that had to be retracted,
> is in git history.
>
> Architecture: [architecture.md](architecture.md) · Leakage: [point-in-time.md](point-in-time.md)
> · Model card: [model-card.md](model-card.md)

---

## The number this project exists to produce — measured 2026-08-16

The published solutions scoring ~0.96 compute their uid aggregates as **full-group
statistics over the whole dataset**, including rows later than the one being scored. This
pipeline computes the same aggregates under `RANGE … 1 PRECEDING`.

The two differ by one window frame and nothing else. Swapping
`AND 1 PRECEDING` for `AND UNBOUNDED FOLLOWING`, rebuilding the feature table and retraining
with the same configuration and the same seeds:

| Formulation | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| point-in-time — a row sees its entity's past only | 0.8957 | 0.5219 |
| transductive — the aggregate sees the row's future | **0.9127** | **0.5847** |
| **the price of causal correctness** | **+0.0169** | **+0.0628** |

Three seeds each, and the gap is roughly **seven times** the resolution of a three-seed
comparison. This is not a noise result.

**It also corrects a claim this file used to make.** The earlier text asserted that this
difference was "most of the gap" to 0.96. It is not: the gap is about 0.064 ROC-AUC and the
window frame accounts for **0.0169 of it, roughly a quarter**. The rest is everything else
the winning solutions do — larger feature sets, ensembling, seed averaging, competition
tuning — and remains unattributed here.

So the honest statement is narrower and more useful than the one it replaces: **refusing to
look at the future costs 0.017 ROC-AUC on this dataset.** That is a price worth naming
precisely, and it is not the whole distance to a leaderboard.

The transductive variant is built by an offline script into its own table and is never read
by the production path.

## Resolution — read this before any comparison

| | sd | a difference of two single fits |
| --- | ---: | ---: |
| ROC-AUC | 0.0029 | 0.0041 |
| PR-AUC | 0.0065 | 0.0092 |

Five seeds, same configuration, nothing moving but the LightGBM seed
([`uv run noise-band`](../src/fraud_detection/cli/noise_band.py)).

Local-to-leaderboard transfer adds a second, larger source of noise. Three submissions:

| Contract | Features | Local ROC-AUC | Public LB | offset |
| --- | ---: | ---: | ---: | ---: |
| `6ea8eb8b` | 184 | 0.8960 | 0.898170 | +0.00217 |
| `bdb97707` | 225 | 0.8963 | 0.896944 | +0.00064 |
| `401d407a` | 224 | **0.8946** | **0.898204** | +0.00360 |

**The offset varies more than either measurement it connects** — 0.00296 against a local
spread of 0.0017 and an LB spread of 0.00126. And the ordering inverts perfectly: sorted by
local score ascending, the leaderboard scores run 0.898204, 0.898170, 0.896944 — descending,
three out of three. The model with the *worst* local ROC-AUC scored *best* on the
leaderboard.

Three points cannot establish an inverse relationship (a perfect inversion happens by chance
one time in six). What they do establish is that within this band **the local score carries
no usable information about the leaderboard**, which is exactly what a noise floor of 0.0029
predicts. All three models are the same model.

**A single fit resolves nothing below ~0.009 PR-AUC or ~0.004 ROC-AUC, and a single
submission resolves nothing below ~0.005 ROC-AUC.** Differences smaller than that are
recorded as *unmeasured*, never as *no effect*. Do not spend a submission comparing two
things that differ by less.

The dominant source of the fit noise is early stopping, not the model: `best_iteration`
ranged over 431–1581 across the five seeds.

## Dataset and splits

| | |
| --- | --- |
| Rows, `train_transaction` | 590,540 |
| Fraud rate | ~3.5% |
| Columns declared to the contract | 502 |
| **Features reaching the model** | **224** (contract `401d407a9afd794f`) |

Time axis on `TransactionDT`. Train ends at 0.75, validation runs 0.85–0.90, test 0.90–1.0.

| | rows |
| --- | ---: |
| train | 442,905 (15,563 fraud, `scale_pos_weight` 27.5) |
| **gap — deliberately unassigned** | **59,054** |
| val | 29,527 |
| test | 59,054 |

**The 10% gap is the point, not an oversight.** A chargeback arrives months after the
transaction, so at deployment time the most recent period is never finished being labelled.
Validating on a window flush against training measures a situation nobody ever has.

Its *width* remains unjustified by measurement: the attempt to measure it turned out to be
tracking population composition rather than elapsed time.

## Current model — `lightgbm/2af70926`

LightGBM, **205 features** under contract `36b7acba59944bac`, Platt calibration,
deterministic. The override is gone: this contract admits nothing the audits rejected.

| Metric | Value |
| --- | --- |
| Test PR-AUC | 0.5210 |
| Expected calibration error | 0.0077 |
| Threshold (1% FP budget) | 0.2273 |
| Recall / precision at threshold | 0.456 / 0.603 |

**The prediction held.** Retiring nineteen columns was measured at −0.0002 ROC-AUC before it
was done; the retrained model came in at 0.5210 PR-AUC against the previous contract's
0.5229, a difference of 0.0019 — well inside the 0.0092 that separates two single fits. A
contract 19 columns lighter performs the same, which is what the measurement said it would.

All five gate checks passed: cold-entity PR-AUC 0.5319, ECE 0.0077, test FPR 0.0117 against
a 1.25× budget, and the dominant segment at a lift of 9.95 against a floor of 5.0.

### Where the model is uneven

By reconstructed client (`card1 + addr1 + (day − D1)`):

| Segment | Rows | Share | PR-AUC |
| --- | ---: | ---: | ---: |
| Client seen in training | 20,405 | 34.6% | 0.5671 |
| Client **not** seen in training | 38,649 | 65.4% | 0.5371 |

By product — **the larger unevenness, and the one that went unnoticed longest**:

| `ProductCD` | Rows | Base rate | PR-AUC | Lift |
| --- | ---: | ---: | ---: | ---: |
| R | 2,938 | 0.0466 | 0.8207 | 17.6 |
| C | 6,481 | 0.1467 | 0.7220 | 4.9 |
| S | 2,437 | 0.0464 | 0.7064 | 15.2 |
| H | 1,772 | 0.0632 | 0.4255 | 6.7 |
| **W** | **45,426** | 0.0198 | **0.1976** | **10.0** |

W is 77% of scored rows. Its PR-AUC is a quarter of R's, and its lift over its own base rate
is mid-table — the raw number and the lift disagree, which is why the gate's check on this
segment is relative rather than absolute.

At the operating threshold the model misses 56% of frauds by count and **67% by value**:
missed frauds average 183 against 117 for caught ones.

## The promotion gate

Five checks, each failing the Dagster run. Every one passed on `80c25e88`:

| Check | Threshold | Result |
| --- | --- | --- |
| PR-AUC above baseline | ≥ 1.10 × 0.0635 | 0.5229 |
| No cold-entity regression | unseen-client PR-AUC ≥ baseline | 0.5371 |
| Calibration sane | ECE ≤ 0.02 | 0.0080 |
| Threshold survives out of sample | test FPR ≤ 1.25 × budget | 0.0108 |
| Dominant segment not regressed | lift ≥ 5.0 on the largest product | 9.97 on W |

The baseline is pinned in `config/orchestration.toml`; the BQML model that produced it is
one `CREATE MODEL` statement over a fixed split and returns the same number every run.

## The feature contract

502 declared, 224 admitted, 19 readmitted by policy override. Six audits contribute
fragments, each carrying its own reproducibility evidence:

| Audit | Rejects | Reproducibility | Reading |
| --- | ---: | --- | --- |
| distribution_shift | 46 | **1.00** at doubled bin count | strongest |
| redundancy | 210 | 0.78 over 93 measurable groups | sound |
| time_consistency | 134 | **0.32** at a 1.5× wider window | weakest, and it rejects the most |
| segment_qualification | 0 (would reject 26) | report-only, on evidence | see below |

`time_consistency` being both the least reproducible and the most prolific is the reason the
19 uid aggregates it rejected were readmitted by override — and that override's own effect
is **unmeasured**, at 0.78 sd.

`segment_qualification` reports rather than rejects: applying its verdicts cost 0.0325 PR-AUC
and gained 0.0005 on the segment it was protecting. It still records that **262 of 502
columns are unmeasurable inside a segment holding 74% of rows**, which is a real coverage
finding.

### Does the contract select on signal? — measured 2026-08-16

Four column sets of **the same size**, so the comparison is about *which* columns and not
how many. Five seeds each; the resolution of a five-seed mean is ±0.0018 ROC-AUC.

| Column set | Columns | ROC-AUC | vs admitted | |
| --- | ---: | ---: | ---: | --- |
| **admitted** — the contract | 224 | **0.8954** | — | |
| admitted minus the 19 overridden | 205 | 0.8951 | −0.0002 | **0.1× resolution — nothing** |
| random draw from everything | 224 | 0.8890 | −0.0063 | 3.5× — real |
| draw from what the audits rejected | 224 | 0.8674 | **−0.0280** | **15.6× — large** |

**The audits select on signal.** Admitted beats random, and random beats rejected, in the
order you would expect if the checks were working — and the margin against the rejected set
is the largest effect measured anywhere in this project.

That is the evidence the contract previously lacked. It had been defended on serving surface
and monitoring surface, honestly, because no measurement supported a quality claim. One does
now.

**The policy override does not earn its place.** Nineteen columns readmitted on the argument
that `time_consistency` measures an aggregate changing meaning rather than a signal
inverting: **−0.0002 ROC-AUC**, a tenth of the resolution. The argument still looks correct;
it is simply not worth nineteen columns. Retiring it leaves a simpler contract that performs
identically.

### The cost of admission — superseded

Earlier ablations on a different split reported the contract *costing* 0.010–0.013 PR-AUC.
Every one of those differences sat inside the noise band and none of them supported a
conclusion. The table above replaces them.

## Open

| | |
| --- | --- |
| ~~Whether the uid override helps~~ | **measured: −0.0002 ROC-AUC.** It does not. |
| Embargo width | the measurement tracked composition; needs the base-rate control in notebook 13 |
| Why the model is weak on `ProductCD == "W"` | features, label, or base rate — not separated |
| ROC-AUC decay under a controlled population | not measured |
