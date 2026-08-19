# Statistical feature audits

The audits that decide which columns reach the model, stated as statistics
rather than as single-feature models. A verdict here is a rank statistic, a
weight-of-evidence table or a two-sample test, so a rejection reads as a
sentence about a bin instead of as the output of a fit.

This half of the repository ends at the contract. Nothing here trains a
production model, and nothing downstream recomputes a verdict.

```
parquet (features.model_input)
   -> notebooks/*.qmd        one question each, each writing a fragment
   -> out/fragments/*.json   verdicts plus the evidence behind them
   -> out/tables/*.csv       the full report a person reads to look up a column
   -> build-contract.qmd     merges fragments; computes nothing
   -> out/contract-body.json
   -> [cut] Python stamps the fingerprint, trains, gates, scores
```

## The audits

Three things a verdict can do to the contract, and the same three marks are used
in every table on this page:

| | |
| :---: | --- |
| ✅ | **Rejects.** A verdict here can remove a column from the feature set, or fix the entity key. |
| 📊 | **Recorded.** Written into the contract as evidence, and never removes anything. |
| ➖ | **Descriptive.** Never reaches the contract at all. |

| Notebook | Question | Method | |
| --- | --- | --- | :---: |
| `does-a-feature-still-mean-the-same-later` | Has a column reversed between an early and a late window? | Weight of evidence, Somers' D, DeLong intervals and test | ✅ |
| `has-the-population-moved` | Did the distribution move, and are the periods distinguishable jointly? | PSI against a measured null, Anderson–Darling, energy test | ✅ |
| `which-columns-say-the-same-thing-twice` | Which columns restate their neighbours? | Variable clustering on rank correlation, redundancy analysis on splines | ✅ |
| `who-is-the-customer-when-the-data-does-not-say` | Is the reconstructed entity real, and which key? | Label purity against a permuted null, coverage, concentration | ✅ |
| `does-a-column-work-inside-every-segment` | Does a pooled association survive conditioning on the product? | Cochran–Mantel–Haenszel, Breslow–Day | 📊 |
| `what-the-columns-are-made-of` | Does a column know more about the period than about fraud; how many dimensions does a V block have; is absence random? | Cramér's V with a bootstrap interval, Horn's parallel analysis, tetrachoric correlation, Little's MCAR | 📊 |
| `what-the-fraud-literature-asks…` | What is true of the phenomenon rather than of a column? | Benford, power-law tail index, Bai–Perron breaks, Rayleigh, Gini/HHI | ➖ |
| `what-does-this-data-look-like` | How much fraud, where, and are the columns the type they claim? | Wilson intervals on every rate, weight of evidence on the amount, cardinality against declared type | ➖ |

`build-contract.qmd` merges the fragments, and asks two questions about the result
itself: whether the admitted set carries more information value than random sets of the
same size, and how much of it survives moving the drift threshold.

No audit trains a model. Where the literature reaches for one — a single-feature
fit to score a column, a classifier to tell two periods apart — a ten-bin
scorecard and a permutation two-sample test answer the same question and print.

## Every statistic, and what it decides

Each row is a test or an interval, not a threshold on a point estimate. The mark
in the last column is the legend above: ✅ can reject, 📊 recorded as evidence,
➖ descriptive only.

| Question | Method | R | |
| --- | --- | --- | :---: |
| Does a column carry signal at all? | Somers' D, identical to AUC on the Mann–Whitney scale | `Hmisc::somers2` | ✅ |
| Is that signal distinguishable from none? | AUC with a DeLong confidence interval | `pROC::ci.auc` | ✅ |
| Did it change between an early and a late window? | DeLong's test for two AUCs, unpaired | `pROC::roc.test` | ✅ |
| Did it reverse direction? | Sign flips in the weight of evidence per bin, weighted by the mass they carry | own, on a pinned binning | ✅ |
| Did the column's distribution move? | PSI against its **measured** null, drawn from the multivariate hypergeometric | own | ✅ |
| Is the move more than sampling noise? | Anderson–Darling and Cramér–von Mises, permutation p-values | `twosamples` | ✅ |
| Which columns restate their neighbours? | Variable clustering on rank correlation, cut on shared variance | `stats::hclust` on Spearman ρ² | ✅ |
| Which are *reconstructable* from the others? | Redundancy analysis on restricted cubic splines | `Hmisc::redun` | ✅ |
| Is the reconstructed customer real? | Label purity against a permuted null that keeps the group sizes | own | ✅ |
| How much information does it carry? | Information value over the same bins | own | 📊 |
| Are the two periods distinguishable **jointly**? | Energy two-sample test — adversarial validation without the adversary | `energy::eqdist.etest` | 📊 |
| How strong is a categorical dependence, and how precisely known? | Cramér's V with Bergsma's correction, bootstrapped on the contingency table | `DescTools`, own | 📊 |
| How many dimensions does a block of V columns really have? | Horn's parallel analysis | `psych::fa.parallel` | 📊 |
| Are the binary M columns one latent thing? | Tetrachoric correlation | `psych::tetrachoric` | 📊 |
| Does an association survive conditioning on the product segment? | Cochran–Mantel–Haenszel | `stats::mantelhaen.test` | 📊 |
| Is it the *same* association in every segment? | Breslow–Day test for homogeneity of odds ratios | `DescTools::BreslowDayTest` | 📊 |
| Is a value missing at random? | Little's MCAR test, against always-observed columns | `naniar::mcar_test` | 📊 |
| Do the amounts look naturally generated? | Benford's law, with the effect size next to the p-value | `benford.analysis` | ➖ |
| How heavy is the tail of the losses? | Power-law tail index, Clauset–Shalizi–Newman | `poweRlaw` | ➖ |
| When did the fraud rate actually change? | Bai–Perron structural breaks | `strucchange::breakpoints` | ➖ |
| Is there a daily cycle? | Rayleigh's test — hour of day is circular, not linear | `circular::rayleigh.test` | ➖ |
| How concentrated is the customer base? | Gini and Herfindahl–Hirschman on transactions per entity | `ineq` | ➖ |

### The descriptive pass

What is in front of the audits, rather than whether a column can be trusted.
Nothing here touches the contract.

| Question | Method | |
| --- | --- | :---: |
| How much fraud is there, and does the rate hold still? | Wilson intervals on the daily rate | ➖ |
| Where does the fraud sit? | Fraud rate per product and per device-info presence, with non-overlapping Wilson intervals | ➖ |
| Does rarity predict fraud? | Fraud rate by frequency band, against each column's own base rate | ➖ |
| What does the amount say on its own? | Weight of evidence and information value over the amount | ➖ |
| Are the columns the type they claim? | Stored type against effective cardinality | ➖ |

## Why not a model

Fitting a single-feature model per column and reading its AUC answers the
question, at three costs: the direction it learned is not recoverable, a
rejection cannot be explained to whoever approves the feature set, and the
verdict depends on hyperparameters that have nothing to do with the data.

Binning plus weight of evidence is also a learned, possibly non-monotone mapping
from values to a fraud ordering — and it prints as eleven rows. The AUC of a
WoE-scored window is exactly the Mann–Whitney statistic, so nothing is given up
in precision.

## Two keys, both required

At 100,000 rows per window almost any difference is statistically significant.
Every rejection therefore needs both:

- **Significance** — a DeLong confidence interval that excludes 0.5, and a
  p-value surviving Benjamini–Hochberg across the whole scan. Without the
  correction, 377 tests at a nominal 5% produce roughly nineteen rejections by
  chance alone.
- **Materiality** — a drop in Gini that clears a stated threshold. Significance
  alone would reject most of the table for movements nobody would act on.

## Running it

```bash
Rscript -e 'renv::restore()'                # once: the pinned library
Rscript -e 'targets::tar_make()'            # the audit graph
Rscript -e 'targets::tar_visnetwork()'      # draw it
quarto render notebooks/                    # the readable version
quarto render build-contract.qmd            # the merge
Rscript tests/run.R                         # the suite, exactly as CI runs it
```

`renv.lock` pins 117 packages to the versions these results were produced under, resolved
against R 4.5.1, and `.Rprofile` activates the project library automatically. The
interpreter version is pinned in CI too: a lockfile records packages and not the R they
were resolved against, so leaving that floating would let a new release change a verdict
while the lockfile still claimed reproducibility.

Every test here is synthetic — no parquet, no warehouse, no credential — which is what
lets the statistics be checked on a pull request rather than only on a machine that has
the dataset.

`targets` is to this package what Dagster is to the pipeline: a dependency graph
with content-addressed invalidation. The two are deliberately not wired
together — the contract file is the only thing they share.

The graph and the notebooks both write fragments, and they must write the *same*
fragment: a fragment carries the policy that produced it, so two definitions of a
threshold would produce two contracts differing for a reason that says nothing about
the data, and whichever ran last would win. Every threshold therefore lives in
`R/policy.R` and nowhere else. A number appearing in a notebook and not in that file
is a bug, not a local override.

`load_windows` sorts every window on the time axis before returning it, and that is
load-bearing rather than tidy. A filtered Arrow scan is multi-threaded and does not
promise file order, so two runs get the same rows in a different sequence — invisible
to anything that reads every row, and fatal to anything that subsamples, because a
seeded `sample.int` then picks the same positions out of a different ordering. "Seeded"
and "reproducible" are not the same claim.

The input is the frame *as the model receives it* — `features.model_input` with the
declared derivations applied — which `uv run export-audit-frame` produces. Auditing the
raw export instead would leave the thirty derived columns unexamined, and eight of them
were rejected the last time the whole set was measured. Point `FRAUDAUDIT_PARQUET`
elsewhere if it is not at the default path. Data is not committed; see `docs/setup.md`.

The contract this produces is stamped by `uv run stamp-contract`, which is the only
thing that crosses back into Python.

## Layout

| | |
| --- | --- |
| `R/io.R` | window cutting, including the quantile convention the migration turned on |
| `R/binning.R` | pinned bins, shared with the drift audit so PSI and WoE compare like with like |
| `R/woe.R` | weight of evidence, information value, and the reversal statistics |
| `R/association.R` | Somers' D, DeLong intervals, the two-sample AUC test, BH |
| `R/time_consistency.R` | the audit and its two verdicts |
| `R/distribution_shift.R` | PSI, its null distribution, marginal and multivariate tests |
| `R/redundancy.R` | rank-correlation clustering, redundancy analysis, block audit |
| `R/segment_qualification.R` | risk dichotomy, CMH, Breslow–Day |
| `R/categorical.R` | Cramér's V, bootstrapped on the contingency table |
| `R/missingness.R` | missingness families, Little's MCAR against always-observed columns |
| `R/dimensionality.R` | parallel analysis, tetrachoric correlation |
| `R/entity.R` | entity keys, purity against a permuted null |
| `R/policy.R` | every threshold, defined once for the graph and the notebooks alike |
| `R/contract.R` | the merge, and nothing else |
| `R/fragments.R` | what leaves the package, and the V-block expansion |
