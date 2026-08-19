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

| Notebook | Question | Method | Feeds contract |
| --- | --- | --- | --- |
| `does-a-feature-still-mean-the-same-later` | Has a column reversed between an early and a late window? | Weight of evidence, Somers' D, DeLong intervals and test | yes |
| `has-the-population-moved` | Did the distribution move, and are the periods distinguishable jointly? | PSI against a measured null, Anderson–Darling, energy test | yes |
| `which-columns-say-the-same-thing-twice` | Which columns restate their neighbours? | Variable clustering on rank correlation, redundancy analysis on splines | yes |
| `does-a-column-work-inside-every-segment` | Does a pooled association survive conditioning on the product? | Cochran–Mantel–Haenszel, Breslow–Day | recorded, not applied |
| `what-the-columns-are-made-of` | Does a column know more about the period than about fraud; how many dimensions does a V block have; is absence random? | Cramér's V with a bootstrap interval, Horn's parallel analysis, tetrachoric correlation, Little's MCAR | no, report-only |
| `who-is-the-customer-when-the-data-does-not-say` | Is the reconstructed entity real, and which key? | Label purity against a permuted null, coverage, concentration | the contract's `entity` block |
| `what-the-fraud-literature-asks…` | What is true of the phenomenon rather than of a column? | Benford, power-law tail index, Bai–Perron breaks, Rayleigh, Gini/HHI | no, report-only |
| `what-does-this-data-look-like` | How much fraud, where, and are the columns the type they claim? | Wilson intervals on every rate, weight of evidence on the amount, cardinality against declared type | no, report-only |

`build-contract.qmd` merges the fragments, and asks two questions about the result
itself: whether the admitted set carries more information value than random sets of the
same size, and how much of it survives moving the drift threshold.

No audit trains a model. Where the literature reaches for one — a single-feature
fit to score a column, a classifier to tell two periods apart — a ten-bin
scorecard and a permutation two-sample test answer the same question and print.

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
Rscript -e 'targets::tar_make()'            # the audit graph
Rscript -e 'targets::tar_visnetwork()'      # draw it
quarto render notebooks/                    # the readable version
quarto render build-contract.qmd            # the merge
Rscript -e 'testthat::test_dir("tests/testthat")'
```

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
