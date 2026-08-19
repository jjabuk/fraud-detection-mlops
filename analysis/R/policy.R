# The settings every audit runs under, in one place.
#
# Both the `targets` graph and the notebooks write contract fragments, and they
# have to write the *same* fragment: a fragment carries the policy that produced
# it, so two definitions of a threshold produce two contracts that differ for a
# reason that says nothing about the data. Whichever ran last would win, silently.
#
# So the policy is defined once, here, and sourced by both. A threshold that
# appears in a notebook and not in this file is a bug, not a local override.
#
# These are also what gets recorded into `references/feature-contract.json`. An
# audit threshold can move without the admitted set moving, and then two models
# pinned to "the same" contract would have been evaluated under different rules --
# which is why the fingerprint covers the policy and not only the columns.

#: Windows. The gap between them is deliberate: skipping the middle mimics the
#: label-maturity lag at deployment, where the most recent period has not
#: finished being labelled.
AUDIT_WINDOWS <- list(train = c(0.0, 0.17), holdout = c(0.83, 1.0))

TC_POLICY <- list(
  train_window    = AUDIT_WINDOWS$train,
  holdout_window  = AUDIT_WINDOWS$holdout,
  bins            = 10L,
  margin          = 0.02,
  alpha           = 0.05,
  min_gini_drop   = 0.04,
  reject_by_block = TRUE
)

DS_POLICY <- list(
  bins          = 10L,
  permutations  = 200L,
  # Materiality, not significance. The measured noise floor at these sample sizes
  # is around 0.0004, so this threshold is roughly 700 times it -- it is a
  # statement about what is worth acting on, and the permutation p-value is what
  # rules out movement indistinguishable from a coin flip.
  psi_threshold = 0.25,
  alpha         = 0.05
)

RD_POLICY <- list(
  min_rho2       = 0.8,
  redun_r2       = 0.9,
  rank_subsample = 20000L
)

SQ_POLICY <- list(
  segment_column = "ProductCD",
  bins           = 10L,
  # Report-only, and measured rather than cautious: applying these verdicts cost
  # 0.0325 PR-AUC and moved the segment they were protecting by 0.0005.
  reject         = FALSE
)

#: The order checks are credited in. It does not change which columns are
#: admitted -- a column caught twice is out either way -- but it decides which
#: check the report attributes each rejection to. Must match `PRECEDENCE` in
#: `fraud_detection.cli.stamp_contract`.
PRECEDENCE <- c("time_consistency", "distribution_shift", "redundancy",
                "segment_qualification")
