# Does a feature still mean the same thing next month?
#
# The experiment: fit a non-parametric ten-bin scorecard on an early window,
# apply it unchanged to a late one. A feature whose ranking inverts between the
# two found a pattern in the present that does not exist in the future, and
# admitting it trains the model on a relationship that has already expired.
#
# Why a scorecard and not a model. Binning plus weight of evidence is a
# learned, possibly non-monotone mapping from values to a fraud ordering -- the
# same thing a single-feature tree learns -- and it prints as eleven rows. For a
# decision that removes a column from a model somebody has to approve, an answer
# that can be read beats one that cannot, and it costs nothing: the AUC of a
# WoE-scored window is exactly the Mann--Whitney statistic either way.

TC_DEFAULTS <- list(
  bins           = 10L,
  max_levels     = 20L,
  margin         = 0.02,
  min_rows       = 100L,
  min_positives  = 10L,
  conf_level     = 0.95,
  alpha          = 0.05,
  min_gini_drop  = 0.04,
  woe_eps        = 0.05
)

VERDICT <- c(PASS = "pass", INVERTED = "inverted", WEAK = "weak", DEGENERATE = "degenerate")

.degenerate_row <- function(feature, note, x_tr, x_ho) {
  data.table::data.table(
    feature = feature, verdict = VERDICT[["DEGENERATE"]],
    auc_train = NA_real_, auc_holdout = NA_real_, delta = NA_real_,
    auc_train_lo = NA_real_, auc_train_hi = NA_real_,
    auc_holdout_lo = NA_real_, auc_holdout_hi = NA_real_,
    dxy_train = NA_real_, dxy_holdout = NA_real_,
    iv_train = NA_real_, iv_holdout = NA_real_,
    woe_rho = NA_real_, woe_flip_mass = NA_real_,
    p_delong = NA_real_, unstable = NA,
    n_train = length(x_tr), n_holdout = length(x_ho),
    null_rate_train = if (length(x_tr)) mean(is.na(x_tr)) else 1,
    null_rate_holdout = if (length(x_ho)) mean(is.na(x_ho)) else 1,
    note = note
  )
}

#' Check one feature across two windows.
#'
#' @param x_tr,x_ho The column, in the early and late window.
#' @param y_tr,y_ho The label, same rows.
#' @param feature Column name, carried into the result.
#' @param opts Overrides for [TC_DEFAULTS].
#' @return A one-row data.table.
time_consistency_feature <- function(x_tr, y_tr, x_ho, y_ho, feature, opts = list()) {
  o <- utils::modifyList(TC_DEFAULTS, opts)

  # Ordered from cheapest to most specific, so the reason a column is called
  # degenerate is the first thing actually wrong with it rather than whichever
  # check happened to run first.
  if (length(x_tr) < o$min_rows || length(x_ho) < o$min_rows) {
    return(.degenerate_row(feature, sprintf("window smaller than min_rows=%d", o$min_rows), x_tr, x_ho))
  }
  if (length(unique(stats::na.omit(y_tr))) < 2L || length(unique(stats::na.omit(y_ho))) < 2L) {
    return(.degenerate_row(feature, "a window has only one class", x_tr, x_ho))
  }
  if (min(sum(y_tr == 1L, na.rm = TRUE), sum(y_ho == 1L, na.rm = TRUE)) < o$min_positives) {
    return(.degenerate_row(feature, sprintf("fewer than min_positives=%d", o$min_positives), x_tr, x_ho))
  }
  if (all(is.na(x_tr))) {
    return(.degenerate_row(feature, "all-null in the training window", x_tr, x_ho))
  }
  if (length(unique(stats::na.omit(x_tr))) < 2L) {
    return(.degenerate_row(feature, "constant in the training window", x_tr, x_ho))
  }

  binning <- fit_binning(x_tr, bins = o$bins, max_levels = o$max_levels)
  if (!is.na(binning$degenerate)) {
    return(.degenerate_row(feature, binning$degenerate, x_tr, x_ho))
  }

  b_tr <- assign_bins(binning, x_tr)
  b_ho <- assign_bins(binning, x_ho)
  tab_tr <- woe_table(b_tr, y_tr)
  tab_ho <- woe_table(b_ho, y_ho)

  # The scorecard is learned once, on the early window, and both windows are
  # scored with it. Scoring the late window with its own weights would ask a
  # different and much weaker question -- "is there any signal later" rather
  # than "does the signal found earlier still hold".
  s_tr <- woe_score(tab_tr, b_tr)
  s_ho <- woe_score(tab_tr, b_ho)

  a_tr <- auc_ci_delong(s_tr, y_tr, conf_level = o$conf_level)
  a_ho <- auc_ci_delong(s_ho, y_ho, conf_level = o$conf_level)
  if (is.na(a_tr$auc) || is.na(a_ho$auc)) {
    return(.degenerate_row(feature, "AUC not estimable on the scored window", x_tr, x_ho))
  }

  test <- delong_two_sample(a_tr$roc, a_ho$roc)
  agree <- woe_agreement(tab_tr, tab_ho, eps = o$woe_eps)
  d_tr <- somers_dxy(x_tr, y_tr)
  d_ho <- somers_dxy(x_ho, y_ho)

  data.table::data.table(
    feature = feature, verdict = NA_character_,
    auc_train = round(a_tr$auc, 4), auc_holdout = round(a_ho$auc, 4),
    delta = round(a_ho$auc - a_tr$auc, 4),
    auc_train_lo = round(a_tr$lower, 4), auc_train_hi = round(a_tr$upper, 4),
    auc_holdout_lo = round(a_ho$lower, 4), auc_holdout_hi = round(a_ho$upper, 4),
    dxy_train = round(d_tr$dxy, 4), dxy_holdout = round(d_ho$dxy, 4),
    iv_train = round(information_value(tab_tr), 4),
    iv_holdout = round(information_value(tab_ho), 4),
    woe_rho = round(agree$rho, 4), woe_flip_mass = round(agree$flip_mass, 4),
    p_delong = test$p_value, unstable = NA,
    n_train = length(x_tr), n_holdout = length(x_ho),
    null_rate_train = round(mean(is.na(x_tr)), 4),
    null_rate_holdout = round(mean(is.na(x_ho)), 4),
    note = ""
  )
}

#' Scan every feature, then apply the multiplicity correction across the scan.
#'
#' The correction cannot live inside the per-feature function: Benjamini--
#' Hochberg is a statement about a family of tests, and the family is the scan.
#' This is also where the strict verdict is decided, for the same reason -- it
#' depends on an adjusted p-value that does not exist until every column is in.
#'
#' The verdict is a pair of tests, not a threshold on a point estimate: the later
#' AUC must lie significantly past 0.5 by a DeLong interval, the difference must
#' survive the family-wide correction, and the loss of discrimination must clear
#' `min_gini_drop`.
#'
#' A fixed margin around 0.5 would not do: at 100k rows per window a DeLong
#' interval is about 0.005 wide, so a margin of 0.02 is four standard errors and
#' passes every real inversion smaller than itself. Testing instead rejects
#' *more* columns, which is why materiality has to be the second key --
#' significance alone, at this sample size, would reject the table.
time_consistency_scan <- function(win, features = NULL, opts = list(), verbose = TRUE) {
  o <- utils::modifyList(TC_DEFAULTS, opts)
  features <- features %||% win$features
  y_tr <- as.integer(win$train[[win$label]])
  y_ho <- as.integer(win$holdout[[win$label]])

  rows <- vector("list", length(features))
  for (i in seq_along(features)) {
    f <- features[[i]]
    rows[[i]] <- time_consistency_feature(
      win$train[[f]], y_tr, win$holdout[[f]], y_ho, f, opts = opts
    )
    if (verbose && i %% 25L == 0L) message(sprintf("  %d/%d", i, length(features)))
  }
  out <- data.table::rbindlist(rows)

  out[, p_delong_bh := adjust_bh(p_delong)]
  out[, verdict := data.table::fifelse(
    !is.na(note) & note != "", VERDICT[["DEGENERATE"]],
    data.table::fifelse(
      auc_train_lo > 0.5 & auc_holdout_hi < 0.5 &
        !is.na(p_delong_bh) & p_delong_bh < o$alpha &
        (2 * auc_train - 1) - (2 * auc_holdout - 1) >= o$min_gini_drop,
      VERDICT[["INVERTED"]],
      data.table::fifelse(
        auc_train_lo <= 0.5 & auc_train_hi >= 0.5 & auc_holdout_lo <= 0.5 & auc_holdout_hi >= 0.5,
        VERDICT[["WEAK"]], VERDICT[["PASS"]]
      )
    )
  )]

  out[, unstable := !is.na(p_delong_bh) & p_delong_bh < o$alpha &
        verdict != VERDICT[["INVERTED"]] &
        (2 * auc_train - 1) - (2 * auc_holdout - 1) >= o$min_gini_drop]

  data.table::setorder(out, -woe_flip_mass, feature)
  out[]
}

`%||%` <- function(a, b) if (is.null(a)) b else a
