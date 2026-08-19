# Does a column that scores well pooled still score inside each segment?
#
# The failure this exists to catch is Simpson's paradox: a column can separate
# fraud beautifully across the whole table and separate nothing at all inside
# every product segment, because what it is really encoding is *which segment a
# row belongs to*. A model built on it inherits the segment mix of the training
# period, and the day the mix moves the column stops working -- with no drift in
# the column itself to warn anyone.
#
# Comparing a pooled AUC with a within-segment AUC and reading the gap against a
# threshold gets the right quantity and stops short of testing it. Two textbook
# procedures do the job properly and both are a century old:
#
#   Cochran--Mantel--Haenszel  is the association still there once segment is
#                              held fixed?
#   Breslow--Day               is it the *same* association in every segment,
#                              or does it differ between them?
#
# The first catches a column that only looked predictive because it tracked the
# segment. The second catches a column that predicts in opposite directions in
# two segments and nets out to something respectable when pooled.
#
# References: Mantel & Haenszel (1959), *JNCI* 22(4); Breslow & Day (1980),
# *Statistical Methods in Cancer Research I*, section 4.4; Agresti (2013),
# *Categorical Data Analysis*, ch. 6.3.

#' Split a column into higher- and lower-risk halves by weight of evidence.
#'
#' A 2x2xK table needs a binary exposure, and the honest dichotomy is the one
#' the data draws: bins whose weight of evidence is positive carry above-average
#' fraud odds, bins whose WoE is negative carry below-average odds. Cutting at
#' the median instead would split on population rather than on risk and would
#' put the interesting bins on both sides of the line.
#'
#' @return Logical vector, `NA` where the column is null.
risk_dichotomy <- function(x, y, bins = 10L, max_levels = 20L) {
  binning <- fit_binning(x, bins = bins, max_levels = max_levels)
  if (!is.na(binning$degenerate)) return(rep(NA, length(x)))
  assigned <- assign_bins(binning, x)
  tab <- woe_table(assigned, y)
  high <- as.character(tab$bin[tab$woe > 0 & tab$n > 0])
  out <- as.character(assigned) %in% high
  out[is.na(x)] <- NA
  out
}

#' Test one column against one segmentation.
#'
#' Both tests need every stratum to hold both outcomes and both exposure levels;
#' strata that do not are dropped and counted, rather than being collapsed into
#' a neighbour, because collapsing strata is how a segment effect gets hidden.
segment_tests <- function(x, y, segment, bins = 10L) {
  high <- risk_dichotomy(x, y, bins = bins)
  ok <- !is.na(high) & !is.na(y) & !is.na(segment)
  if (sum(ok) < 100L) {
    return(list(cmh_or = NA_real_, cmh_p = NA_real_, bd_p = NA_real_,
                strata_used = 0L, strata_dropped = NA_integer_, note = "too few usable rows"))
  }

  tab <- table(exposure = high[ok], outcome = y[ok], stratum = as.character(segment[ok]))
  usable <- apply(tab, 3, function(s) all(dim(s) == c(2L, 2L)) && all(rowSums(s) > 0) && all(colSums(s) > 0))
  dropped <- sum(!usable)
  tab <- tab[, , usable, drop = FALSE]

  if (dim(tab)[[3]] < 2L) {
    return(list(cmh_or = NA_real_, cmh_p = NA_real_, bd_p = NA_real_,
                strata_used = dim(tab)[[3]], strata_dropped = dropped,
                note = "fewer than two usable strata"))
  }

  cmh <- suppressWarnings(try(stats::mantelhaen.test(tab, correct = FALSE), silent = TRUE))
  bd <- suppressWarnings(try(DescTools::BreslowDayTest(tab), silent = TRUE))

  list(
    cmh_or = if (inherits(cmh, "try-error")) NA_real_ else unname(cmh$estimate),
    cmh_p  = if (inherits(cmh, "try-error")) NA_real_ else unname(cmh$p.value),
    bd_p   = if (inherits(bd, "try-error")) NA_real_ else unname(bd$p.value),
    strata_used = dim(tab)[[3]], strata_dropped = dropped, note = ""
  )
}

#' Score every column pooled and inside every segment, and test the difference.
#'
#' The pooled row is kept in the same table on purpose: the finding is never a
#' within-segment number on its own, it is the *gap* between it and the pooled
#' one, and a reader should not have to join two files to see it.
segment_qualification_scan <- function(win, features = NULL, segment_column = "ProductCD",
                                       bins = 10L, verbose = TRUE) {
  features <- setdiff(features %||% win$features, segment_column)
  y <- as.integer(win$train[[win$label]])
  segment <- as.character(win$train[[segment_column]])
  levels_ <- sort(unique(stats::na.omit(segment)))

  # Which segment the verdict is read against. The largest one, because a
  # failure there is a failure for most of the traffic, and a model judged on a
  # pooled metric is judged mostly on this segment anyway.
  dominant <- names(sort(table(segment), decreasing = TRUE))[[1]]

  rows <- vector("list", length(features))
  for (i in seq_along(features)) {
    f <- features[[i]]
    x <- win$train[[f]]
    pooled <- column_auc_directionless(x, y, rep(TRUE, length(x)))

    per_segment <- vapply(levels_, function(lv) {
      column_auc_directionless(x, y, segment == lv & !is.na(segment))
    }, numeric(1))

    tests <- segment_tests(x, y, segment, bins = bins)

    # An unmeasurable dominant segment is a result, not a gap in the data. A
    # column that varies only outside the segment carrying most of the traffic
    # is, in that segment, a constant -- and its pooled association is then
    # substantially a statement about which segment the row belongs to. That is
    # precisely the confound this audit is here to name, so it is recorded
    # rather than dropped.
    rows[[i]] <- data.table::data.table(
      column = f, pooled_auc = pooled,
      dominant_segment = dominant, dominant_auc = per_segment[[dominant]],
      measurable_in_dominant = !is.na(per_segment[[dominant]]),
      segments_measurable = sum(!is.na(per_segment)),
      segments_total = length(per_segment),
      min_segment_auc = suppressWarnings(min(per_segment, na.rm = TRUE)),
      max_segment_auc = suppressWarnings(max(per_segment, na.rm = TRUE)),
      gap_dominant = pooled - per_segment[[dominant]],
      cmh_or = tests$cmh_or, cmh_p = tests$cmh_p, bd_p = tests$bd_p,
      strata_used = tests$strata_used, note = tests$note
    )
    if (verbose && i %% 50L == 0L) message(sprintf("  %d/%d", i, length(features)))
  }

  out <- data.table::rbindlist(rows)
  out[is.infinite(min_segment_auc), min_segment_auc := NA_real_]
  out[is.infinite(max_segment_auc), max_segment_auc := NA_real_]
  out[, `:=`(cmh_p_bh = adjust_bh(cmh_p), bd_p_bh = adjust_bh(bd_p))]

  # Two named failures rather than one score. `segment_artefact` is a column
  # whose pooled association does not survive conditioning on the segment;
  # `heterogeneous` is one whose association is real but different in different
  # segments -- the second is not a reason to drop a column, it is a reason to
  # stop reading its pooled number.
  out[, segment_artefact :=
        (!is.na(cmh_p_bh) & cmh_p_bh > 0.05 & pooled_auc > 0.55) |
        (!measurable_in_dominant & !is.na(pooled_auc) & pooled_auc > 0.55)]
  out[, heterogeneous := !is.na(bd_p_bh) & bd_p_bh < 0.05]
  data.table::setorder(out, -gap_dominant)
  out[]
}

#' Direction-free single-feature AUC over a masked subset.
#'
#' `max(auc, 1 - auc)` because the question here is how much a column separates,
#' not which way round it does so -- the direction question belongs to the time
#' consistency audit, which is the only place it means anything.
column_auc_directionless <- function(x, y, mask) {
  if (is.logical(x)) x <- as.integer(x)
  if (!is.numeric(x)) {
    b <- fit_binning(x)
    if (!is.na(b$degenerate)) return(NA_real_)
    x <- woe_score(woe_table(assign_bins(b, x), y), assign_bins(b, x))
  }
  usable <- mask & !is.na(x) & !is.na(y)
  if (sum(usable) < 100L || sum(y[usable] == 1L) < 10L ||
      length(unique(x[usable])) < 2L) return(NA_real_)
  a <- auc_ci_delong(x[usable], y[usable])$auc
  if (is.na(a)) NA_real_ else max(a, 1 - a)
}

#' Build the segment-qualification fragment.
#'
#' Report-only by default, and that is a measured decision rather than caution:
#' applying these verdicts cost 0.0325 PR-AUC and moved the segment they were
#' protecting by 0.0005. The fragment records what was found and rejects
#' nothing, so the finding survives without the cost.
segment_qualification_fragment <- function(report, reject = FALSE, params = list()) {
  flagged <- report[segment_artefact == TRUE]
  list(
    check = "segment_qualification",
    tool = "fraudaudit::segment_qualification_scan",
    params = utils::modifyList(as.list(params), list(reject = reject)),
    qualification = list(
      columns_scanned = nrow(report),
      segment_artefacts = nrow(flagged),
      heterogeneous = sum(report$heterogeneous, na.rm = TRUE),
      # Columns that look predictive pooled and are a constant inside the
      # segment carrying most of the traffic.
      unmeasurable_in_dominant = sum(!report$measurable_in_dominant &
                                       report$pooled_auc > 0.55, na.rm = TRUE),
      untestable = sum(is.na(report$cmh_p)),
      median_gap_dominant = stats::median(report$gap_dominant, na.rm = TRUE)
    ),
    rejections = if (!reject) list() else lapply(seq_len(nrow(flagged)), function(i) list(
      column = flagged$column[[i]], check = "segment_qualification",
      value = flagged$gap_dominant[[i]], unit = "column"
    ))
  )
}
