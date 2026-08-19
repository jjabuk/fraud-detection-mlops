# Weight of evidence, and the information value it sums to.
#
# This is the credit-scorecard reading of "does this column carry signal", and
# it is the reason this package needs no model to answer the question a
# single-feature gradient boosting fit was answering before. WoE states, per
# bin, how much the fraud odds inside that bin depart from the population odds.
# A rejection is then a sentence -- "in this bin the odds ran 2.4x the base rate
# early and 0.6x late" -- rather than a number that came out of a fit.
#
# References: Siddiqi (2017), *Intelligent Credit Scoring*, ch. 6; Thomas,
# Edelman & Crook (2002), *Credit Scoring and Its Applications*, ch. 8.

#' Weight of evidence and information value over a fitted binning.
#'
#' Zero cells get the Haldane--Anscombe correction of 0.5 rather than an
#' epsilon: an empty bin is a small-sample problem, and 0.5 is the correction
#' with a Bayesian reading (a Jeffreys prior on the bin's odds) instead of an
#' arbitrary floor. Without it a bin holding no fraud returns an infinite WoE
#' and an infinite information value follows it into the verdict.
#'
#' @param bins Factor from [assign_bins()].
#' @param y Integer/logical label, 1 = fraud.
#' @return A data.table, one row per bin level, ordered as the binning is.
woe_table <- function(bins, y) {
  stopifnot(is.factor(bins), length(bins) == length(y))
  y <- as.integer(y)

  dt <- data.table::data.table(bin = bins, y = y)
  tab <- dt[, .(n = .N, pos = sum(y == 1L), neg = sum(y == 0L)), keyby = bin]
  tab <- tab[data.table::CJ(bin = levels(bins), unique = TRUE), on = "bin"]
  tab[is.na(n), `:=`(n = 0L, pos = 0L, neg = 0L)]

  # Haldane--Anscombe, applied only where a cell is empty, so a bin with
  # adequate counts keeps its unshrunk estimate.
  tab[, `:=`(pos_adj = ifelse(pos == 0L | neg == 0L, pos + 0.5, as.numeric(pos)),
             neg_adj = ifelse(pos == 0L | neg == 0L, neg + 0.5, as.numeric(neg)))]

  n_pos <- sum(tab$pos_adj)
  n_neg <- sum(tab$neg_adj)
  tab[, `:=`(p_pos = pos_adj / n_pos, p_neg = neg_adj / n_neg)]
  tab[, woe := log(p_pos / p_neg)]
  tab[, iv_part := (p_pos - p_neg) * woe]
  tab[, share := n / sum(n)]
  tab[, bin := factor(as.character(bin), levels = levels(bins))]

  data.table::setorder(tab, bin)
  tab[, .(bin, n, pos, neg, share, p_pos, p_neg, woe, iv_part)]
}

#' Information value: the total divergence between the fraud and non-fraud
#' distributions over the bins.
#'
#' The conventional bands (below 0.02 useless, 0.02--0.1 weak, 0.1--0.3 medium,
#' above 0.3 strong) are folklore with a long service record, not a test. They
#' are recorded next to the number and never used alone to reject: an IV has no
#' sampling distribution here, which is why the verdict in [time_consistency()]
#' rests on a rank statistic with a confidence interval.
information_value <- function(woe_tab) sum(woe_tab$iv_part)

#' Score a window with weights of evidence learned on another one.
#'
#' This is the whole time-consistency experiment in one line: fit a non-
#' parametric ten-bin scorecard on the early window, apply it unchanged to the
#' late one. Because the mapping is a lookup table rather than a fit, "applied
#' unchanged" is literally true and can be printed.
#'
#' @param woe_tab Table from [woe_table()], fitted on the reference window.
#' @param bins Factor from [assign_bins()] over the window being scored.
#' @return Numeric vector of weights of evidence, one per row.
woe_score <- function(woe_tab, bins) {
  lookup <- stats::setNames(woe_tab$woe, as.character(woe_tab$bin))
  unname(lookup[as.character(bins)])
}

#' Mass-weighted agreement between two weight-of-evidence vectors.
#'
#' Two numbers, because they answer different halves of "did the ordering
#' hold": `rho` is the Spearman correlation of the two WoE vectors across bins,
#' which catches a reshuffle; `flip_mass` is the share of the later window
#' sitting in bins whose WoE changed sign, which catches the case where the
#' ordering is intact everywhere except in the bins that carry the population.
#'
#' A bin is only counted as flipped when both sides clear `eps` -- a WoE
#' wobbling around zero has no direction to reverse.
#'
#' Bins that are empty on either side are dropped before the correlation. Their
#' WoE is the Haldane correction alone, which is zero by construction, and a
#' vector padded with structural zeros correlates with itself no matter what
#' the populated bins did.
woe_agreement <- function(ref_tab, cur_tab, eps = 0.05) {
  stopifnot(identical(as.character(ref_tab$bin), as.character(cur_tab$bin)))
  usable <- is.finite(ref_tab$woe) & is.finite(cur_tab$woe) &
    ref_tab$n > 0L & cur_tab$n > 0L
  rho <- if (sum(usable) >= 3L) {
    suppressWarnings(stats::cor(ref_tab$woe[usable], cur_tab$woe[usable], method = "spearman"))
  } else NA_real_

  flipped <- usable &
    abs(ref_tab$woe) > eps & abs(cur_tab$woe) > eps &
    sign(ref_tab$woe) != sign(cur_tab$woe)

  list(rho = rho, flip_mass = sum(cur_tab$share[flipped]))
}
