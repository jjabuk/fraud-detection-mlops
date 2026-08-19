# Two questions about distribution shift that are usually confused for one.
#
# PSI asks whether a column's *marginal* distribution moved. A multivariate
# two-sample test asks whether the two periods are distinguishable *at all*,
# which is not the same question -- a column can score low on the first and the
# table still score high on the second, because the shift lives in the joint
# distribution rather than in any margin.
#
# Adversarial validation answers the second by training a model to tell the
# periods apart and reading its AUC -- a two-sample test with a model inside it.
# The energy test answers the same question directly: a p-value instead of an
# AUC, and a null distribution instead of a convention.
#
# The binning is the one from binning.R, fitted on the reference window and
# carried unchanged. PSI computed against re-derived edges compares two
# different histograms and reports the difference as stability.
#
# References: Szekely & Rizzo (2004), *InterStat*, on the energy test;
# Anderson & Darling (1952), *Ann. Math. Statist.* 23(2); Siddiqi (2017) ch. 7
# on PSI and its conventional bands.

#' Population stability index between two binned windows.
#'
#' The conventional reading -- below 0.1 no shift, 0.1 to 0.25 moderate, above
#' 0.25 significant -- is folklore with a long service record and no sampling
#' distribution. It is reported, and it is never the whole verdict: what the
#' index *is* at a given sample size under no shift at all is measured, not
#' assumed. See [psi_null()].
#'
#' @param ref_counts,cur_counts Integer counts over the same bin levels.
#' @param eps Floor on a share. PSI divides by the reference share, so an empty
#'   reference bucket would otherwise return infinity for a single stray row.
psi_value <- function(ref_counts, cur_counts, eps = 1e-6) {
  stopifnot(length(ref_counts) == length(cur_counts))
  p_ref <- pmax(ref_counts / sum(ref_counts), eps)
  p_cur <- pmax(cur_counts / sum(cur_counts), eps)
  sum((p_cur - p_ref) * log(p_cur / p_ref))
}

#' The distribution of PSI when nothing has moved.
#'
#' This is the number the conventional thresholds leave out. Pool the two
#' windows, split them at random into the original sizes, and recompute: any PSI
#' that survives is sampling noise, because the split was random by
#' construction. Repeating it gives the null distribution of the index at *these
#' sample sizes and this bin count*, which is what a threshold has to be read
#' against.
#'
#' The permutation is never actually performed. PSI depends on the two windows
#' only through their bin counts, and the counts of a random split of a pooled
#' sample follow a multivariate hypergeometric distribution exactly. Drawing
#' from it costs one `rhyper` per bin -- eleven or twenty numbers -- instead of
#' shuffling 200,000 row indices, which is the difference between eleven minutes
#' and four seconds over this table. The null is identical, not approximated.
#'
#' @return List with the observed value, a permutation p-value and the 99th
#'   percentile of the null.
psi_null <- function(bins_ref, bins_cur, B = 200L, seed = 0L) {
  k <- nlevels(bins_ref)
  c_ref <- tabulate(bins_ref, nbins = k)
  c_cur <- tabulate(bins_cur, nbins = k)
  obs <- psi_value(c_ref, c_cur)

  pooled <- c_ref + c_cur
  n_ref <- sum(c_ref)
  set.seed(seed)

  draws_ref <- matrix(0L, nrow = B, ncol = k)
  remaining_total <- sum(pooled)
  remaining_ref <- rep(n_ref, B)
  for (i in seq_len(k)) {
    taken <- stats::rhyper(B, m = pooled[[i]], n = remaining_total - pooled[[i]],
                           k = remaining_ref)
    draws_ref[, i] <- taken
    remaining_total <- remaining_total - pooled[[i]]
    remaining_ref <- remaining_ref - taken
  }

  draws <- vapply(seq_len(B), function(b) {
    psi_value(draws_ref[b, ], pooled - draws_ref[b, ])
  }, numeric(1))

  list(
    psi = obs,
    p_value = (1 + sum(draws >= obs)) / (B + 1),
    null_q99 = unname(stats::quantile(draws, 0.99)),
    null_median = stats::median(draws)
  )
}

#' Per-column shift report over the two windows.
#'
#' Every column gets four numbers and they are not interchangeable: the index,
#' what the index would be under no shift, whether the observed value is
#' distinguishable from that, and how much of the movement is missingness rather
#' than values. The last one matters here because a third of this table is more
#' than half null, and a column whose *rate of being populated* changed is a
#' different finding from a column whose values moved.
distribution_shift_scan <- function(win, features = NULL, B = 200L,
                                    bins = 10L, max_levels = 20L, verbose = TRUE) {
  features <- features %||% win$features
  rows <- vector("list", length(features))

  for (i in seq_along(features)) {
    f <- features[[i]]
    x_ref <- win$train[[f]]; x_cur <- win$holdout[[f]]
    binning <- fit_binning(x_ref, bins = bins, max_levels = max_levels)

    if (!is.na(binning$degenerate)) {
      rows[[i]] <- data.table::data.table(
        column = f, psi = NA_real_, psi_null_q99 = NA_real_, p_perm = NA_real_,
        null_share_ref = mean(is.na(x_ref)), null_share_cur = mean(is.na(x_cur)),
        null_share_delta = mean(is.na(x_cur)) - mean(is.na(x_ref)),
        psi_excl_null = NA_real_, top_bucket = NA_character_, note = binning$degenerate
      )
      next
    }

    b_ref <- assign_bins(binning, x_ref)
    b_cur <- assign_bins(binning, x_cur)
    nul <- psi_null(b_ref, b_cur, B = B, seed = i)

    # The same index with the null bucket dropped. When the two differ sharply
    # the movement is in how often the column is populated, not in what it says
    # when it is -- and those call for different responses.
    keep <- levels(b_ref) != NULL_BUCKET
    cr <- tabulate(b_ref, nbins = nlevels(b_ref))[keep]
    cc <- tabulate(b_cur, nbins = nlevels(b_cur))[keep]
    psi_ex <- if (sum(cr) > 0 && sum(cc) > 0) psi_value(cr, cc) else NA_real_

    contrib <- {
      lv <- levels(b_ref)
      p_r <- pmax(tabulate(b_ref, nbins = length(lv)) / length(b_ref), 1e-6)
      p_c <- pmax(tabulate(b_cur, nbins = length(lv)) / length(b_cur), 1e-6)
      lv[[which.max((p_c - p_r) * log(p_c / p_r))]]
    }

    rows[[i]] <- data.table::data.table(
      column = f, psi = nul$psi, psi_null_q99 = nul$null_q99, p_perm = nul$p_value,
      null_share_ref = mean(is.na(x_ref)), null_share_cur = mean(is.na(x_cur)),
      null_share_delta = mean(is.na(x_cur)) - mean(is.na(x_ref)),
      psi_excl_null = psi_ex, top_bucket = contrib, note = ""
    )
    if (verbose && i %% 50L == 0L) message(sprintf("  %d/%d", i, length(features)))
  }

  out <- data.table::rbindlist(rows)
  out[, p_perm_bh := adjust_bh(p_perm)]
  # How many times the index exceeds what pure noise produces at this n. A
  # ratio, because the raw index is not comparable across columns with different
  # bin counts and the ratio is.
  out[, psi_over_null := psi / psi_null_q99]
  data.table::setorder(out, -psi)
  out[]
}

#' Are the two windows distinguishable jointly?
#'
#' The energy statistic is a weighted L2 distance between the two empirical
#' characteristic functions; the test is a permutation test on it, so the
#' p-value needs no distributional assumption. This is the adversarial
#' validation answer without the adversary.
#'
#' Subsampled, and not for convenience: the statistic is O(n^2) in memory. The
#' sample size is the one at which the test has adequate power against the
#' effect sizes worth acting on -- see the power curve in the notebook -- rather
#' than the largest that happens to fit.
#'
#' **Never pass the time column.** With `TransactionDT` in the matrix any
#' discriminator separates the two periods perfectly, because the periods are
#' *defined* by it -- and the result is a p-value of zero that says nothing about
#' the features. This is where the folklore that "adversarial AUC is 1 on this
#' competition" partly comes from. `load_windows` excludes the time axis from
#' `features` for exactly this reason, which is why the guard is a convention
#' here rather than a check.
#'
#' @param columns Numeric columns only. Rows with any missing value are dropped,
#'   which is why this is run on a screened subset rather than the whole table.
multivariate_shift <- function(win, columns, n_sub = 1500L, R = 199L, seed = 0L) {
  ref <- stats::na.omit(as.matrix(win$train[, ..columns]))
  cur <- stats::na.omit(as.matrix(win$holdout[, ..columns]))
  if (nrow(ref) < 50L || nrow(cur) < 50L) {
    return(list(statistic = NA_real_, p_value = NA_real_, n_ref = nrow(ref), n_cur = nrow(cur)))
  }

  set.seed(seed)
  ref <- ref[sample.int(nrow(ref), min(n_sub, nrow(ref))), , drop = FALSE]
  cur <- cur[sample.int(nrow(cur), min(n_sub, nrow(cur))), , drop = FALSE]

  # Standardised on the reference window's scale. Without it the statistic is
  # dominated by whichever column happens to have the largest units, and the
  # test reports the shift in that column as a shift in the table.
  centre <- colMeans(ref); scale_ <- apply(ref, 2, stats::sd)
  scale_[scale_ == 0 | !is.finite(scale_)] <- 1
  ref <- scale(ref, centre, scale_); cur <- scale(cur, centre, scale_)

  fit <- energy::eqdist.etest(rbind(ref, cur), sizes = c(nrow(ref), nrow(cur)), R = R)
  list(statistic = unname(fit$statistic), p_value = unname(fit$p.value),
       n_ref = nrow(ref), n_cur = nrow(cur))
}

#' Marginal two-sample tests, for the columns PSI flags.
#'
#' Anderson--Darling rather than Kolmogorov--Smirnov as the headline: KS is most
#' sensitive near the median and least sensitive in the tails, and in fraud the
#' tails are the subject. Both are reported, with permutation p-values.
marginal_tests <- function(win, columns, n_sub = 5000L, nboot = 500L, seed = 0L) {
  set.seed(seed)
  rows <- lapply(columns, function(f) {
    a <- stats::na.omit(as.numeric(win$train[[f]]))
    b <- stats::na.omit(as.numeric(win$holdout[[f]]))
    if (length(a) < 50L || length(b) < 50L || length(unique(c(a, b))) < 3L) {
      return(data.table::data.table(column = f, ad_p = NA_real_, cvm_p = NA_real_,
                                    cliffs_delta = NA_real_))
    }
    a <- a[sample.int(length(a), min(n_sub, length(a)))]
    b <- b[sample.int(length(b), min(n_sub, length(b)))]
    ad  <- twosamples::ad_test(a, b, nboot = nboot)
    cvm <- twosamples::cvm_test(a, b, nboot = nboot)
    # Cliff's delta is the effect size that belongs next to those p-values: a
    # rank statistic on the same scale as Somers' D, so "how much did it move"
    # is answered in the same units the association audit uses.
    delta <- 2 * auc_ci_delong(c(a, b), c(rep(0L, length(a)), rep(1L, length(b))))$auc - 1
    data.table::data.table(column = f, ad_p = unname(ad[[2]]), cvm_p = unname(cvm[[2]]),
                           cliffs_delta = delta)
  })
  out <- data.table::rbindlist(rows)
  out[, `:=`(ad_p_bh = adjust_bh(ad_p), cvm_p_bh = adjust_bh(cvm_p))]
  out[]
}

#' Build the distribution-shift fragment.
#'
#' Rejection needs both keys. `psi_threshold` is materiality -- the movement has
#' to be big enough to act on -- and the permutation p-value is significance.
#' At 100,000 rows per window the second is nearly free, which is exactly why it
#' cannot be the only one: a test that rejects everything has decided nothing.
distribution_shift_fragment <- function(report, psi_threshold = 0.25, alpha = 0.05,
                                        params = list()) {
  rejected <- report[!is.na(psi) & psi > psi_threshold & !is.na(p_perm_bh) & p_perm_bh < alpha]
  list(
    check = "distribution_shift",
    tool = "fraudaudit::distribution_shift_scan",
    params = utils::modifyList(as.list(params),
                               list(psi_threshold = psi_threshold, alpha = alpha)),
    qualification = list(
      columns_scanned = nrow(report),
      degenerate = sum(is.na(report$psi)),
      rejected = nrow(rejected),
      significant_only = sum(!is.na(report$p_perm_bh) & report$p_perm_bh < alpha),
      material_only = sum(!is.na(report$psi) & report$psi > psi_threshold),
      # What the index looks like when nothing has moved, at this n. The gap
      # between this and the conventional 0.25 is the reason the threshold
      # cannot be read as a significance level.
      median_null_q99 = stats::median(report$psi_null_q99, na.rm = TRUE),
      max_psi = max(report$psi, na.rm = TRUE)
    ),
    rejections = lapply(seq_len(nrow(rejected)), function(i) list(
      column = rejected$column[[i]], check = "distribution_shift",
      value = rejected$psi[[i]], unit = "column"
    ))
  )
}
