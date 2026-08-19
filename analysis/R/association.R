# Rank statistics, with the uncertainty attached.
#
# The area under the ROC curve of a single column is the Mann--Whitney U
# statistic rescaled: the probability that a randomly drawn fraud outranks a
# randomly drawn non-fraud. Nothing has to be fitted to obtain it, which is why
# the audits in this package need no model. Somers' D is the same statistic
# centred on zero, D = 2C - 1, and is the form the credit literature reports.
#
# Every estimate here carries a confidence interval, because the verdict this
# feeds is a comparison of two AUCs and the interesting cases sit within a few
# hundredths of 0.5. DeLong's variance estimator is used throughout: it is
# non-parametric, exact for the rank statistic, and does not resample, which
# matters at 100k rows per window and 377 columns.
#
# References: DeLong, DeLong & Clarke-Pearson (1988), *Biometrics* 44(3);
# Somers (1962), *American Sociological Review* 27(6).

#' Somers' D of a raw column against the label, on complete cases.
#'
#' Direction-carrying: a negative D means high values of `x` go with the
#' negative class. This is the monotone reading of the association and is
#' reported alongside, never used for the verdict on its own -- a column whose
#' fraud rate is high at both ends scores near zero here while carrying real
#' signal, which is exactly the case the weight-of-evidence path exists for.
#'
#' Returns `NA` for unordered categoricals rather than coercing them. Somers' D
#' is a measure of *ordinal* association; the levels of `ProductCD` have no
#' order, so every value it could return would be an artefact of how the levels
#' happened to be encoded. Those columns are judged on weight of evidence,
#' which needs no ordering of its own.
somers_dxy <- function(x, y) {
  if (is.logical(x)) x <- as.integer(x)
  if (!is.numeric(x)) return(list(dxy = NA_real_, c_index = NA_real_, n = sum(!is.na(x))))
  ok <- !is.na(x) & !is.na(y)
  if (sum(ok) < 2L || length(unique(x[ok])) < 2L || length(unique(y[ok])) < 2L) {
    return(list(dxy = NA_real_, c_index = NA_real_, n = sum(ok)))
  }
  s <- Hmisc::somers2(as.numeric(x[ok]), as.integer(y[ok]))
  list(dxy = unname(s[["Dxy"]]), c_index = unname(s[["C"]]), n = sum(ok))
}

#' AUC with a DeLong confidence interval, at a *fixed* direction.
#'
#' `direction = "<"` is pinned rather than inferred. pROC's default flips the
#' comparison whenever that would push the AUC above 0.5, which is convenient
#' for reporting a single curve and fatal here: an AUC below 0.5 in the later
#' window is the finding, and an implementation that quietly turns it into its
#' complement reports every inverted feature as healthy.
auc_ci_delong <- function(score, y, conf_level = 0.95) {
  ok <- !is.na(score) & !is.na(y)
  if (sum(ok) < 2L || length(unique(y[ok])) < 2L || length(unique(score[ok])) < 2L) {
    return(list(auc = NA_real_, lower = NA_real_, upper = NA_real_, roc = NULL, n = sum(ok)))
  }
  r <- pROC::roc(response = as.integer(y[ok]), predictor = as.numeric(score[ok]),
                 levels = c(0L, 1L), direction = "<", quiet = TRUE)
  ci <- as.numeric(pROC::ci.auc(r, method = "delong", conf.level = conf_level))
  list(auc = unname(as.numeric(pROC::auc(r))), lower = ci[[1]], upper = ci[[3]],
       roc = r, n = sum(ok))
}

#' DeLong's test that two AUCs are equal, for independent samples.
#'
#' `paired = FALSE`: the two windows are different rows, not two scores on the
#' same rows, so the covariance term is zero and the test compares the estimates
#' against the sum of their variances. Passing the paired form here would
#' understate the standard error and turn ordinary sampling noise into a
#' significant change on almost every column.
delong_two_sample <- function(roc_ref, roc_cur) {
  if (is.null(roc_ref) || is.null(roc_cur)) return(list(p_value = NA_real_, statistic = NA_real_))
  out <- try(pROC::roc.test(roc_ref, roc_cur, method = "delong", paired = FALSE),
             silent = TRUE)
  if (inherits(out, "try-error")) return(list(p_value = NA_real_, statistic = NA_real_))
  list(p_value = unname(out$p.value), statistic = unname(out$statistic))
}

#' Benjamini--Hochberg adjustment.
#'
#' Not optional at this width. The scan tests 377 columns; at a nominal 5% that
#' is roughly nineteen columns rejected by chance alone, and a contract that
#' cannot tell those nineteen from real findings is not evidence of anything.
#' BH rather than Bonferroni because the cost here is a dropped feature, not a
#' false safety claim, and the tests are positively dependent -- neighbouring V
#' columns move together -- which is the regime BH is valid under.
adjust_bh <- function(p) stats::p.adjust(p, method = "BH")
