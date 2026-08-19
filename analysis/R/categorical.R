# Association between two categorical things, with an interval on it.
#
# PSI over binned levels answers "did the mix move" and says nothing about how
# strong the dependence is or how precisely it was measured. Cramer's V does:
# it is chi-squared rescaled onto [0, 1] by the table's dimensions, so a value
# from a 5x2 table is comparable with one from a 20x2 table, which the raw
# statistic is not.
#
# The interval is bootstrapped rather than taken from an asymptotic formula. V
# is a bounded, skewed statistic and its sampling distribution is not symmetric
# near zero, which is exactly where most of these columns sit -- a normal
# interval there routinely covers negative values that the statistic cannot
# take.
#
# References: Cramer (1946), *Mathematical Methods of Statistics*, section 21.9;
# Bergsma (2013) on the bias correction.

#' Cramer's V from a contingency table, with Bergsma's bias correction.
#'
#' Written out rather than delegated so that the bootstrap below can call it
#' thousands of times on count vectors without rebuilding a table each time.
#' `DescTools::CramerV` is used in the tests as the reference implementation.
#'
#' The correction subtracts the expected value of phi-squared under
#' independence. Uncorrected V is biased upward whenever the table is sparse,
#' and a twenty-level column against a binary outcome is sparse in its tail
#' levels no matter how many rows there are.
cramer_v_from_table <- function(tab, bias_correct = TRUE) {
  n <- sum(tab)
  if (n == 0 || any(dim(tab) < 2L)) return(NA_real_)
  expected <- outer(rowSums(tab), colSums(tab)) / n
  keep <- expected > 0
  chi2 <- sum((tab[keep] - expected[keep])^2 / expected[keep])
  r <- nrow(tab); c_ <- ncol(tab)

  if (!bias_correct) return(sqrt((chi2 / n) / min(r - 1L, c_ - 1L)))

  phi2 <- max(0, chi2 / n - (r - 1L) * (c_ - 1L) / (n - 1L))
  r_t <- r - (r - 1L)^2 / (n - 1L)
  c_t <- c_ - (c_ - 1L)^2 / (n - 1L)
  denom <- min(r_t - 1, c_t - 1)
  if (denom <= 0) return(NA_real_)
  sqrt(phi2 / denom)
}

#' Cramer's V with a bootstrap percentile interval.
#'
#' The resampling is done on the *cell counts*, not on the rows. V depends on
#' the two vectors only through their contingency table, and the table of a
#' bootstrap resample is a multinomial draw on the observed cell proportions --
#' so one `rmultinom` call replaces R passes over 200,000 rows. Same
#' distribution, three orders of magnitude less work.
#'
#' @param x,y Two categorical vectors of the same length.
#' @param bias_correct Apply Bergsma's correction.
#' @param R Bootstrap resamples.
cramers_v <- function(x, y, bias_correct = TRUE, R = 500L, conf_level = 0.95, seed = 0L) {
  ok <- !is.na(x) & !is.na(y)
  x <- as.character(x)[ok]; y <- as.character(y)[ok]
  if (length(x) < 50L || length(unique(x)) < 2L || length(unique(y)) < 2L) {
    return(list(v = NA_real_, lower = NA_real_, upper = NA_real_, n = length(x),
                p_value = NA_real_))
  }

  tab <- table(x, y)
  observed <- cramer_v_from_table(tab, bias_correct)
  chi <- suppressWarnings(stats::chisq.test(tab))

  set.seed(seed)
  n <- sum(tab)
  draws_flat <- stats::rmultinom(R, size = n, prob = as.numeric(tab) / n)
  draws <- vapply(seq_len(R), function(i) {
    cramer_v_from_table(matrix(draws_flat[, i], nrow = nrow(tab)), bias_correct)
  }, numeric(1))

  probs <- c((1 - conf_level) / 2, 1 - (1 - conf_level) / 2)
  ci <- stats::quantile(draws, probs, na.rm = TRUE, names = FALSE)

  list(v = observed, lower = ci[[1]], upper = ci[[2]], n = n,
       p_value = unname(chi$p.value))
}

#' Every categorical column, against the label and against the window.
#'
#' Two associations, and the pair is the point. Against the *label* is how much
#' the column knows about fraud. Against the *window* is how much it knows about
#' which period a row came from -- and a column that knows more about the period
#' than about the outcome is a drift detector wearing a feature's clothes.
categorical_scan <- function(win, features = NULL, R = 300L, max_levels = 50L,
                             verbose = TRUE) {
  features <- features %||% win$features
  is_categorical <- vapply(features, function(f) {
    x <- win$train[[f]]
    is.character(x) || is.factor(x) || is.logical(x)
  }, logical(1))
  features <- features[is_categorical]

  y_tr <- as.integer(win$train[[win$label]])
  rows <- vector("list", length(features))

  for (i in seq_along(features)) {
    f <- features[[i]]
    x_tr <- win$train[[f]]; x_ho <- win$holdout[[f]]

    # Rare levels are pooled before the table is built. A level seen twice
    # contributes a cell that chi-squared's asymptotics do not cover, and
    # keeping hundreds of them inflates V through the dimension term alone.
    keep <- names(sort(table(as.character(x_tr)), decreasing = TRUE))
    keep <- keep[seq_len(min(max_levels, length(keep)))]
    lump <- function(v) {
      v <- as.character(v)
      ifelse(is.na(v), NULL_BUCKET, ifelse(v %in% keep, v, OTHER_BUCKET))
    }

    to_label <- cramers_v(lump(x_tr), y_tr, R = R, seed = i)
    window_marker <- c(rep("early", length(x_tr)), rep("late", length(x_ho)))
    to_window <- cramers_v(c(lump(x_tr), lump(x_ho)), window_marker, R = R, seed = i)

    # A column constant inside a window has no measurable association with the
    # label there -- and if it still separates the two windows, that is the
    # purest form of the failure this comparison looks for, not a case to skip.
    # An unmeasurable label association is read as "tells us nothing about
    # fraud", which is what it is.
    label_measurable <- !is.na(to_label$v)
    period_better <- !is.na(to_window$v) && (
      if (label_measurable) to_window$lower > to_label$upper else to_window$lower > 0.1
    )

    rows[[i]] <- data.table::data.table(
      column = f, levels = length(keep), label_measurable = label_measurable,
      v_label = to_label$v, v_label_lo = to_label$lower, v_label_hi = to_label$upper,
      v_window = to_window$v, v_window_lo = to_window$lower, v_window_hi = to_window$upper,
      p_label = to_label$p_value, p_window = to_window$p_value,
      knows_period_better = period_better
    )
    if (verbose && i %% 5L == 0L) message(sprintf("  %d/%d", i, length(features)))
  }

  out <- data.table::rbindlist(rows)
  out[, `:=`(p_label_bh = adjust_bh(p_label), p_window_bh = adjust_bh(p_window))]
  data.table::setorder(out, -v_window)
  out[]
}
