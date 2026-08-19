# A binning, pinned to a reference window.
#
# The same object serves two audits. Weight of evidence asks whether the *odds*
# inside a bin still hold later; the population stability index asks whether the
# *mass* inside that bin moved. Both questions are meaningless unless the bin
# edges are the same on both sides of the comparison, so the edges are fitted
# once, on the reference window, and carried to every later window unchanged.
#
# Recomputing edges per call silently compares two different histograms and
# reports the difference as stability.

NULL_BUCKET  <- "__null__"
OTHER_BUCKET <- "__other__"

#' Fit bin edges on a reference window.
#'
#' Missingness is a bucket, not a row to drop. On columns that are 86% null a
#' change in how often the column is populated *is* the signal, and dropping
#' those rows deletes the finding.
#'
#' Categorical levels are capped at `max_levels` by frequency in the reference;
#' anything else, including levels that appear only later, collapses into
#' `__other__` rather than being ignored.
#'
#' @param x A vector.
#' @param bins Target bin count for numerics. Quantile-based, so bins carry
#'   roughly equal mass and a heavy tail does not produce nine empty buckets.
#' @param max_levels Level cap for categoricals.
#' @return A `binning` object, or one marked `degenerate` with the reason.
fit_binning <- function(x, bins = 10L, max_levels = 20L) {
  null_share <- mean(is.na(x))
  degenerate <- function(kind, why) {
    structure(list(kind = kind, degenerate = why, null_share = null_share),
              class = "binning")
  }

  if (is.logical(x)) x <- as.integer(x)

  if (is.numeric(x)) {
    v <- x[!is.na(x)]
    if (!length(v)) return(degenerate("numeric", "no non-null values"))
    distinct <- unique(v)
    if (length(distinct) < 2L) return(degenerate("numeric", "constant"))

    # A quantile grid is the wrong instrument for a column whose mass sits on
    # one value. Most of the V block is like that -- V297 takes 13 distinct
    # values with 96% of rows on one of them -- and every quantile from 0 to 0.9
    # then lands on the same number, collapsing the grid to a single interval.
    # Such a column is not degenerate, it is discrete, and the honest binning is
    # one bin per value: exact, lossless, and readable as a table of counts.
    if (length(distinct) <= bins) {
      return(structure(list(kind = "discrete", values = sort(distinct),
                            degenerate = NA_character_, null_share = null_share),
                       class = "binning"))
    }

    probs <- seq(0, 1, length.out = bins + 1L)
    edges <- unique(stats::quantile(v, probs, type = 7L, names = FALSE))
    if (length(edges) >= 3L) {
      edges[[1]] <- -Inf
      edges[[length(edges)]] <- Inf
      return(structure(list(kind = "numeric", edges = edges, degenerate = NA_character_,
                            null_share = null_share), class = "binning"))
    }

    # The grid collapsed but the column has many values: keep the frequent ones
    # as their own bins and collapse the sparse tail, which is where the
    # quantile grid was putting everything anyway.
    tab <- sort(table(v), decreasing = TRUE)
    keep <- as.numeric(names(tab)[seq_len(min(max_levels, length(tab)))])
    structure(list(kind = "discrete", values = sort(keep), degenerate = NA_character_,
                   null_share = null_share), class = "binning")
  } else {
    v <- as.character(x)[!is.na(x)]
    if (!length(v)) return(degenerate("categorical", "no non-null values"))
    tab <- sort(table(v), decreasing = TRUE)
    levels <- names(tab)[seq_len(min(max_levels, length(tab)))]
    structure(list(kind = "categorical", levels = levels, degenerate = NA_character_,
                   null_share = null_share), class = "binning")
  }
}

#' Assign a vector to the bins of a fitted `binning`.
#'
#' @return A factor whose levels are fixed by the binning, so two windows always
#'   produce the same level set even when one of them never fills a bucket.
assign_bins <- function(binning, x) {
  stopifnot(inherits(binning, "binning"))
  if (!is.na(binning$degenerate)) stop("degenerate binning: ", binning$degenerate, call. = FALSE)
  if (is.logical(x)) x <- as.integer(x)

  if (binning$kind == "numeric") {
    out <- as.character(cut(as.numeric(x), breaks = binning$edges,
                            labels = bin_labels_core(binning),
                            include.lowest = TRUE, right = FALSE))
  } else if (binning$kind == "discrete") {
    # Matched on the numbers, then indexed into the label vector. Formatting the
    # data and the bin values separately looks equivalent and is not: `format()`
    # pads to a common width across whatever vector it is handed, so the same
    # value renders differently in the two calls, nothing matches, and every row
    # silently becomes `__null__`.
    xs <- as.numeric(x)
    labels <- bin_labels_core(binning)
    idx <- match(xs, binning$values)
    out <- ifelse(is.na(xs), NA_character_,
                  ifelse(is.na(idx), OTHER_BUCKET, labels[idx]))
  } else {
    xs <- as.character(x)
    out <- ifelse(is.na(xs), NA_character_,
                  ifelse(xs %in% binning$levels, xs, OTHER_BUCKET))
  }
  out[is.na(out)] <- NULL_BUCKET
  factor(out, levels = bin_labels(binning))
}

#' The value-carrying levels of a binning, in order.
bin_labels_core <- function(binning) {
  switch(binning$kind,
    numeric = {
      e <- binning$edges
      sprintf("[%s, %s)", format(e[-length(e)], trim = TRUE), format(e[-1], trim = TRUE))
    },
    discrete = c(formatC(binning$values, format = "g", digits = 15), OTHER_BUCKET),
    categorical = c(binning$levels, OTHER_BUCKET)
  )
}

#' The full, fixed level set of a binning, `__null__` last.
#'
#' Null is a level rather than a dropped row, and it is last so that a bin
#' ordering read off this vector is an ordering of values, not of absence.
bin_labels <- function(binning) c(bin_labels_core(binning), NULL_BUCKET)
