# How many things is a block of columns actually measuring?
#
# The V columns arrive in blocks whose members correlate heavily, and the
# obvious response is to compress each block. The question that decides whether
# that is worth doing is how many dimensions a block really has -- and the usual
# answers to it are guesses. An eigenvalue-greater-than-one rule and a scree
# plot read by eye are conventions, not tests, and both systematically overstate
# dimensionality on correlated blocks.
#
# Horn's parallel analysis is the test: generate data with the same shape and
# no structure, and keep only the components whose eigenvalues exceed what pure
# noise produced at that rank. It has a null distribution, and it is the method
# the factor-analysis literature settled on decades ago.
#
# The binary M columns need their own correlation. Pearson between two
# dichotomies is bounded well below 1 by the marginal split alone, so a pair of
# binaries that are the same variable can look weakly related. The tetrachoric
# coefficient estimates the correlation of the latent continuous variables the
# dichotomies came from, which is the quantity that matters here.
#
# References: Horn (1965), *Psychometrika* 30(2); Olsson (1979) on polychoric
# estimation; Revelle's `psych` documentation on when tetrachoric is required.

#' Horn's parallel analysis for one block of columns.
#'
#' @param n_sub Row subsample. The eigenvalue comparison is stable long before
#'   100,000 rows and the simulation cost is linear in them.
#' @param n_iter Simulated datasets forming the null.
parallel_analysis <- function(win, columns, n_sub = 10000L, n_iter = 50L, seed = 0L) {
  columns <- intersect(columns, names(win$train))
  numeric_cols <- columns[vapply(columns, function(f) is.numeric(win$train[[f]]), logical(1))]
  if (length(numeric_cols) < 3L) {
    return(list(components = NA_integer_, columns = length(numeric_cols),
                variance_first = NA_real_, note = "fewer than three numeric columns"))
  }

  set.seed(seed)
  idx <- sample.int(nrow(win$train), min(n_sub, nrow(win$train)))
  m <- as.matrix(win$train[idx, ..numeric_cols])
  storage.mode(m) <- "double"

  # Median imputation, stated. The alternative is listwise deletion, which on a
  # block that is 68% null removes the block. Imputing at the median makes the
  # missing rows contribute nothing to the covariance rather than contributing
  # a pattern -- conservative for a question about dimensionality, and the
  # missingness itself is audited separately.
  for (j in seq_len(ncol(m))) {
    med <- stats::median(m[, j], na.rm = TRUE)
    m[is.na(m[, j]), j] <- if (is.finite(med)) med else 0
  }
  keep <- apply(m, 2, stats::var) > 0
  m <- m[, keep, drop = FALSE]
  if (ncol(m) < 3L) {
    return(list(components = NA_integer_, columns = ncol(m),
                variance_first = NA_real_, note = "fewer than three varying columns"))
  }

  fit <- try(suppressWarnings(utils::capture.output(
    pa <- psych::fa.parallel(m, fa = "pc", n.iter = n_iter, plot = FALSE, sim = FALSE)
  )), silent = TRUE)
  if (inherits(fit, "try-error")) {
    return(list(components = NA_integer_, columns = ncol(m),
                variance_first = NA_real_, note = "parallel analysis failed"))
  }

  eigen_values <- eigen(stats::cor(m), only.values = TRUE)$values
  list(components = pa$ncomp, columns = ncol(m),
       variance_first = eigen_values[[1]] / sum(eigen_values), note = "")
}

#' Tetrachoric correlation among the binary columns.
#'
#' Returns the matrix and the mean off-diagonal, which is the number worth
#' quoting: the M block is either one latent thing measured nine ways or nine
#' things, and the Pearson matrix cannot tell those apart.
tetrachoric_matrix <- function(win, columns, n_sub = 10000L, seed = 0L) {
  columns <- intersect(columns, names(win$train))
  set.seed(seed)
  idx <- sample.int(nrow(win$train), min(n_sub, nrow(win$train)))
  m <- as.data.frame(win$train[idx, ..columns])
  m <- as.data.frame(lapply(m, function(x) {
    if (is.logical(x)) return(as.integer(x))
    if (is.character(x) || is.factor(x)) return(as.integer(factor(x)) - 1L)
    as.integer(x)
  }))
  binary <- vapply(m, function(x) length(unique(stats::na.omit(x))) == 2L, logical(1))
  m <- m[, binary, drop = FALSE]
  if (ncol(m) < 2L) return(list(rho = NULL, mean_off_diagonal = NA_real_,
                                columns = ncol(m), note = "fewer than two binary columns"))

  out <- try(suppressWarnings(utils::capture.output(
    tc <- psych::tetrachoric(m)
  )), silent = TRUE)
  if (inherits(out, "try-error")) {
    return(list(rho = NULL, mean_off_diagonal = NA_real_, columns = ncol(m),
                note = "tetrachoric failed"))
  }
  rho <- tc$rho
  list(rho = rho, mean_off_diagonal = mean(abs(rho[upper.tri(rho)]), na.rm = TRUE),
       columns = ncol(m), note = "")
}

#' Parallel analysis for every pinned block.
#'
#' The compression ratio is the finding: a block of twenty columns that parallel
#' analysis says carries two components is a block where a representative per
#' correlation group is throwing away less than it appears to.
dimensionality_scan <- function(win, blocks, n_sub = 10000L, n_iter = 30L, verbose = TRUE) {
  rows <- lapply(seq_along(blocks), function(i) {
    name <- names(blocks)[[i]]
    pa <- parallel_analysis(win, blocks[[i]], n_sub = n_sub, n_iter = n_iter, seed = i)
    if (verbose) message(sprintf("  %s (%d/%d)", name, i, length(blocks)))
    data.table::data.table(
      block = name, columns = pa$columns, components = pa$components,
      compression = pa$columns / pmax(pa$components, 1L),
      variance_first_component = pa$variance_first, note = pa$note
    )
  })
  out <- data.table::rbindlist(rows)
  data.table::setorder(out, -compression)
  out[]
}
