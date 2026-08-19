test_that("PSI is zero for identical distributions and grows with separation", {
  counts <- c(100L, 200L, 300L, 400L)
  expect_equal(psi_value(counts, counts), 0)
  expect_lt(psi_value(counts, c(110L, 200L, 300L, 390L)),
            psi_value(counts, c(400L, 300L, 200L, 100L)))
})

test_that("PSI is symmetric in its two arguments", {
  a <- c(50L, 150L, 300L); b <- c(200L, 100L, 200L)
  expect_equal(psi_value(a, b), psi_value(b, a), tolerance = 1e-12)
})

test_that("an empty reference bucket does not return infinity", {
  expect_true(is.finite(psi_value(c(0L, 100L), c(5L, 95L))))
})

test_that("the hypergeometric null matches an actual permutation", {
  # The whole speed argument rests on this: drawing the split counts from a
  # multivariate hypergeometric is the permutation distribution, not an
  # approximation of it. Checked against the brute-force version.
  set.seed(11)
  lv <- c("a", "b", "c", "d")
  ref <- factor(sample(lv, 3000, replace = TRUE, prob = c(.4, .3, .2, .1)), levels = lv)
  cur <- factor(sample(lv, 2000, replace = TRUE, prob = c(.4, .3, .2, .1)), levels = lv)

  fast <- psi_null(ref, cur, B = 4000L, seed = 1L)

  pooled <- c(as.integer(ref), as.integer(cur)); n_ref <- length(ref)
  set.seed(2)
  brute <- vapply(seq_len(4000), function(i) {
    idx <- sample.int(length(pooled))
    psi_value(tabulate(pooled[idx[seq_len(n_ref)]], nbins = 4L),
              tabulate(pooled[idx[-seq_len(n_ref)]], nbins = 4L))
  }, numeric(1))

  # Same null: the quantiles agree to within Monte Carlo error at B = 4000.
  expect_equal(fast$null_q99, unname(quantile(brute, 0.99)), tolerance = 0.25)
  expect_equal(fast$null_median, median(brute), tolerance = 0.25)
})

test_that("a real shift is separated from sampling noise", {
  set.seed(12)
  b <- fit_binning(rnorm(10000))
  quiet <- psi_null(assign_bins(b, rnorm(20000)), assign_bins(b, rnorm(20000)), B = 200L)
  moved <- psi_null(assign_bins(b, rnorm(20000)), assign_bins(b, rnorm(20000, mean = 0.3)), B = 200L)

  expect_gt(quiet$p_value, 0.05)
  expect_lt(moved$p_value, 0.05)
  # And the point the notebook makes: the noise floor at this n is two orders of
  # magnitude below the conventional 0.25 cut.
  expect_lt(quiet$null_q99, 0.01)
  expect_gt(moved$psi, 20 * moved$null_q99)
})

test_that("the null-excluded index separates missingness from value drift", {
  # A column that became far more populated while its values stayed put: the
  # pooled index is large, the value-only index is not. These are different
  # findings and the report has to keep them apart.
  set.seed(13)
  v <- rnorm(20000)
  ref <- c(v[1:5000], rep(NA_real_, 15000))
  cur <- c(v[5001:20000], rep(NA_real_, 5000))
  b <- fit_binning(ref)
  br <- assign_bins(b, ref); bc <- assign_bins(b, cur)

  pooled_psi <- psi_null(br, bc, B = 50L)$psi
  keep <- levels(br) != NULL_BUCKET
  value_psi <- psi_value(tabulate(br, nlevels(br))[keep], tabulate(bc, nlevels(bc))[keep])

  expect_gt(pooled_psi, 0.5)
  expect_lt(value_psi, 0.05)
})

test_that("the energy test tells a moved joint distribution from a still one", {
  skip_if_not_installed("energy")
  set.seed(14)
  mk <- function(n, shift = 0) {
    data.table::data.table(a = rnorm(n, shift), b = rnorm(n), isFraud = 0L)
  }
  still <- list(train = mk(3000), holdout = mk(3000), features = c("a", "b"), label = "isFraud")
  moved <- list(train = mk(3000), holdout = mk(3000, shift = 0.3), features = c("a", "b"), label = "isFraud")

  expect_gt(multivariate_shift(still, c("a", "b"), n_sub = 600L, R = 99L)$p_value, 0.05)
  expect_lt(multivariate_shift(moved, c("a", "b"), n_sub = 600L, R = 99L)$p_value, 0.05)
})

test_that("a fragment records each policy key once", {
  # `c(list(alpha = ...), params)` keeps both copies when params already has the
  # key, and the duplicate reappears on the next read as `alpha.1`.
  report <- data.table::data.table(
    column = c("a", "b"), psi = c(0.9, 0.01), psi_null_q99 = c(0.001, 0.001),
    p_perm = c(0.005, 0.5), p_perm_bh = c(0.005, 0.5))
  frag <- distribution_shift_fragment(report, psi_threshold = 0.25, alpha = 0.05,
                                      params = list(psi_threshold = 0.25, alpha = 0.05,
                                                    bins = 10L))
  expect_equal(anyDuplicated(names(frag$params)), 0L)
  expect_equal(frag$params$psi_threshold, 0.25)
})
