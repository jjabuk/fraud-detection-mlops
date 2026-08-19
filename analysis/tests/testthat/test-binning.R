test_that("quantile_nearest reproduces the polars order statistic", {
  # The edge case the whole migration turned on: with n = 590540 and p = 0.83,
  # linear interpolation and the nearest order statistic disagree by one row.
  x <- seq_len(1000)
  expect_equal(quantile_nearest(x, 0), 1)
  expect_equal(quantile_nearest(x, 1), 1000)
  expect_equal(quantile_nearest(x, 0.5), x[[round(0.5 * 999) + 1L]])
})

test_that("windows are half-open except at the top", {
  w <- time_windows(1:100, train = c(0, 0.2), holdout = c(0.8, 1.0))
  expect_false(any(w$train %in% w$holdout))
  expect_true(100 %in% w$holdout)  # the last row is not silently dropped
})

test_that("overlapping windows are refused", {
  expect_error(time_windows(1:100, train = c(0, 0.6), holdout = c(0.5, 1)), "overlaps")
})

test_that("a mass-concentrated column bins discretely rather than degenerating", {
  # 96% of rows on one value: the quantile grid collapses, but the column is
  # perfectly binnable one value at a time. This is most of the V block.
  x <- c(rep(0, 9600), sample(1:12, 400, replace = TRUE))
  b <- fit_binning(x)
  expect_equal(b$kind, "discrete")
  expect_true(is.na(b$degenerate))
})

test_that("discrete bins survive the round trip through their labels", {
  # Formatting the data and the bin values in separate calls to format() pads
  # them to different widths and silently sends every row to __null__.
  x <- c(rep(0, 100), rep(100000, 50), rep(1e-5, 50))
  b <- fit_binning(x)
  assigned <- assign_bins(b, x)
  expect_equal(sum(assigned == NULL_BUCKET), 0L)
  expect_equal(length(unique(as.character(assigned))), 3L)
})

test_that("levels absent from the reference collapse into __other__", {
  b <- fit_binning(c(rep("a", 10), rep("b", 5)), max_levels = 2)
  expect_equal(as.character(assign_bins(b, c("a", "zzz", NA))),
               c("a", OTHER_BUCKET, NULL_BUCKET))
})

test_that("nulls are a bucket, never a dropped row", {
  b <- fit_binning(c(1:100, rep(NA, 50)))
  expect_true(NULL_BUCKET %in% bin_labels(b))
  expect_equal(sum(assign_bins(b, c(NA, NA, 5)) == NULL_BUCKET), 2L)
})

test_that("a loaded window comes back in a deterministic row order", {
  # A filtered Arrow scan is multi-threaded and does not promise file order:
  # two runs return the same rows in a different sequence. Nothing that reads
  # every row notices, and every audit that subsamples does -- `set.seed(0);
  # sample.int(n, k)` then picks the same positions out of a different ordering,
  # so the *sample* differs while the seed and the data do not. That made the
  # redundancy audit, and with it the contract, differ between identical runs.
  #
  # The property that fixes it is testable without reproducing the race: the
  # frame must come back sorted on the time axis.
  skip_if_not_installed("arrow")
  path <- file.path(tempdir(), "order-check.parquet")
  set.seed(9)
  n <- 2000
  arrow::write_parquet(
    data.frame(
      TransactionID = seq_len(n),
      TransactionDT = sample(seq_len(n) * 100),   # deliberately shuffled on disk
      isFraud = rbinom(n, 1, 0.2),
      a = rnorm(n)
    ), path)

  win <- load_windows(path, features = "a", train = c(0, 0.5), holdout = c(0.5, 1))
  expect_false(is.unsorted(win$train$TransactionDT %||% seq_len(nrow(win$train))))
  # The sort keys are dropped again unless they were asked for as features.
  expect_setequal(names(win$train), c("a", "isFraud"))

  again <- load_windows(path, features = "a", train = c(0, 0.5), holdout = c(0.5, 1))
  expect_identical(win$train$a, again$train$a)
})
