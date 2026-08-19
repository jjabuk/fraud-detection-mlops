test_that("families are formed by exact null count", {
  win <- list(train = data.table::data.table(
    a = c(1, NA, NA, 4), b = c(1, NA, NA, 4), c = c(NA, 2, 3, 4), d = 1:4),
    features = c("a", "b", "c", "d"))
  fams <- missingness_families(win)
  expect_equal(length(fams), 1L)          # only a and b share a count
  expect_setequal(fams[[1]], c("a", "b"))
})

test_that("reference columns are the always-observed ones", {
  win <- list(train = data.table::data.table(
    always = rnorm(500), sometimes = c(rnorm(400), rep(NA, 100)),
    constant = rep(1, 500)),
    features = c("always", "sometimes", "constant"))
  refs <- reference_columns(win)
  expect_true("always" %in% refs)
  expect_false("sometimes" %in% refs)
  expect_false("constant" %in% refs)      # no variation, nothing to compare
})

test_that("MCAR is not rejected when absence really is random", {
  set.seed(51); n <- 4000
  ref <- rnorm(n)
  target <- rnorm(n)
  target[sample.int(n, n / 2)] <- NA      # absence independent of everything
  win <- list(train = data.table::data.table(target = target, ref = ref, other = rnorm(n)),
              features = c("target", "ref", "other"))
  out <- mcar_test_family(win, "target", reference = c("ref", "other"), n_sub = n)
  expect_gt(out$p_value, 0.01)
})

test_that("MCAR is rejected when absence depends on an observed column", {
  set.seed(52); n <- 4000
  ref <- rnorm(n)
  target <- rnorm(n)
  target[ref > 0.5] <- NA                 # absence driven by `ref`
  win <- list(train = data.table::data.table(target = target, ref = ref, other = rnorm(n)),
              features = c("target", "ref", "other"))
  out <- mcar_test_family(win, "target", reference = c("ref", "other"), n_sub = n)
  expect_lt(out$p_value, 0.01)
})

test_that("a family with no reference columns is reported, not silently tested", {
  win <- list(train = data.table::data.table(a = c(1, NA, 3, NA)), features = "a")
  out <- mcar_test_family(win, "a", reference = character(0))
  expect_true(is.na(out$p_value))
  expect_match(out$note, "reference")
})
