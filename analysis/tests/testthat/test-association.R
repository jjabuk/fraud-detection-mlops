test_that("Somers' D is the AUC on the Mann-Whitney scale", {
  set.seed(3); n <- 5000; x <- runif(n); y <- rbinom(n, 1, plogis(-2 + 3 * x))
  s <- somers_dxy(x, y)
  expect_equal(s$dxy, 2 * s$c_index - 1, tolerance = 1e-9)
  expect_equal(s$c_index, auc_ci_delong(x, y)$auc, tolerance = 1e-6)
})

test_that("direction is pinned, so an inverted feature reports an AUC below 0.5", {
  # pROC's default would flip the comparison and report 0.65 for both, which is
  # precisely the finding this audit exists to detect.
  set.seed(4); n <- 5000; x <- runif(n)
  y_up   <- rbinom(n, 1, plogis(-2 + 3 * x))
  y_down <- rbinom(n, 1, plogis(-2 + 3 * (1 - x)))
  expect_gt(auc_ci_delong(x, y_up)$auc, 0.5)
  expect_lt(auc_ci_delong(x, y_down)$auc, 0.5)
})

test_that("Somers' D refuses unordered categoricals instead of coercing them", {
  s <- somers_dxy(c("a", "b", "c", "a"), c(0L, 1L, 1L, 0L))
  expect_true(is.na(s$dxy))
})

test_that("the DeLong interval covers the truth and narrows with n", {
  set.seed(5)
  wide   <- auc_ci_delong(runif(500),   rbinom(500, 1, 0.3))
  narrow <- auc_ci_delong(runif(50000), rbinom(50000, 1, 0.3))
  expect_gt(wide$upper - wide$lower, narrow$upper - narrow$lower)
  expect_true(narrow$lower <= 0.5 && narrow$upper >= 0.5)  # no signal, by construction
})

test_that("BH leaves a single test alone and shrinks a family", {
  expect_equal(adjust_bh(0.03), 0.03)
  # One borderline result among ninety-nine nulls is what multiplicity costs:
  # 0.03 alone survives, 0.03 found by looking a hundred times does not.
  expect_gt(adjust_bh(c(0.03, rep(0.5, 99)))[[1]], 0.05)
  # Identical p-values are the case BH does *not* shrink -- every rank cancels
  # the family size -- so the family has to actually vary for the test to bite.
  expect_equal(unname(adjust_bh(rep(0.03, 100))), rep(0.03, 100))
})
