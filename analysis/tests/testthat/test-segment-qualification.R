test_that("the dichotomy splits on risk, not on population", {
  set.seed(31); n <- 4000
  x <- runif(n)
  y <- rbinom(n, 1, ifelse(x > 0.8, 0.4, 0.02))
  high <- risk_dichotomy(x, y)
  # The risky group is the top fifth, not the top half a median split would give.
  expect_lt(mean(high, na.rm = TRUE), 0.45)
  expect_gt(mean(y[high], na.rm = TRUE), mean(y[!high], na.rm = TRUE))
})

test_that("Simpson's paradox is caught: pooled association, none within strata", {
  # x is pure segment membership. Pooled it looks predictive, because the
  # segments have different base rates; within either segment it says nothing.
  set.seed(32); n <- 6000
  segment <- rep(c("A", "B"), each = n / 2)
  x <- ifelse(segment == "A", rnorm(n, 0), rnorm(n, 5))
  y <- rbinom(n, 1, ifelse(segment == "A", 0.02, 0.30))

  pooled <- column_auc_directionless(x, y, rep(TRUE, n))
  expect_gt(pooled, 0.7)

  tests <- segment_tests(x, y, segment)
  # Conditioned on segment the association is gone: CMH cannot reject.
  expect_gt(tests$cmh_p, 0.05)
})

test_that("a genuine association survives conditioning on the segment", {
  set.seed(33); n <- 6000
  segment <- rep(c("A", "B"), each = n / 2)
  x <- runif(n)
  y <- rbinom(n, 1, plogis(-3 + 3 * x))  # same relationship in both strata
  tests <- segment_tests(x, y, segment)
  expect_lt(tests$cmh_p, 0.05)
  expect_gt(tests$cmh_or, 1)
  # And Breslow-Day should not find the strata different, because they are not.
  expect_gt(tests$bd_p, 0.01)
})

test_that("Breslow-Day flags an association that reverses between segments", {
  set.seed(34); n <- 8000
  segment <- rep(c("A", "B"), each = n / 2)
  x <- runif(n)
  # Opposite directions, netting out to something unremarkable when pooled.
  p <- ifelse(segment == "A", plogis(-3 + 3 * x), plogis(-3 + 3 * (1 - x)))
  y <- rbinom(n, 1, p)
  tests <- segment_tests(x, y, segment)
  expect_lt(tests$bd_p, 0.05)
})

test_that("strata without both outcomes are dropped and counted", {
  set.seed(35); n <- 3000
  segment <- rep(c("A", "B", "C"), each = n / 3)
  x <- runif(n)
  y <- rbinom(n, 1, plogis(-2 + 2 * x))
  y[segment == "C"] <- 0L          # no fraud at all in C
  tests <- segment_tests(x, y, segment)
  expect_equal(tests$strata_used, 2L)
  expect_equal(tests$strata_dropped, 1L)
})

test_that("directionless AUC ignores which way the column points", {
  set.seed(36); n <- 3000
  x <- runif(n)
  up   <- rbinom(n, 1, plogis(-2 + 3 * x))
  down <- rbinom(n, 1, plogis(-2 + 3 * (1 - x)))
  a <- column_auc_directionless(x, up, rep(TRUE, n))
  b <- column_auc_directionless(x, down, rep(TRUE, n))
  expect_gt(a, 0.5); expect_gt(b, 0.5)
  expect_equal(a, b, tolerance = 0.05)
})
