test_that("Cramer's V matches the reference implementation", {
  set.seed(41)
  tab <- table(sample(letters[1:4], 2000, TRUE), sample(c("x", "y"), 2000, TRUE))
  expect_equal(cramer_v_from_table(tab, bias_correct = TRUE),
               unname(DescTools::CramerV(tab, correct = TRUE)), tolerance = 1e-8)
  expect_equal(cramer_v_from_table(tab, bias_correct = FALSE),
               unname(DescTools::CramerV(tab, correct = FALSE)), tolerance = 1e-8)
})

test_that("V is 0 under independence and 1 under a perfect relationship", {
  independent <- matrix(c(250, 250, 250, 250), nrow = 2)
  expect_lt(cramer_v_from_table(independent, bias_correct = FALSE), 1e-9)
  perfect <- matrix(c(500, 0, 0, 500), nrow = 2)
  expect_equal(cramer_v_from_table(perfect, bias_correct = FALSE), 1, tolerance = 1e-9)
})

test_that("the bias correction pulls a sparse table down", {
  sparse <- matrix(c(40, 2, 1, 3, 38, 2, 1, 2, 39), nrow = 3)
  expect_lt(cramer_v_from_table(sparse, TRUE), cramer_v_from_table(sparse, FALSE))
})

test_that("the multinomial bootstrap brackets the estimate and narrows with n", {
  set.seed(42)
  small <- cramers_v(sample(letters[1:3], 300, TRUE), sample(c("a", "b"), 300, TRUE), R = 400L)
  big <- cramers_v(sample(letters[1:3], 30000, TRUE), sample(c("a", "b"), 30000, TRUE), R = 400L)
  expect_true(small$lower <= small$v && small$v <= small$upper)
  expect_gt(small$upper - small$lower, big$upper - big$lower)
})

test_that("a column that marks the period is separated from one that marks fraud", {
  set.seed(43); n <- 4000
  # `marker` is present only late; `signal` relates to the label in both windows.
  win <- list(
    train = data.table::data.table(marker = rep("old", n), signal = sample(c("hi","lo"), n, TRUE),
                                   isFraud = 0L),
    holdout = data.table::data.table(marker = rep("new", n), signal = sample(c("hi","lo"), n, TRUE),
                                     isFraud = 0L),
    features = c("marker", "signal"), label = "isFraud")
  win$train[, isFraud := rbinom(n, 1, ifelse(signal == "hi", 0.2, 0.05))]
  win$holdout[, isFraud := rbinom(n, 1, ifelse(signal == "hi", 0.2, 0.05))]

  out <- categorical_scan(win, R = 200L, verbose = FALSE)
  expect_true(out[column == "marker"]$knows_period_better)
  expect_false(out[column == "signal"]$knows_period_better)
})
