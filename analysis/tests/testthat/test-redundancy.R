test_that("the rank correlation matrix is symmetric with a unit diagonal", {
  set.seed(21); n <- 2000
  a <- rnorm(n); b <- a + rnorm(n, sd = 0.1); c_ <- rnorm(n)
  win <- list(train = data.table::data.table(a = a, b = b, c = c_, isFraud = 0L),
              features = c("a", "b", "c"), label = "isFraud")
  rho <- spearman_matrix(win, n_sub = n)
  expect_equal(diag(rho), setNames(rep(1, 3), c("a", "b", "c")))
  expect_equal(rho, t(rho))
  expect_gt(rho["a", "b"], 0.9)
  expect_lt(abs(rho["a", "c"]), 0.1)
})

test_that("clustering groups a restated column with its original", {
  set.seed(22); n <- 3000
  a <- rnorm(n)
  win <- list(
    train = data.table::data.table(a = a, a_copy = a * -2 + 1, far = rnorm(n), isFraud = 0L),
    features = c("a", "a_copy", "far"), label = "isFraud")
  cl <- variable_clusters(spearman_matrix(win, n_sub = n), min_rho2 = 0.8)
  # Sign is irrelevant: the distance is 1 - rho^2, so a perfectly inverted copy
  # is the same column.
  expect_equal(cl$membership[["a"]], cl$membership[["a_copy"]])
  expect_true(cl$membership[["far"]] != cl$membership[["a"]])
})

test_that("the representative is the least missing, then the most informative", {
  win <- list(train = data.table::data.table(
    thin = c(rep(NA_real_, 80), rnorm(20)),
    full = rnorm(100), isFraud = 0L))
  tc <- data.table::data.table(feature = c("thin", "full"), iv_train = c(0.9, 0.1))
  reps <- choose_representatives(list(g1 = c("thin", "full")), win, tc_report = tc)
  # `thin` has nine times the information value and represents nothing in the
  # 80% of rows where it is absent.
  expect_equal(unique(reps$representative), "full")
})

test_that("correlation and predictability are asked separately", {
  # Three columns pairwise correlated but none reconstructable from the others,
  # against one that is an exact sum. Only the second kind is a rejection.
  set.seed(23); n <- 4000
  x <- rnorm(n); y <- rnorm(n)
  win <- list(train = data.table::data.table(x = x, y = y, s = x + y, isFraud = 0L),
              features = c("x", "y", "s"), label = "isFraud")
  red <- redundancy_within_groups(list(g1 = c("x", "y", "s")), win, r2 = 0.9, n_sub = n)
  expect_true(any(red$predictable))
  expect_false(all(red$predictable))
})

test_that("a singleton group is never a rejection", {
  win <- list(train = data.table::data.table(only = rnorm(500), isFraud = 0L))
  red <- redundancy_within_groups(list(g1 = "only"), win, n_sub = 500L)
  expect_false(any(red$predictable))
  expect_equal(red$note, "singleton")
})

test_that("a pinned block that holds is tighter inside than out", {
  set.seed(24); n <- 3000
  core <- rnorm(n)
  m <- cbind(a = core + rnorm(n, sd = .2), b = core + rnorm(n, sd = .2),
             c = core + rnorm(n, sd = .2), z = rnorm(n), w = rnorm(n))
  rho <- cor(apply(m, 2, rank))
  audit <- audit_pinned_blocks(rho, list(real = c("a", "b", "c"), fake = c("a", "z")))
  expect_true(audit[block == "real"]$holds)
  expect_gt(audit[block == "real"]$ratio, audit[block == "fake"]$ratio)
})

test_that("the representative does not depend on the order members arrived in", {
  # Two members tied on null rate and information value: without an explicit
  # tiebreak the winner is whichever was listed first, and the same data then
  # produces a different contract depending on how it was read in.
  win <- list(train = data.table::data.table(
    b = c(rnorm(90), rep(NA_real_, 10)),
    a = c(rnorm(90), rep(NA_real_, 10))))
  tc <- data.table::data.table(feature = c("a", "b"), iv_train = c(0.5, 0.5))
  forward <- choose_representatives(list(g = c("a", "b")), win, tc_report = tc)
  backward <- choose_representatives(list(g = c("b", "a")), win, tc_report = tc)
  expect_equal(unique(forward$representative), unique(backward$representative))
})
