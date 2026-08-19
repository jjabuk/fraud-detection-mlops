test_that("information value rises with separation", {
  set.seed(1); n <- 20000; x <- runif(n)
  weak   <- woe_table(assign_bins(fit_binning(x), x), rbinom(n, 1, plogis(-3 + 0.2 * x)))
  strong <- woe_table(assign_bins(fit_binning(x), x), rbinom(n, 1, plogis(-3 + 4.0 * x)))
  expect_lt(information_value(weak), information_value(strong))
})

test_that("an empty bin gets the Haldane correction, not an infinity", {
  bins <- factor(c(rep("a", 100), rep("b", 100)), levels = c("a", "b", "c"))
  y <- c(rep(0L, 100), rep(1L, 100))
  tab <- woe_table(bins, y)
  expect_true(all(is.finite(tab$woe)))
  expect_true(all(is.finite(tab$iv_part)))
})

test_that("agreement separates a stable feature from an inverted one", {
  set.seed(2); n <- 20000
  x <- runif(n); y <- rbinom(n, 1, plogis(-3 + 3 * x))
  b <- fit_binning(x); ref <- woe_table(assign_bins(b, x), y)

  x2 <- runif(n)
  stable   <- woe_table(assign_bins(b, x2), rbinom(n, 1, plogis(-3 + 3 * x2)))
  inverted <- woe_table(assign_bins(b, x2), rbinom(n, 1, plogis(-3 + 3 * (1 - x2))))

  expect_gt(woe_agreement(ref, stable)$rho, 0.8)
  expect_lt(woe_agreement(ref, inverted)$rho, -0.8)
  # The separation is an order of magnitude, so the threshold sits well clear of
  # both: a few percent of mass always lands in bins whose WoE wobbles across
  # the eps band by chance, and that is not what "the feature reversed" means.
  expect_lt(woe_agreement(ref, stable)$flip_mass, 0.15)
  expect_gt(woe_agreement(ref, inverted)$flip_mass, 0.5)
})

test_that("bins empty on one side do not pad the correlation with zeros", {
  # Their WoE is the correction alone, which is zero by construction; left in,
  # a vector of structural zeros correlates with itself whatever the data did.
  lv <- c("a", "b", "c", "d")
  f <- function(y) woe_table(factor(rep(c("a", "b", "c"), each = 60), levels = lv), y)
  ref <- f(c(rep(1L, 45), rep(0L, 15), rep(1L, 30), rep(0L, 30), rep(1L, 15), rep(0L, 45)))
  cur <- f(c(rep(0L, 45), rep(1L, 15), rep(1L, 30), rep(0L, 30), rep(0L, 15), rep(1L, 45)))
  # Level "d" is empty on both sides. With it counted the vectors are five
  # points long and the correlation is diluted; dropped, the three real bins
  # reverse exactly.
  expect_equal(woe_agreement(ref, cur)$rho, -1)
})

test_that("a two-bin column reports no correlation but still reports flipped mass", {
  # Spearman on two points is +/-1 whatever the numbers are, so it is withheld.
  # The mass statistic is what carries a binary column verdict.
  lv <- c("a", "b")
  ref <- woe_table(factor(rep(c("a", "b"), each = 50), levels = lv),
                   c(rep(1L, 40), rep(0L, 10), rep(0L, 40), rep(1L, 10)))
  cur <- woe_table(factor(rep(c("a", "b"), each = 50), levels = lv),
                   c(rep(0L, 40), rep(1L, 10), rep(1L, 40), rep(0L, 10)))
  ag <- woe_agreement(ref, cur)
  expect_true(is.na(ag$rho))
  expect_equal(ag$flip_mass, 1)
})
