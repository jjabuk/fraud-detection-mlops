test_that("parallel analysis recovers a known one-factor block", {
  skip_if_not_installed("psych")
  set.seed(61); n <- 3000
  latent <- rnorm(n)
  block <- as.data.table(lapply(1:6, function(i) latent + rnorm(n, sd = 0.3)))
  setnames(block, paste0("v", 1:6))
  win <- list(train = block, features = names(block))
  out <- parallel_analysis(win, names(block), n_sub = n, n_iter = 20L)
  expect_equal(out$components, 1L)
  expect_gt(out$variance_first, 0.7)
})

test_that("parallel analysis separates two independent factors", {
  skip_if_not_installed("psych")
  set.seed(62); n <- 3000
  f1 <- rnorm(n); f2 <- rnorm(n)
  block <- data.table::data.table(
    a = f1 + rnorm(n, sd = .2), b = f1 + rnorm(n, sd = .2), c = f1 + rnorm(n, sd = .2),
    d = f2 + rnorm(n, sd = .2), e = f2 + rnorm(n, sd = .2), f = f2 + rnorm(n, sd = .2))
  win <- list(train = block, features = names(block))
  expect_equal(parallel_analysis(win, names(block), n_sub = n, n_iter = 20L)$components, 2L)
})

test_that("a block too small to analyse says so rather than guessing", {
  win <- list(train = data.table::data.table(a = rnorm(100), b = rnorm(100)),
              features = c("a", "b"))
  out <- parallel_analysis(win, c("a", "b"), n_sub = 100L)
  expect_true(is.na(out$components))
  expect_match(out$note, "fewer than three")
})

test_that("tetrachoric exceeds Pearson on dichotomised data", {
  skip_if_not_installed("psych")
  # Pearson between two dichotomies is bounded below the latent correlation by
  # the marginal split alone. That bound is the reason this function exists.
  set.seed(63); n <- 5000
  latent1 <- rnorm(n); latent2 <- 0.8 * latent1 + rnorm(n, sd = 0.6)
  win <- list(train = data.table::data.table(
    a = as.integer(latent1 > 0.8), b = as.integer(latent2 > 0.8)),
    features = c("a", "b"))
  tc <- tetrachoric_matrix(win, c("a", "b"), n_sub = n)
  pearson <- abs(cor(win$train$a, win$train$b))
  expect_gt(tc$mean_off_diagonal, pearson)
})
