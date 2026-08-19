test_that("an anchor separates two spells of the same card", {
  frame <- data.table::data.table(
    card1 = c(1, 1, 1, 1),
    D1 = c(0, 1, 0, 1),                       # days since this spell started
    TransactionDT = c(0, 86400, 500 * 86400, 501 * 86400)
  )
  without <- entity_uid(frame, "card1")
  with_anchor <- entity_uid(frame, "card1", anchor = "D1")
  expect_equal(length(unique(without)), 1L)   # one card, one entity
  expect_equal(length(unique(with_anchor)), 2L)  # reissued: two entities
})

test_that("purity is measured against a null that keeps the group sizes", {
  # Labels assigned at random within fixed groups: the observed purity must land
  # inside the null it is compared with, because that is exactly what it is.
  set.seed(71)
  uid <- rep(seq_len(500), each = 4)
  y <- rbinom(2000, 1, 0.2)
  out <- entity_purity(uid, y, B = 100L)
  expect_gt(out$p_value, 0.05)
  expect_equal(out$pure_share, out$null_mean, tolerance = 0.1)
})

test_that("a real entity structure clears the null", {
  set.seed(72)
  uid <- rep(seq_len(500), each = 4)
  entity_fraud <- rbinom(500, 1, 0.2)         # the label belongs to the entity
  y <- rep(entity_fraud, each = 4)
  out <- entity_purity(uid, y, B = 100L)
  expect_equal(out$pure_share, 1)
  expect_lt(out$p_value, 0.05)
  expect_gt(out$pure_share - out$null_mean, 0.3)
})

test_that("singletons are excluded from purity and counted separately", {
  # A key giving every row its own group is homogeneous by definition. Counting
  # those would let it score a perfect 1.0 while explaining nothing.
  uid <- as.character(seq_len(100))
  out <- entity_purity(uid, rbinom(100, 1, 0.3), B = 20L)
  expect_equal(out$singleton_share, 1)
  expect_true(is.na(out$pure_share))
})

test_that("coverage reports the rows a key cannot reach", {
  uid <- c("a", "a", NA, NA, "b")
  cov <- entity_coverage(uid, time = c(0, 86400, 0, 0, 0))
  expect_equal(cov$rows_covered, 0.6)
  expect_equal(cov$entities, 2L)
})

test_that("comparison names the candidate without setting a data.table key", {
  # `data.table(key = ...)` sets the table's key rather than creating a column,
  # and fails on a value that is not a column name.
  set.seed(73); n <- 2000
  win <- list(
    train = data.table::data.table(card1 = sample(200, n, TRUE), addr1 = sample(20, n, TRUE),
                                   TransactionDT = sample(n) * 100, isFraud = rbinom(n, 1, 0.1)),
    label = "isFraud")
  out <- compare_entity_keys(win, list(a = list(columns = "card1", anchor = NULL)),
                             B = 10L, n_sub = n)
  expect_equal(out$candidate, "a")
})

test_that("the pipeline key check compares partitions, not strings", {
  # The two sides use different separators -- SQL formats `1_10_0`, this pastes
  # `1|10|0` -- so what has to match is which rows group together, not the text.
  frame <- data.table::data.table(
    card1 = c(1, 1, 2, NA), addr1 = c(10, 10, 20, 10),
    D1 = c(0, 1, 0, 0), TransactionDT = c(0, 86400, 0, 0),
    client_uid = c("1_10_0", "1_10_0", "2_20_0", NA))
  out <- verify_against_pipeline_key(frame, c("card1", "addr1"), anchor = "D1")
  expect_equal(out$agreement, 1)
  expect_equal(out$groups_here, out$groups_pipeline)
})

test_that("a disagreement is reported rather than rounded away", {
  frame <- data.table::data.table(
    card1 = c(1, 1, 2), addr1 = c(10, 10, 20), D1 = c(0, 1, 0),
    TransactionDT = c(0, 86400, 0),
    client_uid = c("a", "b", "c"))          # pipeline splits what this joins
  expect_lt(verify_against_pipeline_key(frame, c("card1", "addr1"), anchor = "D1")$agreement, 1)
})

test_that("a missing column stops rather than reporting something reassuring", {
  # This returned a polite `available = FALSE` and rendered its note in the
  # notebook on an export that *had* the column -- the caller had simply not
  # asked for it by name. The check was never once run on the data it exists
  # for, and said so in a way that read like a result.
  frame <- data.table::data.table(card1 = 1:3, addr1 = 1:3, D1 = 0, TransactionDT = 0)
  expect_error(
    verify_against_pipeline_key(frame, c("card1", "addr1"), anchor = "D1"),
    "EXCLUDED_COLUMNS")
})
