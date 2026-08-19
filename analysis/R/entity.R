# Who is the customer, when the dataset does not say?
#
# The data names no customer. One can be reconstructed by combining columns --
# a card, an address, and an anchor that separates two spells of the same card --
# but a reconstruction that looks plausible and one that is correct are different
# things, and guessing wrong shatters real customers into unrelated groups.
#
# The test is label purity. When a label propagates within an entity (one
# fraudulent card marks that card's later transactions too), a correct
# reconstruction produces label-homogeneous groups and a wrong one does not.
#
# Purity alone does not carry the argument. A figure of 0.97 sounds decisive and
# is not interpretable on its own:
# at a 3.5% base rate, groups of size two are homogeneous 93% of the time by
# chance alone. Every purity here is therefore reported against what the same
# group *sizes* produce when the labels are shuffled, which is the only version
# of the number that carries evidence.
#
# The key this returns is the one the pipeline then uses. Its construction lives
# in Python, in `evaluation/entity_purity.py`, because it runs on every training
# job; the question of whether it is the right key is answered here.

#' Build a candidate entity key.
#'
#' @param columns Columns pasted together to form the key.
#' @param anchor Optional column whose value, floored to days, separates two
#'   spells of the same card. Without one, a card reissued to a different holder
#'   is a single entity.
#' @param time Time column, needed when an anchor is given.
entity_uid <- function(frame, columns, anchor = NULL, time = "TransactionDT") {
  parts <- lapply(columns, function(c) as.character(frame[[c]]))
  if (!is.null(anchor)) {
    # The anchor is a "days since" column: subtracting it from the transaction
    # day gives a per-entity start date that is constant within a spell and
    # changes when the card is reissued.
    start_day <- floor(frame[[time]] / 86400) - frame[[anchor]]
    parts <- c(parts, list(as.character(start_day)))
  }
  do.call(paste, c(parts, sep = "|"))
}

#' Label purity of the groups a key induces, against a shuffled null.
#'
#' Singletons are excluded from the purity figure and counted separately. A group
#' of one is homogeneous by definition and tells us nothing about whether the key
#' found a customer; a key that gives every row its own group would otherwise
#' score a perfect 1.0 while explaining nothing.
#'
#' @param B Label shuffles forming the null. Shuffling the labels while keeping
#'   the groups fixed preserves both the base rate and the group size
#'   distribution, so the null answers exactly "what would this key's purity be
#'   if the entity meant nothing".
entity_purity <- function(uid, label, B = 200L, seed = 0L) {
  ok <- !is.na(uid) & !is.na(label)
  uid <- uid[ok]; label <- as.integer(label[ok])

  # `by = "uid"` quoted, deliberately. An unquoted `uid` here resolves to the
  # character vector in scope rather than to the column, and data.table then
  # reads its *values* as column names.
  dt <- data.table::data.table(uid = uid, y = label)
  groups <- dt[, .(n = .N, pos = sum(y)), by = "uid"]
  multi <- groups[n > 1L]
  if (!nrow(multi)) {
    return(list(pure_share = NA_real_, null_mean = NA_real_, null_q99 = NA_real_,
                p_value = NA_real_, groups = nrow(groups), multi_groups = 0L,
                singleton_share = 1, rows_covered = length(uid)))
  }

  observed <- mean(multi$pos == 0L | multi$pos == multi$n)

  set.seed(seed)
  draws <- vapply(seq_len(B), function(i) {
    # The whole label vector is permuted once per draw, so the base rate and the
    # group sizes are both preserved and only the assignment of labels to
    # entities is destroyed. Resampling within groups instead would change the
    # base rate per draw and compare against the wrong null.
    dt[, y_perm := sample(y)]
    shuffled <- dt[, .(n = .N, pos = sum(y_perm)), by = "uid"][n > 1L]
    mean(shuffled$pos == 0L | shuffled$pos == shuffled$n)
  }, numeric(1))

  list(
    pure_share = observed,
    null_mean = mean(draws),
    null_q99 = unname(stats::quantile(draws, 0.99)),
    # One-sided: the claim is that the key finds *more* homogeneity than chance.
    p_value = (1 + sum(draws >= observed)) / (B + 1),
    groups = nrow(groups), multi_groups = nrow(multi),
    singleton_share = mean(groups$n == 1L), rows_covered = length(uid)
  )
}

#' How much of the table a key actually reaches, and how concentrated it is.
#'
#' A key built from columns that are often null covers only the rows where they
#' are populated. Coverage is what stops a key being judged on the rows it
#' happens to work for.
entity_coverage <- function(uid, time) {
  reachable <- !is.na(uid)
  sizes <- table(uid[reachable])
  spans <- data.table::data.table(uid = uid[reachable], t = time[reachable])[
    , .(span_days = (max(t) - min(t)) / 86400), by = "uid"]
  list(
    rows_covered = mean(reachable),
    entities = length(sizes),
    median_size = stats::median(as.numeric(sizes)),
    gini = ineq::Gini(as.numeric(sizes)),
    median_span_days = stats::median(spans$span_days),
    multi_row_share = mean(as.numeric(sizes) > 1)
  )
}

#' Does a candidate reproduce the key the pipeline actually builds?
#'
#' The chosen key is built once, in SQL, and carried into `model_input` as
#' `client_uid`. This closes the loop: the candidate this analysis recommends has
#' to agree with the column the warehouse produces, row for row, including which
#' rows have no key at all.
#'
#' A disagreement is not a tie to be broken by preference. It means the statement
#' and the analysis are describing different entities, and every purity figure
#' above is then about something other than what the model splits on.
#'
#' A missing column raises. It used to return `available = FALSE` with a polite
#' note, and the note rendered in the notebook on an export that *had* the
#' column -- because `client_uid` is in `EXCLUDED_COLUMNS`, so the caller has to
#' ask for it by name, and the caller did not. The check was therefore written,
#' tested against a synthetic frame, and never once run on the data it exists
#' for, while reporting something reassuring.
#'
#' A soft failure that looks like a result is worse than no check. This one stops.
#'
#' @return A list with the agreement rate, the group counts and the null shares.
verify_against_pipeline_key <- function(frame, columns, anchor = NULL,
                                        time = "TransactionDT",
                                        uid_column = "client_uid") {
  if (!uid_column %in% names(frame)) {
    stop("`", uid_column, "` is not in the frame. It is an entity key, so it is in ",
         "`EXCLUDED_COLUMNS` and the default feature set leaves it out -- load it by ",
         "name: `load_windows(path, features = c(..., \"", uid_column, "\"))`. If the ",
         "export genuinely predates the column, re-run the pipeline and export again.",
         call. = FALSE)
  }

  mine <- entity_uid(frame, columns, anchor = anchor, time = time)
  mine[Reduce(`|`, lapply(columns, function(c) is.na(frame[[c]])))] <- NA
  theirs <- as.character(frame[[uid_column]])

  # Compared as a partition, not as strings: the two use different separators, so
  # what has to match is which rows are grouped together and which have no group.
  both_null <- is.na(mine) & is.na(theirs)
  both_set <- !is.na(mine) & !is.na(theirs)
  agree <- both_null | (both_set &
    (as.integer(factor(mine)) == as.integer(factor(theirs))))

  list(
    agreement = mean(agree),
    null_here = mean(is.na(mine)),
    null_pipeline = mean(is.na(theirs)),
    groups_here = length(unique(stats::na.omit(mine))),
    groups_pipeline = length(unique(stats::na.omit(theirs)))
  )
}

#' Compare candidate keys on the same rows.
#'
#' The comparison is the point. Any single key's purity is uninterpretable; what
#' decides the reconstruction is whether adding a component buys homogeneity that
#' the shuffled null does not already give away.
compare_entity_keys <- function(win, candidates, label = NULL, time = "TransactionDT",
                                B = 100L, n_sub = 100000L, seed = 0L) {
  label <- label %||% win$label
  frame <- win$train
  if (nrow(frame) > n_sub) {
    set.seed(seed)
    frame <- frame[sample.int(nrow(frame), n_sub)]
  }
  y <- as.integer(frame[[label]])
  # Not `t`: an absent column would make that name fall through to `base::t`,
  # and the failure surfaces much later as "invalid 'type' (closure)".
  time_values <- frame[[time]]
  if (is.null(time_values)) {
    stop("the frame has no `", time, "` column — the entity comparison needs the ",
         "time axis, which `load_windows` excludes from features by default. Load ",
         "it explicitly with `features = c(..., \"", time, "\")`.", call. = FALSE)
  }

  rows <- lapply(seq_along(candidates), function(i) {
    spec <- candidates[[i]]
    uid <- entity_uid(frame, spec$columns, anchor = spec$anchor, time = time)
    uid[Reduce(`|`, lapply(spec$columns, function(c) is.na(frame[[c]])))] <- NA
    pur <- entity_purity(uid, y, B = B, seed = i)
    cov <- entity_coverage(uid, time_values)
    data.table::data.table(
      # Not `key =`: that is a reserved argument of `data.table()` and sets the
      # table's key rather than creating a column, which fails on a value that
      # is not a column name.
      candidate = names(candidates)[[i]],
      components = paste(c(spec$columns, spec$anchor), collapse = " + "),
      rows_covered = cov$rows_covered, entities = cov$entities,
      median_size = cov$median_size, singleton_share = pur$singleton_share,
      gini = cov$gini, median_span_days = cov$median_span_days,
      pure_share = pur$pure_share, null_mean = pur$null_mean,
      lift_over_null = pur$pure_share - pur$null_mean, p_value = pur$p_value
    )
  })
  out <- data.table::rbindlist(rows)
  data.table::setorder(out, -lift_over_null)
  out[]
}
