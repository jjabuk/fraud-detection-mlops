# Which columns are saying the same thing twice?
#
# Two different claims live under the word "redundant" and the audit keeps them
# apart. A column can be *correlated* with a neighbour -- a pairwise statement,
# symmetric, and no reason on its own to drop either. Or it can be *predictable*
# from the others taken together -- an asymmetric statement, and the one that
# actually justifies removal, because a column a flexible fit reconstructs from
# its neighbours contributes nothing they do not already carry.
#
# Both are answered without a model in the machine-learning sense: variable
# clustering is a hierarchy over a rank-correlation matrix, and redundancy
# analysis fits restricted cubic splines by least squares. Both print.
#
# References: Harrell (2015), *Regression Modeling Strategies*, ch. 4.7 on
# variable clustering and redundancy analysis; Sarle (1990) on the VARCLUS
# procedure this follows.

#' Spearman rank correlation over a row subsample.
#'
#' Rank rather than Pearson because most of this table is zero-inflated counts
#' with heavy tails, where a Pearson coefficient is dominated by a handful of
#' extreme rows.
#'
#' Ranked once per column and then correlated as a matrix, rather than asked for
#' `method = "spearman", use = "pairwise.complete.obs"`. Those options make R
#' rank each pair separately -- 111,000 pairs over 472 columns, which does not
#' finish. Ranking up front turns the whole thing into one matrix product.
#'
#' Missing values take the mean rank, which makes missingness neutral for the
#' correlation rather than a category that binds columns together. That is a
#' choice, and it is the conservative one here: null-pattern families are what
#' the pinned V-block partition already encodes, and letting them drive the
#' clustering too would find the same structure twice and call it corroboration.
#'
#' @param n_sub Rows to sample. A correlation estimated on 20,000 rows has a
#'   standard error near 0.007, far below any threshold worth cutting the
#'   hierarchy at, so more rows buy precision the decision cannot use.
spearman_matrix <- function(win, features = NULL, n_sub = 20000L, seed = 0L) {
  features <- features %||% win$features
  numeric_features <- features[vapply(features, function(f) {
    is.numeric(win$train[[f]]) || is.logical(win$train[[f]])
  }, logical(1))]

  set.seed(seed)
  idx <- sample.int(nrow(win$train), min(n_sub, nrow(win$train)))
  m <- as.matrix(win$train[idx, ..numeric_features])
  storage.mode(m) <- "double"

  keep <- apply(m, 2, function(col) {
    v <- col[!is.na(col)]
    length(v) > 100L && stats::var(v) > 0
  })
  m <- m[, keep, drop = FALSE]

  ranked <- apply(m, 2, function(col) {
    r <- rank(col, na.last = "keep", ties.method = "average")
    r[is.na(r)] <- mean(r, na.rm = TRUE)
    r
  })
  constant <- apply(ranked, 2, stats::var) == 0
  ranked <- ranked[, !constant, drop = FALSE]

  rho <- suppressWarnings(stats::cor(ranked))
  rho[!is.finite(rho)] <- 0
  diag(rho) <- 1
  rho
}

#' Cut a variable-clustering hierarchy into groups.
#'
#' The distance is 1 - rho^2, so it is the *shared variance* that binds a group
#' rather than the sign of the correlation -- two columns that move exactly
#' opposite each other carry the same information and belong together.
#'
#' Average linkage rather than complete: complete linkage requires every pair in
#' a group to clear the threshold, which on a block of forty V columns splits
#' families that are plainly one family over a single loose member.
#'
#' @param min_rho2 Shared variance at which two columns count as the same
#'   column. Stated, recorded in the fragment, and varied in the notebook --
#'   a threshold nobody varied is a threshold nobody tested.
variable_clusters <- function(rho, min_rho2 = 0.8) {
  d <- stats::as.dist(1 - rho^2)
  tree <- stats::hclust(d, method = "average")
  memb <- stats::cutree(tree, h = 1 - min_rho2)
  list(tree = tree, membership = memb, groups = split(names(memb), memb))
}

#' Pick the column that stays and say why.
#'
#' Least missing first, then the strongest information value among the
#' survivors. Missingness leads because a representative that is null in 86% of
#' rows represents its group only in the rows it is present for, whichever way
#' its information value reads.
choose_representatives <- function(groups, win, tc_report = NULL) {
  null_rate <- vapply(unlist(groups, use.names = FALSE),
                      function(f) mean(is.na(win$train[[f]])), numeric(1))
  iv <- if (!is.null(tc_report)) {
    stats::setNames(tc_report$iv_train, tc_report$feature)
  } else stats::setNames(numeric(0), character(0))

  rows <- lapply(names(groups), function(g) {
    members <- groups[[g]]
    score <- data.table::data.table(
      column = members,
      null_rate = null_rate[members],
      iv = unname(ifelse(members %in% names(iv), iv[members], NA_real_))
    )
    # `column` is the final key, and it is not decoration. Two members of a group
    # frequently share a null rate and an information value to four decimals, and
    # without an explicit tiebreak the winner is whichever the input happened to
    # list first -- so the same data produces a different contract depending on
    # the order it arrived in. Sorting on the name makes the choice arbitrary and
    # *stated* rather than arbitrary and hidden.
    data.table::setorder(score, null_rate, -iv, column)
    data.table::data.table(
      group = g, size = length(members), column = members,
      representative = score$column[[1]],
      is_representative = members == score$column[[1]],
      null_rate = null_rate[members],
      iv = unname(ifelse(members %in% names(iv), iv[members], NA_real_))
    )
  })
  data.table::rbindlist(rows)
}

#' Which columns a flexible fit reconstructs from the others in their group.
#'
#' `Hmisc::redun` regresses each column on the rest using restricted cubic
#' splines and removes, one at a time, whichever is best predicted -- flexible
#' enough to catch a non-linear restatement, parametric enough that the R^2 it
#' reports is the ordinary one.
#'
#' Run inside clusters rather than over the whole table, and that is a
#' complexity argument, not a statistical one: the procedure is quadratic in
#' columns, and a group of forty is instant where 472 does not finish.
#'
#' Splines are attempted and then given up on per group. Most of this table is
#' zero-inflated discrete counts -- 4,176 zeros and a long thin tail is typical
#' -- and a restricted cubic spline needs three distinct interior knots it
#' cannot find there. Where that happens the fit falls back to linear, which is
#' the honest description of a column taking four values anyway, and the report
#' records which basis produced each verdict rather than quietly mixing them.
#'
#' @param max_group Groups larger than this are skipped and reported as such,
#'   rather than silently truncated.
redundancy_within_groups <- function(groups, win, r2 = 0.9, n_sub = 10000L,
                                     max_group = 25L, seed = 0L) {
  set.seed(seed)
  idx <- sample.int(nrow(win$train), min(n_sub, nrow(win$train)))

  rows <- lapply(names(groups), function(g) {
    members <- groups[[g]]
    if (length(members) < 2L) {
      return(data.table::data.table(group = g, column = members,
                                    predictable = FALSE, r2 = NA_real_, note = "singleton"))
    }
    if (length(members) > max_group) {
      return(data.table::data.table(group = g, column = members, predictable = NA,
                                    r2 = NA_real_, note = "group larger than max_group"))
    }
    frame <- as.data.frame(win$train[idx, ..members])
    frame <- frame[, vapply(frame, function(c) {
      v <- c[!is.na(c)]; length(v) > 100L && stats::var(as.numeric(v)) > 0
    }, logical(1)), drop = FALSE]
    if (ncol(frame) < 2L) {
      return(data.table::data.table(group = g, column = members, predictable = FALSE,
                                    r2 = NA_real_, note = "not enough usable columns"))
    }
    # `redun` reports its knot troubles by printing frequency tables, one per
    # offending column, which on this table is thousands of lines of noise
    # around a result that is fine.
    quiet_redun <- function(nk) {
      out <- NULL
      utils::capture.output(
        out <- suppressWarnings(try(
          Hmisc::redun(~ ., data = frame, r2 = r2, nk = nk, allcat = FALSE),
          silent = TRUE))
      )
      out
    }

    basis <- "spline"
    fit <- quiet_redun(3L)
    if (inherits(fit, "try-error") || is.null(fit$Out)) {
      basis <- "linear"
      fit <- quiet_redun(0L)
    }
    if (inherits(fit, "try-error") || is.null(fit$Out)) {
      return(data.table::data.table(group = g, column = members, predictable = NA,
                                    r2 = NA_real_, note = "redun failed"))
    }
    data.table::data.table(
      group = g, column = names(frame),
      predictable = names(frame) %in% fit$Out,
      r2 = unname(fit$rsq[match(names(frame), fit$Out)]),
      note = basis
    )
  })
  data.table::rbindlist(rows, fill = TRUE)
}

#' The full redundancy report.
redundancy_scan <- function(win, features = NULL, min_rho2 = 0.8, r2 = 0.9,
                            tc_report = NULL, n_sub = 20000L) {
  rho <- spearman_matrix(win, features, n_sub = n_sub)
  clusters <- variable_clusters(rho, min_rho2 = min_rho2)
  reps <- choose_representatives(clusters$groups, win, tc_report = tc_report)
  redun <- redundancy_within_groups(clusters$groups, win, r2 = r2)

  out <- merge(reps, redun[, .(column, predictable, redun_r2 = r2, redun_note = note)],
               by = "column", all.x = TRUE)
  data.table::setorder(out, -size, group, -is_representative)
  attr(out, "tree") <- clusters$tree
  attr(out, "rho") <- rho
  out[]
}

#' Does the pinned V-block partition hold on this data?
#'
#' The partition in `references/column-groups-v.json` is human judgement read
#' off correlation heatmaps in published work, taken as given rather than
#' re-derived. Whether it still describes *this* data is a separate question,
#' and this answers it: for each pinned block, the mean shared variance inside
#' it against the mean shared variance from its members to everything else.
#'
#' A block tighter inside than out is doing its job. One that is not has been
#' inherited rather than verified.
audit_pinned_blocks <- function(rho, blocks) {
  cols <- colnames(rho)
  rows <- lapply(names(blocks), function(name) {
    members <- intersect(blocks[[name]], cols)
    if (length(members) < 2L) return(NULL)
    outside <- setdiff(cols, members)
    inside <- rho[members, members]^2
    within <- mean(inside[upper.tri(inside)])
    between <- if (length(outside)) mean(rho[members, outside, drop = FALSE]^2) else NA_real_
    data.table::data.table(
      block = name, n_members = length(members),
      mean_rho2_within = within, mean_rho2_between = between,
      ratio = within / between, holds = within > between
    )
  })
  out <- data.table::rbindlist(Filter(Negate(is.null), rows))
  data.table::setorder(out, ratio)
  out[]
}

#' Build the redundancy fragment.
#'
#' Only non-representatives that a flexible fit also reconstructs are rejected.
#' Correlation alone is not enough: it is symmetric, so it names a pair without
#' saying which member to drop, and the choice of representative is what turns
#' it into a decision.
redundancy_fragment <- function(report, params = list()) {
  rejected <- report[!is_representative & isTRUE_na(predictable)]
  list(
    check = "redundancy",
    tool = "fraudaudit::redundancy_scan",
    params = params,
    qualification = list(
      columns_clustered = nrow(report),
      groups = length(unique(report$group)),
      singletons = sum(report$size == 1L),
      largest_group = max(report$size),
      non_representatives = sum(!report$is_representative),
      also_predictable = nrow(rejected),
      # The gap between those two is the whole argument for asking twice: the
      # columns that correlate with a neighbour but that nothing reconstructs.
      correlated_not_predictable = sum(!report$is_representative) - nrow(rejected)
    ),
    rejections = lapply(seq_len(nrow(rejected)), function(i) list(
      column = rejected$column[[i]], check = "redundancy",
      value = rejected$redun_r2[[i]], unit = paste0("group:", rejected$group[[i]])
    ))
  )
}

isTRUE_na <- function(x) !is.na(x) & x
