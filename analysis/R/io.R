# Reading the audited frame, and cutting the two time windows out of it.
#
# The windows are the whole experiment: every verdict in this package is a
# statement about a feature measured early and measured late, so a window that
# drifts by one row makes two runs incomparable. They are therefore defined
# here, once, and pinned into the contract fragment that the audit writes.

#' Quantile with polars' `nearest` semantics.
#'
#' Deliberately not `stats::quantile(type = 7)`. The Python implementation this
#' package replaces cut its windows with `polars` defaults, which is
#' `interpolation="nearest"` -- the order statistic at `round(p * (n - 1))`, not
#' a linear interpolation between two of them. On this data the difference is a
#' single row at the 0.83 edge. It is pinned because the window definition
#' belongs to the contract, and a contract that moves because a library default
#' moved is a contract nobody can reproduce.
#'
#' @param x Numeric vector. Sorted internally.
#' @param p Probability in `[0, 1]`.
#' @return A scalar of the same type as `x`.
quantile_nearest <- function(x, p) {
  stopifnot(is.numeric(x), length(x) > 0L, p >= 0, p <= 1)
  xs <- sort(x)
  xs[[round(p * (length(xs) - 1L)) + 1L]]
}

#' Row indices of an early and a late window, by quantiles of the time column.
#'
#' The gap between `train[2]` and `holdout[1]` is deliberate: skipping the middle
#' mimics the label-maturity lag you actually have at deployment, where the most
#' recent period has not finished being labelled yet.
#'
#' Windows are half-open `[lo, hi)`, except a window whose upper bound is 1.0,
#' which includes it -- otherwise the single largest row would be silently
#' dropped from the holdout.
#'
#' **These are quantiles of the time value, so the windows hold equal numbers of
#' rows and unequal amounts of calendar.** On this data the first 17% of rows is
#' 22.6 days and the last 17% is 35.1 days -- transaction volume falls over the
#' period, so the later window covers 1.56x the span. That is a real confound for
#' any audit that compares the two: part of what reads as "this feature changed"
#' is "the later window watched for longer".
#'
#' It is left as it is, for one reason and with one mitigation. The reason is
#' that the alternative -- equal calendar spans -- makes the windows hold
#' unequal numbers of rows, and every statistic downstream then has different
#' power in the two windows, which is a worse confound for a test than an
#' unequal span is for an estimate. The mitigation is that the reproducibility
#' check varies the window and reports whether the verdict survives; a rejection
#' driven by the span rather than by the feature does not.
#'
#' What this rules out is reading the audit as a statement about a fixed period.
#' It is a statement about the first and last sixth of the *traffic*.
#'
#' @param time Numeric vector, the time axis.
#' @param train,holdout Length-2 numeric, quantile bounds within `[0, 1]`.
#' @return List of two integer index vectors, and the four edge values used.
time_windows <- function(time, train = c(0.0, 0.17), holdout = c(0.83, 1.0)) {
  for (w in list(train = train, holdout = holdout)) {
    if (!(w[[1]] >= 0 && w[[1]] < w[[2]] && w[[2]] <= 1)) {
      stop("window ", deparse(w), " is not an increasing range within [0, 1]", call. = FALSE)
    }
  }
  if (train[[2]] > holdout[[1]]) {
    stop("train window overlaps holdout; the check is only meaningful when ",
      "training strictly precedes scoring",
      call. = FALSE
    )
  }

  xs <- sort(time)
  q <- function(p) xs[[round(p * (length(xs) - 1L)) + 1L]]
  edges <- c(q(train[[1]]), q(train[[2]]), q(holdout[[1]]), q(holdout[[2]]))

  cut_idx <- function(lo, hi, closed_right) {
    if (closed_right) which(time >= lo & time <= hi) else which(time >= lo & time < hi)
  }

  list(
    train   = cut_idx(edges[[1]], edges[[2]], closed_right = train[[2]] == 1.0),
    holdout = cut_idx(edges[[3]], edges[[4]], closed_right = holdout[[2]] == 1.0),
    edges   = edges
  )
}

#' Columns the audit is allowed to see.
#'
#' Mirrors `fraud_detection.core.schema.EXCLUDED_COLUMNS`. The identifier, the
#' time axis and the label are not features; `client_uid` is an entity key with
#' 217,735 levels, and a model that memorised it would have memorised clients it
#' can never meet again.
EXCLUDED_COLUMNS <- c(
  "TransactionID",
  "TransactionDT",
  "isFraud",
  "split",
  "card_seen_in_train",
  "client_uid",
  "origin"
)

#: The unique key, used to break ties when restoring a deterministic row order.
ID_COLUMN <- "TransactionID"

#' Load the two windows out of the model-input parquet.
#'
#' Reads the time axis and the label first, decides the row sets, and only then
#' reads the features -- two windows of ~100k rows instead of 590,540, which is
#' the difference between a 600 MB frame and a 2.2 GB one.
#'
#' @param path Parquet file written from `features.model_input`.
#' @param label,time Column names.
#' @param features Character vector, or NULL for "everything not excluded".
#' @return List with `train`, `holdout` (data.tables), `features`, `edges`.
load_windows <- function(path,
                         label = "isFraud",
                         time = "TransactionDT",
                         features = NULL,
                         train = c(0.0, 0.17),
                         holdout = c(0.83, 1.0)) {
  ds <- arrow::open_dataset(path)
  all_cols <- names(ds)

  axis <- data.table::as.data.table(
    arrow::read_parquet(path, col_select = dplyr::all_of(c(time, label)))
  )
  win <- time_windows(axis[[time]], train = train, holdout = holdout)

  if (is.null(features)) features <- setdiff(all_cols, EXCLUDED_COLUMNS)
  missing <- setdiff(features, all_cols)
  if (length(missing)) {
    stop("not in the parquet: ", paste(utils::head(missing, 5), collapse = ", "), call. = FALSE)
  }

  # Pushed down to the parquet reader rather than filtered after the fact. The
  # full frame is 502 columns over 590,540 rows; the two windows together are a
  # third of that, and the difference is a 2 GB allocation this machine does not
  # need to make.
  #
  # **Then sorted, and that is not cosmetic.** A filtered Arrow scan is
  # multi-threaded and does not promise to return rows in file order: two runs
  # get the same rows in a different sequence. Nothing that reads every row
  # notices -- an AUC and a bin count do not care -- but every audit that
  # subsamples does, because `set.seed(0); sample.int(n, k)` then picks the same
  # *positions* out of a different *ordering* and therefore a different sample.
  #
  # That made the redundancy audit, and with it the contract, differ between two
  # runs of identical code on identical data. Sorting on the time axis, ties
  # broken by the identifier, restores the file's own order and makes every
  # seeded subsample reproducible.
  e <- win$edges
  sort_keys <- intersect(c(time, ID_COLUMN), all_cols)
  wanted <- unique(c(features, label, sort_keys))
  window_frame <- function(lo, hi, closed_right) {
    q <- arrow::open_dataset(path)
    q <- if (closed_right) {
      dplyr::filter(q, .data[[time]] >= lo & .data[[time]] <= hi)
    } else {
      dplyr::filter(q, .data[[time]] >= lo & .data[[time]] < hi)
    }
    frame <- data.table::as.data.table(dplyr::collect(dplyr::select(q, dplyr::all_of(wanted))))
    data.table::setorderv(frame, sort_keys)
    drop <- setdiff(sort_keys, c(features, label))
    if (length(drop)) frame[, (drop) := NULL]
    frame[]
  }

  list(
    train    = window_frame(e[[1]], e[[2]], closed_right = train[[2]] == 1.0),
    holdout  = window_frame(e[[3]], e[[4]], closed_right = holdout[[2]] == 1.0),
    features = features,
    label    = label,
    edges    = win$edges,
    windows  = list(train = train, holdout = holdout)
  )
}
