# What leaves this package.
#
# A fragment is a verdict plus the evidence that produced it, in the shape the
# Python contract assembler already merges. Nothing here decides admission --
# that is the contract's job, and it needs every check's fragment before it can.
# Nothing here computes a statistic either; a fragment is a projection of a
# report that was computed elsewhere and can be reread.
#
# The fingerprint is deliberately absent. `FeatureContract.fingerprint()` hashes
# a canonical JSON rendering of the admitted set and the policy, and `from_dict`
# rejects a file whose stored hash disagrees with its contents -- that is how a
# hand-edited contract is caught. Reproducing Python's `json.dumps(sort_keys=
# True)` byte-for-byte from R would put that detector at the mercy of two
# serialisers agreeing forever. R writes the body; Python stamps it.

#' Expand column rejections to whole blocks.
#'
#' A V block is a set of columns that move together, taken from published work
#' and pinned in `references/column-groups-v.json`. When any member inverts, the
#' block is rejected as a unit and every member carries the worst member's
#' delta: partial admission of a block would keep the columns that happened to
#' sit below the margin this month, which is a threshold artefact rather than a
#' finding about the data.
#'
#' @param rejected data.table with `feature` and `value`.
#' @param blocks Named list of character vectors.
#' @return data.table of `column`, `value`, `unit`.
expand_blocks <- function(rejected, blocks) {
  by_column <- stats::setNames(rejected$value, rejected$feature)
  out <- list()
  claimed <- character(0)

  for (name in names(blocks)) {
    members <- blocks[[name]]
    hits <- intersect(members, names(by_column))
    if (!length(hits)) next
    worst <- min(by_column[hits], na.rm = TRUE)
    members <- members[!is.na(members)]
    out[[length(out) + 1L]] <- data.table::data.table(
      column = members, value = worst, unit = paste0("block:", name)
    )
    claimed <- c(claimed, members)
  }

  loose <- setdiff(names(by_column), claimed)
  if (length(loose)) {
    out[[length(out) + 1L]] <- data.table::data.table(
      column = loose, value = unname(by_column[loose]), unit = "column"
    )
  }
  if (!length(out)) return(data.table::data.table(column = character(0), value = numeric(0), unit = character(0)))
  data.table::rbindlist(out)
}

#' Build the time-consistency fragment from a scan.
#'
#' @param report Output of [time_consistency_scan()].
#' @param blocks Named list of V blocks, or NULL to reject column by column.
#' @param params The policy this scan ran under, recorded so a later reader can
#'   tell which numbers made the verdict.
time_consistency_fragment <- function(report, blocks = NULL, params = list()) {
  v <- report$verdict
  rejected <- report[v == "inverted", .(feature, value = delta)]
  rows <- expand_blocks(rejected, blocks %||% list())

  list(
    check = "time_consistency",
    tool = "fraudaudit::time_consistency_scan",
    # `modifyList`, not `c()`: concatenating two lists that share a key keeps
    # both copies, and the duplicate survives into the JSON where it reappears
    # on the next read as `alpha.1`.
    params = as.list(params),
    qualification = list(
      features_scanned = nrow(report),
      degenerate = sum(v == "degenerate"),
      weak = sum(v == "weak"),
      inverted_columns = nrow(rejected),
      rejections_by_block = sum(rows$unit != "column"),
      # The two numbers that say how much the multiplicity correction cost, so
      # a reader can see whether the correction is doing any work at all.

      median_flip_mass_inverted = if (nrow(rejected)) {
        stats::median(report[v == "inverted"]$woe_flip_mass, na.rm = TRUE)
      } else NA_real_
    ),
    rejections = lapply(seq_len(nrow(rows)), function(i) list(
      column = rows$column[[i]], check = "time_consistency",
      value = rows$value[[i]], unit = rows$unit[[i]]
    ))
  )
}

#' Write a fragment as JSON and its report as CSV.
#'
#' Both, always. The JSON is what the contract reads; the CSV is what a person
#' reads when they want to know why a column they cared about was dropped, and
#' a verdict nobody can look up is not evidence.
write_fragment <- function(fragment, report, dir = "analysis/out", name = fragment$check) {
  dir.create(file.path(dir, "fragments"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(dir, "tables"), recursive = TRUE, showWarnings = FALSE)
  json_path <- file.path(dir, "fragments", paste0(name, ".json"))
  csv_path  <- file.path(dir, "tables", paste0(name, ".csv"))
  # `na = "null"` matters: the default renders a missing number as the string
  # "NA", which is valid JSON and a type error on the other side of the seam.
  jsonlite::write_json(fragment, json_path, auto_unbox = TRUE, pretty = TRUE,
                       digits = 10, null = "null", na = "null")
  data.table::fwrite(report, csv_path)
  invisible(c(fragment = json_path, report = csv_path))
}

#' Load the pinned V-block partition, flattened to one vector per block.
#'
#' The file records each block as a list of correlation groups; the
#' time-consistency check rejects at block granularity, so the groups are
#' flattened. Matches `feature_audit._blocks`.
#'
#' The partition itself is human judgement taken from published work and pinned
#' rather than re-derived -- see the `provenance` key in the file. Whether it
#' still holds on this data is the redundancy audit's question, not this one's.
load_blocks <- function(path = "references/column-groups-v.json") {
  doc <- jsonlite::fromJSON(path, simplifyVector = FALSE)$blocks
  stats::setNames(
    lapply(doc, function(b) unlist(b$groups, use.names = FALSE)),
    vapply(doc, function(b) b$block, character(1))
  )
}
