# The contract body.
#
# This is the only file that reads every fragment, and it computes nothing. A
# fragment carries a verdict and the evidence behind it; this turns the set of
# them into one statement about every declared column, in the order the checks
# are stated to apply.
#
# The fingerprint is not written here. `FeatureContract.fingerprint()` on the
# Python side hashes a canonical rendering of the admitted set and the policy,
# and `from_dict` refuses a file whose stored hash disagrees with its contents
# -- that is how a hand-edited contract is caught. Reproducing Python's
# `json.dumps(sort_keys=True)` byte-for-byte from R would make that detector
# depend on two serialisers agreeing forever. R writes the body; Python stamps
# it. That stamping is the cut between this half of the repository and the ML
# half.

#' Declare the columns a contract is about.
#'
#' Source is where a field comes from at scoring time, and it is a deployment
#' fact rather than a statistical one: a `request` field arrives with the
#' transaction, a `retrieved` field has to be fetched, and a `derived` field is
#' computed from the others. A contract that admits a retrieved field is
#' committing to an extra lookup in the scoring path, which is why the count is
#' in the summary.
declare_columns <- function(win, transaction_schema, identity_schema) {
  request <- vapply(jsonlite::fromJSON(transaction_schema, simplifyVector = FALSE),
                    function(f) f$name, character(1))
  retrieved <- vapply(jsonlite::fromJSON(identity_schema, simplifyVector = FALSE),
                      function(f) f$name, character(1))

  dtype_of <- function(x) {
    if (is.logical(x)) "bool"
    else if (is.integer(x)) "int"
    else if (is.numeric(x)) if (all(x[!is.na(x)] == trunc(x[!is.na(x)]))) "int" else "float"
    else "str"
  }

  data.table::rbindlist(lapply(win$features, function(f) {
    data.table::data.table(
      name = f,
      source = if (f %in% request) "request" else if (f %in% retrieved) "retrieved" else "derived",
      dtype = dtype_of(win$train[[f]])
    )
  }))
}

#' Merge fragments into a contract body.
#'
#' First rejection wins, in the order the checks are given. That order is a
#' policy decision, not an implementation detail: it decides which check gets
#' *credit* for a column that several of them would have caught, and therefore
#' what the funnel in the report looks like. It is recorded in the body.
#'
#' A column rejected by two checks is not rejected twice. The alternative --
#' recording every check that fired -- reads better and makes the summary
#' counts stop adding up to the total, which is worse.
#'
#' @param fragments List of fragments, in precedence order.
#' @param declared Table from [declare_columns()].
#' @param policy The full policy every fragment ran under.
build_contract_body <- function(fragments, declared, policy, data_meta = list()) {
  rejections <- data.table::rbindlist(lapply(fragments, function(fr) {
    if (!length(fr$rejections)) return(NULL)
    data.table::rbindlist(lapply(fr$rejections, function(r) data.table::data.table(
      name = r$column, rejected_by = r$check,
      rejected_value = if (is.null(r$value)) NA_real_ else as.numeric(r$value),
      rejected_unit = r$unit
    )))
  }), fill = TRUE)

  first <- if (is.null(rejections) || !nrow(rejections)) {
    data.table::data.table(name = character(0), rejected_by = character(0),
                           rejected_value = numeric(0), rejected_unit = character(0))
  } else {
    rejections[!duplicated(name)]
  }

  cols <- merge(declared, first, by = "name", all.x = TRUE, sort = FALSE)
  cols[, admitted := is.na(rejected_by)]
  cols[is.na(rejected_by), `:=`(rejected_by = "", rejected_unit = "")]

  list(
    version = "1",
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    data = data_meta,
    policy = policy,
    summary = list(
      declared = nrow(cols),
      admitted = sum(cols$admitted),
      rejected = sum(!cols$admitted),
      request_fields = sum(cols$admitted & cols$source == "request"),
      retrieved_fields = sum(cols$admitted & cols$source == "retrieved"),
      derived_fields = sum(cols$admitted & cols$source == "derived"),
      precedence = vapply(fragments, function(f) f$check, character(1))
    ),
    fragments = lapply(fragments, function(f) list(
      check = f$check, tool = f$tool, params = f$params,
      qualification = f$qualification, rejected = length(f$rejections)
    )),
    columns = lapply(seq_len(nrow(cols)), function(i) list(
      name = cols$name[[i]], source = cols$source[[i]], dtype = cols$dtype[[i]],
      admitted = cols$admitted[[i]], rejected_by = cols$rejected_by[[i]],
      rejected_value = if (is.na(cols$rejected_value[[i]])) NULL else cols$rejected_value[[i]],
      rejected_unit = cols$rejected_unit[[i]]
    ))
  )
}

#' Read the fragments a run produced, in precedence order.
read_fragments <- function(dir = "out/fragments", order = NULL) {
  files <- list.files(dir, pattern = "[.]json$", full.names = TRUE)
  frags <- lapply(files, jsonlite::fromJSON, simplifyVector = FALSE)
  names(frags) <- vapply(frags, function(f) f$check, character(1))
  if (is.null(order)) return(frags)
  missing <- setdiff(order, names(frags))
  if (length(missing)) {
    stop("no fragment for: ", paste(missing, collapse = ", "),
         " -- run its notebook or `tar_make()` first", call. = FALSE)
  }
  frags[order]
}

#' Write the declaration, which is what stamping needs.
#'
#' Only the declaration, not the whole body. The body renders the merged contract
#' for a person to read; stamping needs the column list, its sources and its
#' dtypes, and nothing else -- every verdict is already in the fragments. Writing
#' the whole thing would put a near-copy of `references/feature-contract.json`
#' into the repository, and a second copy of a decided thing is a thing that can
#' disagree with the first.
#'
#' Source is the part that cannot be inferred from the data: no statistic knows
#' whether a column arrives with the request or has to be looked up, and getting
#' it wrong is how a request schema ends up asking callers for features they
#' cannot have.
write_declaration <- function(declared, data_meta = list(),
                              path = "out/declaration.json") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  payload <- list(
    version = "1",
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    data = data_meta,
    columns = lapply(seq_len(nrow(declared)), function(i) list(
      name = declared$name[[i]], source = declared$source[[i]],
      dtype = declared$dtype[[i]]
    ))
  )
  jsonlite::write_json(payload, path, auto_unbox = TRUE, pretty = TRUE,
                       digits = 10, null = "null", na = "null")
  invisible(path)
}
