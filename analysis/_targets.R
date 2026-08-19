# The audit graph.
#
# `targets` is to this package what Dagster is to the pipeline: a dependency
# graph with content-addressed invalidation, so re-running after a change
# recomputes the affected audits and nothing else. The two are deliberately not
# connected. Dagster owns everything downstream of the contract; this graph owns
# everything upstream of it, and the contract file is the only thing they share.
#
#   tar_make()          run what is out of date
#   tar_visnetwork()    draw the graph
#   tar_read(tc_report) the scan, as a table

library(targets)

tar_option_set(
  packages = c(
    "arrow", "data.table", "dplyr", "pROC", "Hmisc", "jsonlite",
    "energy", "twosamples", "DescTools", "naniar", "psych", "ineq"
  ),
  format = "rds"
)

invisible(lapply(list.files("R", full.names = TRUE, pattern = "[.]R$"), source))

# The frame *as the model receives it*, which means with the declared derivations
# already applied. `uv run export-audit-frame` produces it. Auditing the raw export
# instead would leave the thirty derived columns unexamined, and eight of them were
# rejected by an audit the last time the whole set was measured.
MODEL_INPUT <- Sys.getenv(
  "FRAUDAUDIT_PARQUET",
  "../data/local/cache/audit_frame.parquet"
)


list(
  tar_target(parquet_file, MODEL_INPUT, format = "file"),
  tar_target(windows, load_windows(
    parquet_file,
    train   = TC_POLICY$train_window,
    holdout = TC_POLICY$holdout_window
  )),
  tar_target(tc_report, time_consistency_scan(
    windows,
    opts = list(
      bins = TC_POLICY$bins, margin = TC_POLICY$margin,
      alpha = TC_POLICY$alpha, min_gini_drop = TC_POLICY$min_gini_drop
    ),
    verbose = FALSE
  )),
  tar_target(v_blocks, load_blocks("../references/column-groups-v.json")),
  tar_target(tc_fragment, time_consistency_fragment(
    tc_report,
    blocks = if (isTRUE(TC_POLICY$reject_by_block)) v_blocks else NULL,
    params = TC_POLICY
  )),
  tar_target(tc_written, write_fragment(tc_fragment, tc_report, dir = "out")),
  tar_target(ds_report, distribution_shift_scan(
    windows,
    B = DS_POLICY$permutations, bins = DS_POLICY$bins, verbose = FALSE
  )),
  tar_target(ds_fragment, distribution_shift_fragment(
    ds_report,
    psi_threshold = DS_POLICY$psi_threshold,
    alpha = DS_POLICY$alpha, params = DS_POLICY
  )),
  tar_target(ds_written, write_fragment(ds_fragment, ds_report, dir = "out")),
  tar_target(rd_report, redundancy_scan(
    windows,
    min_rho2 = RD_POLICY$min_rho2, r2 = RD_POLICY$redun_r2,
    tc_report = tc_report, n_sub = RD_POLICY$rank_subsample
  )),
  tar_target(block_audit, audit_pinned_blocks(attr(rd_report, "rho"), v_blocks)),
  tar_target(rd_fragment, redundancy_fragment(rd_report, params = RD_POLICY)),
  tar_target(rd_written, write_fragment(rd_fragment, rd_report, dir = "out")),
  tar_target(sq_report, segment_qualification_scan(
    windows,
    segment_column = SQ_POLICY$segment_column,
    bins = SQ_POLICY$bins, verbose = FALSE
  )),
  tar_target(sq_fragment, segment_qualification_fragment(
    sq_report,
    reject = SQ_POLICY$reject, params = SQ_POLICY
  )),
  tar_target(sq_written, write_fragment(sq_fragment, sq_report, dir = "out")),

  # Report-only from here down: these describe the table rather than admitting
  # or rejecting anything, so they write tables and no fragment.
  tar_target(categorical_report, categorical_scan(windows, R = 500L, verbose = FALSE)),
  tar_target(missingness_report, missingness_scan(windows, verbose = FALSE)),
  tar_target(dimensionality_report, dimensionality_scan(windows, v_blocks,
    n_iter = 30L, verbose = FALSE
  )),

  # Its own frame: the entity question needs the time axis and the key
  # components, and `windows` excludes the time axis because it is not a feature.
  tar_target(entity_windows, load_windows(
    parquet_file,
    features = c("card1", "card2", "card3", "addr1", "D1", "TransactionDT"),
    train = TC_POLICY$train_window, holdout = TC_POLICY$holdout_window
  )),
  tar_target(entity_report, compare_entity_keys(entity_windows, list(
    "card1"                      = list(columns = "card1", anchor = NULL),
    "card1 + addr1"              = list(columns = c("card1", "addr1"), anchor = NULL),
    "card1 + addr1 + D1"         = list(columns = c("card1", "addr1"), anchor = "D1"),
    "card1 + card2 + addr1 + D1" = list(columns = c("card1", "card2", "addr1"), anchor = "D1")
  ), B = 200L)),
  tar_target(report_tables,
    {
      dir.create("out/tables", recursive = TRUE, showWarnings = FALSE)
      data.table::fwrite(categorical_report, "out/tables/categorical.csv")
      data.table::fwrite(missingness_report, "out/tables/missingness.csv")
      data.table::fwrite(dimensionality_report, "out/tables/dimensionality.csv")
      data.table::fwrite(entity_report, "out/tables/entity_keys.csv")
      # The verdict on the *borrowed* partition. ATTRIBUTION.md claims the pinned
      # V-block grouping was checked against this data rather than taken on trust,
      # and a claim of that kind has to point at a committed number.
      data.table::fwrite(block_audit, "out/tables/pinned_blocks.csv")
      c(
        "out/tables/categorical.csv", "out/tables/missingness.csv",
        "out/tables/dimensionality.csv", "out/tables/entity_keys.csv",
        "out/tables/pinned_blocks.csv"
      )
    },
    format = "file"
  )
)
