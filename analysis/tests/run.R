# One command for the suite, so CI and a laptop run the same thing.
#
# `test_dir` rather than `test_local`: the functions in R/ are a package's worth of
# code with a DESCRIPTION, but they are sourced rather than installed. Installing
# would mean maintaining a NAMESPACE for a package nothing outside this directory
# imports, and would put a build step between an edit and its test.
#
# Every test here is synthetic. Nothing reads the parquet, nothing reaches a
# warehouse, and nothing needs a credential -- which is what lets this run on a
# pull request rather than only on a machine that has the data.

suppressPackageStartupMessages({
  library(testthat)
  library(arrow)
  library(data.table)
  library(dplyr)
  library(pROC)
  library(Hmisc)
  library(jsonlite)
  library(energy)
  library(twosamples)
  library(DescTools)
  library(naniar)
  library(psych)
  library(ineq)
})

invisible(lapply(list.files("R", full.names = TRUE, pattern = "[.]R$"), source))

testthat::test_dir("tests/testthat", stop_on_failure = TRUE, reporter = "summary")
