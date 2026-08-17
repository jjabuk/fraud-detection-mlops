# BigQuery schemas

`bq_schema_train_transaction.json` and `bq_schema_train_identity.json` pin the column names and
types of the two raw tables in BigQuery's own JSON schema format. They are hand-maintained
rather than left to autodetect.

They have two consumers:

- the ingestion assets, which load the CSV against the pinned schema instead of re-running
  autodetect, so a truncated or corrupt file fails the load rather than landing with different
  types. [`orchestration/raw_load.py`](../src/fraud_detection/orchestration/raw_load.py) also
  compares the live table's types against these files before writing.
- OpenTofu in [`iaac/`](../iaac/), which declares both tables with
  `schema = file("../schemas/...")`.

The second consumer is the point. A schema change becomes a `tofu plan` diff somebody reviews,
rather than a table that reshaped itself on the next load.
