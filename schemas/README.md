# BigQuery Schemas

The `bq_schema_train_transaction.json` and `bq_schema_train_identity.json` files contain the column names and types of the two raw tables, in BigQuery's own JSON schema format. They are **hand-crafted** rather than letting BigQuery infer the types.

They are then **consumed twice**:

- by the ingestion assets, which load the CSV against the pinned schema instead of
  re-running autodetect — so a corrupt or truncated file fails the load rather than
  quietly landing with different types;
- by OpenTofu in [`iaac/`](../iaac/), which declares both tables against these files.

That second consumer is the point. A schema change becomes a `tofu plan` diff somebody
reviews, rather than a table that silently reshaped itself on the next load.