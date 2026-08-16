resource "google_bigquery_dataset" "raw" {
  dataset_id  = "raw"
  location    = var.bq_location
  description = "Raw IEEE-CIS ingest data loaded from CSV files"
  labels      = var.labels

  delete_contents_on_destroy = false

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_bigquery_table" "ieee_train_transaction_raw" {
  dataset_id  = google_bigquery_dataset.raw.dataset_id
  table_id    = "ieee_train_transaction_raw"
  description = "Full IEEE-CIS train_transaction.csv dump, WRITE_TRUNCATE-loaded by raw_transactions_bigquery (Dagster)."
  labels      = var.labels

  schema = file("${path.module}/../schemas/bq_schema_train_transaction.json")

  # Deliberately NOT ignore_changes on schema: the whole point of managing
  # this table here is that a schema edit shows up as a plan diff. If a
  # future change is BigQuery-incompatible in place (e.g. a type change,
  # not just adding a nullable field), apply surfaces that as an API
  # error -- not a silent recreate.
  deletion_protection = false

  lifecycle {
    prevent_destroy = false
  }
}

// Same reasoning as ieee_train_transaction_raw above, and the same
// deliberate absence of ignore_changes on schema.
resource "google_bigquery_table" "ieee_train_identity_raw" {
  dataset_id  = google_bigquery_dataset.raw.dataset_id
  table_id    = "ieee_train_identity_raw"
  description = "Full IEEE-CIS train_identity.csv dump, WRITE_TRUNCATE-loaded by raw_identity_bigquery (Dagster)."
  labels      = var.labels

  schema = file("${path.module}/../schemas/bq_schema_train_identity.json")

  deletion_protection = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_bigquery_table" "ieee_train_joined" {
  dataset_id  = google_bigquery_dataset.raw.dataset_id
  table_id    = "ieee_train_joined"
  description = "Joined transaction and identity tables, managed by Dagster."
  labels      = var.labels

  deletion_protection = false

  lifecycle {
    ignore_changes = [schema]
  }
}
