
resource "google_bigquery_dataset" "features" {
  dataset_id  = "features"
  location    = var.bq_location
  description = "Feature store for point-in-time engineered fraud features"
  labels      = var.labels

  delete_contents_on_destroy = false

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_bigquery_table" "transaction_features" {
  dataset_id  = google_bigquery_dataset.features.dataset_id
  table_id    = "transaction_features"
  description = "Engineered features, managed by Dagster."
  labels      = var.labels

  deletion_protection = false

  lifecycle {
    ignore_changes = [schema]
  }
}

resource "google_bigquery_table" "model_input" {
  dataset_id  = google_bigquery_dataset.features.dataset_id
  table_id    = "model_input"
  description = "Final model input table, managed by Dagster."
  labels      = var.labels

  deletion_protection = false

  lifecycle {
    ignore_changes = [schema]
  }
}

resource "google_bigquery_table" "split_assignment" {
  dataset_id  = google_bigquery_dataset.features.dataset_id
  table_id    = "split_assignment"
  description = "Train/test split assignments, managed by Dagster."
  labels      = var.labels

  deletion_protection = false

  lifecycle {
    ignore_changes = [schema]
  }
}
