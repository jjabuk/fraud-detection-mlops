resource "google_bigquery_dataset" "inference" {
  dataset_id  = "inference"
  location    = var.bq_location
  description = "Inference request/prediction logs for monitoring and drift analysis"
  labels      = var.labels

  delete_contents_on_destroy = false

  lifecycle {
    prevent_destroy = true
  }
}