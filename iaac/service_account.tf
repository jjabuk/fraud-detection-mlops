resource "google_service_account" "mlops" {
  account_id   = var.service_account_id
  display_name = "Fraud MLOps Service Account"
  description  = "Workload identity for fraud data pipelines and model serving"
}

locals {
  # roles/run.invoker: google_cloud_run_v2_job.score_batch exists, this
  # account is what executes it, and executing a job is run.jobs.run --
  # granted by run.invoker. The old comment was true when it was written and
  # became false the moment the job resource landed.
  iam_profiles = {
    dev = {
      project_roles = [
        "roles/bigquery.jobUser",
        // Writes experiment runs and their metrics to Vertex AI Experiments,
        // and uploads the promoted artifact to the Model Registry.
        "roles/aiplatform.user",
        // Reads training splits through the BigQuery Storage API, which needs
        // bigquery.readsessions.create.
        "roles/bigquery.readSessionUser",
        // Executes the batch scoring job.
        "roles/run.invoker",
        // Pulls the scoring image at job start. Read-only on purpose: CI
        // pushes images, the workload only ever reads them.
        "roles/artifactregistry.reader",
      ]
      dataset_roles = {
        raw       = "roles/bigquery.dataEditor"
        features  = "roles/bigquery.dataEditor"
        inference = "roles/bigquery.dataEditor"
      }
    }
    prod = {
      project_roles = [
        "roles/bigquery.jobUser",
        "roles/aiplatform.user",
        "roles/bigquery.readSessionUser",
        "roles/run.invoker",
        "roles/artifactregistry.reader",
      ]
      dataset_roles = {
        raw       = "roles/bigquery.dataViewer"
        features  = "roles/bigquery.dataEditor"
        inference = "roles/bigquery.dataEditor"
      }
    }
  }

  selected_iam_profile = local.iam_profiles[var.environment]

  dataset_ids = {
    raw       = google_bigquery_dataset.raw.dataset_id
    features  = google_bigquery_dataset.features.dataset_id
    inference = google_bigquery_dataset.inference.dataset_id
  }
}

resource "google_project_iam_member" "mlops_project_roles" {
  for_each = toset(local.selected_iam_profile.project_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.mlops.email}"
}

resource "google_bigquery_dataset_iam_member" "mlops_dataset_roles" {
  for_each = local.selected_iam_profile.dataset_roles

  dataset_id = local.dataset_ids[each.key]
  role       = each.value
  member     = "serviceAccount:${google_service_account.mlops.email}"
}

// Read & Write: raw_transactions_bigquery's load_table_from_uri path
// reads the staged raw CSV, and raw_transaction_kaggle_to_gcs stages/overwrites 
// it via upload_from_filename().
// objectAdmin grants full control (create, read, update, delete).
resource "google_storage_bucket_iam_member" "mlops_raw_data_admin" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlops.email}"
}

// Training writes model artifacts here; Vertex AI Model Registry reads them
// back out when a model is registered.
resource "google_storage_bucket_iam_member" "mlops_models_admin" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlops.email}"
}

output "mlops_service_account_email" {
  description = "Email address of the workload service account"
  value       = google_service_account.mlops.email
}

output "iam_environment" {
  description = "IAM profile currently selected by var.environment"
  value       = var.environment
}
