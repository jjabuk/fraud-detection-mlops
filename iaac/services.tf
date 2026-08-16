// APIs are enabled here only when a resource or a pipeline stage actually
// needs one -- the same rule the IAM profiles in service_account.tf follow,
// so the surface stays honest rather than aspirational.
//

// Needed by ExperimentTracker: Vertex AI Experiments is the tracker of record for every training run.
resource "google_project_service" "aiplatform" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

// Needed by google_cloud_run_v2_job.score_batch: the batch scoring job.
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

// Needed by google_artifact_registry_repository.images: CI pushes the scoring image
// here, and both the job and the Vertex registry entry pull it from here.
resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

// Builds the scoring image without a working local Docker daemon. The designed path is
// still GitHub Actions on merge (.github/workflows/image.yml) -- this is the manual
// equivalent, `gcloud builds submit`, and it is declared here rather than enabled by hand
// so the enabled-API set stays something `tofu plan` can account for.
resource "google_project_service" "cloudbuild" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}
