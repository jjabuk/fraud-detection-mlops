// Where the scoring image lives.
//
// Same region as the datasets and the job: an image pulled across regions is a cold start
// paid on every execution, for a container that exists to run once and exit.
resource "google_artifact_registry_repository" "images" {
  repository_id = "fraud-detection"
  location      = var.region
  format        = "DOCKER"
  description   = "Batch scoring / serving image. Tagged with the git SHA that built it."
  labels        = var.labels

  // Untagged images are build leftovers -- a re-push under the same tag orphans the
  // previous digest. Tagged ones are kept: an immutable tag whose image was swept is a
  // rollback that cannot happen.
  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" // 7 days
    }
  }

  depends_on = [google_project_service.artifactregistry]
}

locals {
  // The one place the image URI is assembled. The job and the Vertex registration both
  // read it from here, so they cannot disagree about which image is "the" image.
  image_uri = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/fraud-detection:${var.image_tag}"
}
