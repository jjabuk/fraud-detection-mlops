// Batch scoring as a Cloud Run **Job**, not a Service.
//
// A Job is a container that runs to completion and exits; a Service is a container that
// waits for HTTP. Scoring a fixed set of transactions is the first shape, and modelling it
// as the second would mean paying for an idle listener and inventing a caller for it.
// The endpoint the image also carries exists for the Vertex registry's container contract
// (see src/fraud_detection/serving/app.py), not because anything calls it.
resource "google_cloud_run_v2_job" "score_batch" {
  name     = "fraud-score-batch"
  location = var.region
  labels   = var.labels

  deletion_protection = false

  template {
    template {
      service_account = google_service_account.mlops.email

      // Overrides the image's default command, which is the serving process. One image,
      // two entrypoints -- see the Dockerfile for why that is one image and not two.
      containers {
        image = local.image_uri
        // Through a shell because of the mkdir: Cloud Run gives the container a fresh
        // tmpfs at /tmp, so the DAGSTER_HOME baked into the image is gone by the time
        // the process starts, and Dagster refuses to start without it. Found by running
        // the image, not by reading it.
        command = ["sh", "-c"]
        args = [
          "mkdir -p \"$DAGSTER_HOME\" && exec dagster job execute -m fraud_detection.orchestration.definitions.inference -j inference_job",
        ]

        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "SERVING_IMAGE_URI"
          value = local.image_uri
        }

        resources {
          limits = {
            // The scoring frame is ~500k rows across the admitted columns, loaded into
            // memory once. Measured peak sits under 6 GiB; 8 leaves room for the Arrow
            // copy during the BigQuery read without leaving room for a leak to hide in.
            cpu    = "2"
            memory = "8Gi"
          }
        }
      }

      // The BigQuery jobs are the slow part and they are synchronous from the container's
      // point of view. Cloud Run's 10-minute default would kill the run mid-query.
      timeout     = "3600s"
      max_retries = 1
    }
  }

  depends_on = [google_project_service.run]
}
