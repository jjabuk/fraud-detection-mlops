// Staging bucket for the static Kaggle raw CSV dump. Ingestion's
// load_table_from_uri path reads directly from here -- location must 
// match bq_location (both europe-central2), since load_table_from_uri rejects cross-region
// bucket/dataset pairs.

resource "google_storage_bucket" "raw_data" {
  name                        = "${var.project_id}-raw-data"
  location                    = var.region
  uniform_bucket_level_access = true
  labels                      = var.labels

  # Uniform access controls *who* is granted access; this controls whether the bucket can
  # be made public at all. They are not the same guarantee: with UBLA alone, one
  # `allUsers` IAM binding still publishes the whole bucket, and nothing in the config
  # would have objected. "enforced" makes that binding fail instead -- the difference
  # between a policy and a convention. The tfstate bucket is bootstrapped by hand with
  # the same flag (see versions.tf).
  public_access_prevention = "enforced"

  # A bad re-download/overwrite of the staged file is recoverable from the
  # previous generation instead of just gone -- same reasoning as the
  # tfstate bucket's own versioning (see versions.tf).
  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

// Trained model artifacts. Separate from raw_data on purpose: the raw dump is
// an immutable input that must never be lost, while this bucket holds
// regenerable outputs written on every training run. Versioned all the same,
// because a model that scored well and got overwritten is not regenerable in
// practice -- the data and the seed may reproduce it, the afternoon spent
// deciding it was the one will not.
resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-models"
  location                    = var.region
  uniform_bucket_level_access = true
  labels                      = var.labels

  # Same reasoning as raw_data, and it matters more here: this bucket holds the promotion
  # marker and the pickled model the scoring job loads and executes. A publicly writable
  # path to either is remote code execution with extra steps.
  public_access_prevention = "enforced"

  versioning {
    enabled = true
  }

  # Superseded versions stop being interesting once a newer model has been
  # through the validation gate, and without a sweep every training run leaves
  # one behind forever.
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
    }
    action {
      type = "Delete"
    }
  }
}
