# Infrastructure-as-Code Backend Setup

Here's how to spin up your remote OpenTofu (or Terraform) state in Google Cloud Storage so you can apply the IaC configuration for this project.

To make this fully copy-pasteable, let's export a few variables. Run this once in your terminal:

```bash
export PROJECT_ID=<your-project-id>
export REGION=europe-central2
export STATE_BUCKET="${PROJECT_ID}-tfstate"
```

## Why we create the state bucket by hand

You can't use OpenTofu to create the bucket that stores OpenTofu's own state. It's a classic chicken-and-egg problem, so we just bootstrap the backend bucket using `gcloud` before running any IaC commands.

## Prerequisites

- `gcloud` CLI installed
- `tofu` (OpenTofu) installed
- Permissions to create buckets and tweak IAM in your GCP project
- You're logged in (`gcloud auth login`)

## Let's get the backend running

1. Set your active GCP project:

```bash
gcloud config set project "${PROJECT_ID}"
```

1. Create the state bucket:

```bash
gcloud storage buckets create "gs://${STATE_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  --public-access-prevention
```

1. Enable object versioning (this is critical in case you ever need to recover a borked state file):

```bash
gcloud storage buckets update "gs://${STATE_BUCKET}" --versioning
```

1. Verify it's all set up correctly:

```bash
gcloud storage buckets describe "gs://${STATE_BUCKET}" \
  --project="${PROJECT_ID}" | rg -i 'versioning|name:|location:'
```

You should see something like:

- `name: <your-project-id>-tfstate`
- `location: EUROPE-CENTRAL2`
- `versioning_enabled: true`

## What's already in the box

I've already wired up a few things in this directory:

1. **GCS backend** in [versions.tf](versions.tf) pointing to the bucket and `iaac` prefix.
2. **Default project_id** in [variables.tf](variables.tf).
3. **Environment-aware IAM profiles** for the service account in [service_account.tf](service_account.tf):
   - `dev`: `roles/bigquery.jobUser` on the project, plus editor rights on `raw`, `features`, and `prediction_logs` datasets.
   - `prod`: Same project role, but stricter dataset roles (viewer on `raw`, editor on the rest).
     _(Note: `roles/run.invoker` isn't here yet because we don't have a Cloud Run resource in `iaac/` to invoke. We add permissions only when the resource exists.)_
4. **Example variable sets** ready to go in [environments/dev.tfvars](environments/dev.tfvars) and [environments/prod.tfvars](environments/prod.tfvars).

## Applying changes

If you ever change the backend configuration, re-initialize:

```bash
tofu init -reconfigure
```

When you're ready to see what Tofu will do:

```bash
tofu plan -var-file=environments/dev.tfvars
# or
tofu plan -var-file=environments/prod.tfvars
```

## A quick note about ADC quotas

If `gcloud` yells at you about a mismatched quota project for Application Default Credentials, just run this to silence it and avoid unexpected API quota headaches:

```bash
gcloud auth application-default set-quota-project "${PROJECT_ID}"
```
