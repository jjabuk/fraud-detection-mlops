# Google Cloud

Which GCP services this pipeline uses, which it deliberately does not, and the
platform-specific details that are not visible from the Python.

Architecture and module boundaries: [architecture.md](architecture.md). Runbook and
authentication: [setup.md](setup.md). Everything below is declared in
[`iaac/`](../iaac/) unless stated otherwise.

## 1. What exists in the project

| Resource                                       | Why it is shaped this way                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Datasets `raw`, `features`, `inference`        | Three datasets rather than one, because IAM is granted per dataset. The prod profile holds `dataViewer` on `raw` and `dataEditor` only where it writes, so a bug in the scoring path cannot truncate the source tables.                                                                                                                                                                                 |
| Tables declared in OpenTofu, not by the loader | `raw` and `features` tables are `google_bigquery_table` resources whose schema is `file("../schemas/*.json")`. A column change is a `tofu plan` diff instead of a silent re-inference on the next load.                                                                                                                                                                                                 |
| Bucket `<project>-raw-data`                    | The staged Kaggle dump, in the same region as the datasets because `load_table_from_uri` rejects a cross-region pair. Versioned, so a bad re-download is recoverable, and carrying `prevent_destroy` because it is an input nothing here regenerates.                                                                                                                                                   |
| Bucket `<project>-models`                      | Model artifacts, their figures, and the promotion marker that is the deployment state the scoring job reads. Versioned as well, with a lifecycle rule expiring non-current versions after 30 days, since otherwise every training run leaves one behind forever.                                                                                                                                        |
| Artifact Registry repository                   | One repository for the scoring image, tagged with the git SHA.                                                                                                                                                                                                                                                                                                                                          |
| Service account with dev and prod IAM profiles | Project-level roles are kept to the ones that cannot be granted per resource (`bigquery.jobUser`, `bigquery.readSessionUser`, `aiplatform.user`, `run.invoker`, `artifactregistry.reader`, that last one read-only because CI pushes images and the workload only pulls them). Everything else is a dataset or bucket binding, and the prod profile downgrades `raw` from `dataEditor` to `dataViewer`. |
| Cloud Run Job `score_batch`                    | 2 vCPU, 8 GiB, `timeout 3600s`, `max_retries 1`, tmpfs on `/tmp`. The memory figure is measured rather than guessed: the peak sits comfortably below the limit, with the headroom there for Arrow's copy during the fetch.                                                                                                                                                                              |
| Enabled services, in code                      | `aiplatform`, `run`, `artifactregistry`, `cloudbuild` are `google_project_service` resources, so a fresh project reaches a working state from `tofu apply` alone.                                                                                                                                                                                                                                       |

`image_tag` has no default. An apply that forgets it fails instead of quietly rolling the job
back onto whatever a mutable tag points at.

Both buckets set `public_access_prevention = "enforced"` on top of uniform bucket-level
access, which are not the same guarantee: UBLA decides how access is granted, while
enforcement decides whether the bucket can be made public at all. With UBLA alone, one
`allUsers` binding still publishes everything and nothing in the config objects. On the models
bucket that matters more than usual, since it holds the pickle the scoring job loads and runs.

Local runs authenticate by service-account impersonation and CI by Workload Identity
Federation, so no service-account key exists in either place. The one long-lived credential
this project could have had is the one it does not have.

## 2. BigQuery details that are not visible from the Python

**Loads go through a widened staging table.** The pinned schema types several columns as
`INTEGER`, and BigQuery's CSV loader rejects the file outright if a single value disagrees.
[`_load_via_staging`](../src/fraud_detection/orchestration/assets/ingestion.py) loads into
`<table>_staging` with those columns relaxed to `FLOAT64`, asserts that every value is in fact
integral, then `CAST`s into the pinned types with the query's destination set to the real
table. The staging copy is dropped in a `finally`, because it is the same volume again on the
largest table in the project and is regenerable by definition.

The point of the detour is the failure mode it converts. Without it, a load either fails with
a message about a row number, or succeeds with autodetected types that differ from what
Terraform declared. With it, a non-integral value fails a named assertion that says which
column it was.

**Re-running the load is free.** The source object's etag is written into the destination
table's labels, and `_table_is_current` compares row count, etag and the live types against the
pinned schema before deciding to load. A pipeline re-run over an unchanged file does no work
and pays for no bytes.

**Frames never pass through the orchestrator.** Every transformation asset issues one
`CREATE OR REPLACE TABLE … AS SELECT` and returns a table name. The two exceptions, the audits
and training, are named in [code-structure.md](code-structure.md) with the reason.

**No partitioning or clustering, and that is a decision.** The dataset is a fixed 590,540-row
file, the pipeline scans it whole on every materialization, and the queries have no selective
predicate to prune on. Partitioning by day of `TransactionDT` would produce partitions of a few
thousand rows and add metadata overhead against no saved bytes. This is the first thing to
change if the pipeline ever ingested a daily feed: partition `raw` on ingestion date, cluster
`features` on the entity columns the windows partition by.

## 3. Vertex AI: what is used and what is not

| Used           | For what                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Experiments    | Every run including the BQML baseline, through one `ExperimentTracker` resource, so two models are never compared on two metric implementations. |
| Model Registry | The promoted artifact, registered against the same image digest the Cloud Run Job runs.                                                          |

| Not used                             | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vertex Pipelines                     | It would be a second orchestrator. Dagster already owns the asset graph, the checks and the lineage, and running both means deciding twice where a step belongs.                                                                                                                                                                                                                                                                                                                                                                                          |
| Managed Datasets                     | It wraps a table with metadata and a split definition. Both already exist here as their own objects: `features.model_input` is the table, and the admitted column set is [`references/feature-contract.json`](../references/feature-contract.json), fingerprinted and stamped onto the model at fit time. A managed dataset would add a second description of the same data with nothing downstream reading it.                                                                                                                                           |
| Feature Store                        | It answers "what is this entity's state right now", which is the online question. Scoring is batch, and `raw.scoring_history` is the same entity-keyed state materialized once instead of looked up per request. Adopting it would be paying for the online path without building it.                                                                                                                                                                                                                                                                     |
| Model Monitoring                     | It watches an endpoint's traffic, and there is no endpoint. `inference.prediction_logs` already records what a monitor would read; what is missing is the schedule, and [adversarial-drift.md](adversarial-drift.md) argues that a PSI cron would be the wrong monitor anyway.                                                                                                                                                                                                                                                                            |
| Endpoints and online prediction      | Out of scope for the reason in [architecture.md](architecture.md): the hard part is the point lookup against entity state, not the HTTP surface.                                                                                                                                                                                                                                                                                                                                                                                                          |
| Vertex Training                      | The fit runs in the Dagster process. Moving it to a training job buys managed compute for a model that trains on one machine, and costs a container round-trip per experiment.                                                                                                                                                                                                                                                                                                                                                                            |
| AutoML Tabular                       | It replaces the fit, which is not where the risk in this problem sits. The velocity features have to be computed under a `RANGE … 1 PRECEDING` frame before any trainer sees them, so the leakage-critical work happens upstream of AutoML either way, and what remains is a model that cannot be seeded, refitted five times for a noise band, or stamped with a contract fingerprint. The measurements in [MEASUREMENTS.md](MEASUREMENTS.md) took roughly forty refits; on node-hours that is a different kind of decision. See below on how it splits. |
| Model Evaluation                     | The evaluation the gate reads is per segment: PR-AUC by `ProductCD`, and by whether the client was seen in training, each against its own base rate. Vertex's evaluation resource reports the pooled figure, which is the number [MEASUREMENTS.md](MEASUREMENTS.md) shows to be misleading on this model — segment `W` carries 77% of scored rows at roughly a quarter of segment `R`'s PR-AUC. Each run writes its own `metrics.json` beside the artifact instead.                                                                                       |
| Explainable AI, feature attributions | Vertex serves attributions from a deployed model or a batch explanation job. SHAP runs inside the training pipeline instead, global and per-decision, so the explanations exist before the gate decides anything rather than after something is deployed.                                                                                                                                                                                                                                                                                                 |
| Workbench                            | The notebooks in [`eda/`](../eda/README.md) run locally against the same `fraud_detection.evaluation` and `fraud_detection.training` modules the pipeline imports, which [`tests/test_layering.py`](../tests/test_layering.py) enforces. A managed notebook would host that same code at an hourly rate.                                                                                                                                                                                                                                                  |
| BQML as the production model         | It is here as the baseline the gate compares against, which is the job it is good at.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

**On AutoML and the split.** AutoML Tabular does support a chronological split, so time-based
validation on its own is not what rules it out. What the chronological split cannot express is
the embargo: it assigns contiguous shares of the Time column to train, validation and test,
with no way to leave a band unassigned. The 10% of the time axis this pipeline deliberately
does not assign, because a chargeback arrives months after the transaction
([MEASUREMENTS.md](MEASUREMENTS.md)), would have to come back as a manual split column — which
is the assignment `split_assignment` already computes. AutoML would replace the fit, not the
split and not the feature engineering upstream of it.

The registry entry is deliberately non-trivial. `Model.upload` requires a serving container and
LightGBM has no prebuilt one, so registering against the prebuilt sklearn image would create an
entry that cannot serve. The batch image therefore also implements Vertex's custom-container
contract, `/health` and `/predict`, and the registry entry points at that digest. One image,
two entrypoints, and the Cloud Run Job overrides the command.

## 4. What a reviewer should know is missing

- **No cost or runtime figures are published here.** They would be a property of one run on one
  project, and this project does not run on a cadence.
- **No VPC Service Controls, no CMEK.** Both are the right answer for real payment data and
  neither is exercised on a public competition dataset; claiming them without running them
  would be worse than the gap.
- **Nothing rolls the Cloud Run Job forward automatically.** The push workflow is dispatched
  by hand and needs the WIF secrets configured;
  moving the job onto the new tag stays a `tofu apply` with the SHA.
- **One project, one region.** Serving is stateless and scales to zero, so a regional
  deployment matches the failure model this system actually has.
