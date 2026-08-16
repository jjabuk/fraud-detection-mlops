# Code structure

**Which code a notebook may import, and which belongs to the orchestrator.**

Three layers, one direction of dependency:

```text
src/fraud_detection/
    schema.py              ┐
    evaluation/            │  pure — pandas, numpy, sklearn, lightgbm
    training/              │  no Dagster, no cloud SDK
    feature_contract/      ┘

    core/promotion.py      ┘  which artifact carries the production alias

    assets/                ┐  orchestration — Dagster, google.cloud
    resources.py           │  imports the pure layer; nothing imports it back
    definitions/           ┘

    serving/               the container's HTTP surface — FastAPI, imports the pure layer
```

> **The rule:** `assets/` may import from anywhere. Nothing may import from `assets/`.

It is enforced by [`tests/test_layering.py`](../tests/test_layering.py), which reads the import graph rather than trusting this document.

## The second boundary: two code locations

The layering above says what a _notebook_ may import. It says nothing about what may be
deployed apart from what, and that is a different question with a different answer.

`definitions/` holds three Dagster code locations, loaded as separate processes from one `workspace.yaml`. The seam checked below is the one between the first two; `inference` consumes `model_input` and the promotion marker and builds nothing either of them needs:

```text
    feature_platform      raw CSVs -> features.model_input
                          + the audits -> references/feature-contract.json

                                    |  the seam: two artefacts, nothing else
                                    v

    model_factory         those two -> splits -> baseline, LightGBM
                          -> explanations -> validation gate -> promotion
```

**Exactly two things cross**, and the model factory declares both as external assets in
[`definitions/model_factory.py`](../src/fraud_detection/definitions/model_factory.py):

| Crossing           | How it is consumed                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `model_input`      | depended on **by key**, never by value — the table name comes from `schema.qualified(...)`, a constant both sides share |
| `feature_contract` | read from `references/feature-contract.json`, with the fingerprint pinned onto the trained model                        |

The by-key rule is the part that makes the boundary real. Passing a table name down the graph as a return value is fine inside one location and impossible across two, so an asset that still takes `model_input: str` is an asset that has not actually crossed. That is why `split_assignment`, `bqml_baseline`, `lightgbm_model` and `model_explanations` take `deps=[AssetKey("model_input")]` instead.

[`tests/test_code_locations.py`](../tests/test_code_locations.py) checks the seam rather
than describing it: that the crossing is exactly those two keys, that the platform really
builds what the factory declares, that neither location's module imports the other, and
that no asset is built twice.

## Why this split matters

- **One definition of each transformation.** A pure function can be called by both paths; if it lived inside a Dagster asset, scoring would have to reimplement it, and two implementations of one feature drift apart on a timescale of weeks.
- **Tests run without credentials.** CI runs on every PR because the logic under test constructs no cloud clients.
- **Portability is real.** The pure layer is the part that could move; if the modelling recipe were welded to Dagster, the portability claim would be false.

## Pure does not mean "no I/O"

It means: takes its inputs as arguments, does not construct clients, does not know what an
orchestrator is.

`training/data.load_split(client, project, split, ...)` takes a BigQuery client as a
parameter. That is correct and is the pattern to follow — dependency injection, not a
repository abstraction. Resist building an adapter layer: a function parameter already does
the job, and an interface with one implementation is an interface built for an imagined
requirement.

## What lives where

| Module                                                | Holds                                                  | Module | Purpose | Pure? |
| ----------------------------------------------------- | ------------------------------------------------------ | ------ | ------- | ----- |
| `schema.py`                                           | Column sets and table names shared across the boundary | yes    |
| `evaluation/`                                         | The five feature audits                                | yes    |
| `training/`                                           | The LightGBM recipe, metrics, and SHAP                 | yes    |
| `training/data.py`                                    | Split loading and feature preparation                  | yes    |
| `training/calibration.py`, `threshold.py`, `plots.py` | The decision layer                                     | yes    |
| `feature_contract/`                                   | The admitted-column artefact and its adapters          | yes    |
| `core/promotion.py`                                   | Parsing and validating the gate's promotion marker     | yes    |
| `assets/`                                             | Dagster assets — load, store, log                      | no     |
| `resources.py`                                        | BigQuery, Vertex, GCS handles                          | no     |
| `definitions/`                                        | The three code locations Dagster loads                 | no     |
| `serving/`                                            | `/predict` + `/health`, the registry's container contract | no  |

## Using it from a notebook

```python
from fraud_detection.evaluation.entity_purity import Anchor, EntityKey, compare
from fraud_detection.training.model import train_lightgbm
from fraud_detection.training.data import prepare_features
```

No credentials, no orchestrator. The notebooks in [`eda/notebooks/`](../eda/notebooks/) do exactly this against local CSVs.

## Where compute actually runs

The layering says what imports what. It says nothing about where work happens, and those
are different questions:

| Stage                                        | Runs                                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| Ingestion, join, feature engineering, splits | BigQuery — one statement, result never leaves the warehouse |
| Feature audits (`assets/feature_audit.py`)   | **Locally**, in the Dagster process                         |
| Training (`assets/training.py`)              | **Locally**, in the Dagster process                         |
| Scoring windows and model input              | BigQuery, over `raw.scoring_history` (train ∪ test)         |
| Batch scoring (`assets/inference.py`)        | **Cloud Run Job**, from the image this repository builds    |

The two local exceptions are deliberate and bounded — LightGBM cannot be expressed in SQL,
and each pulls a few hundred MB. They are also the reason "where does Dagster run" is an
open question rather than a deferred one.
