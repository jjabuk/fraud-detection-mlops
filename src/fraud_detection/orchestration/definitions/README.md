# Dagster Definitions Architecture

This directory contains the two primary code locations (Dagster Definitions) that orchestrate the ML pipeline. The split is deliberate: the **Feature Platform** owns what a feature is and whether it is fit to use, and the **Model Factory** owns what to do with the features that passed.

## Code Location 1: Feature Platform (`feature_platform.py`)

Raw Kaggle CSVs go in, two artifacts come out:

- `features.model_input` — the joined table, one row per transaction, containing every candidate column.
- `references/feature-contract.json` — decides which of those columns a model is allowed to use, and records the verdicts that rejected each of the rest.

Nothing here knows what a model is.

```mermaid
graph TD
    subgraph raw_ingestion [Raw Ingestion]
        kaggle_source[kaggle_source]
        schema_generation[schema_generation]
        ingestion[ingestion]
        join[join]
        
        kaggle_source --> schema_generation --> ingestion --> join
    end

    subgraph feature_store [Feature Store]
        transaction_features[transaction_features]
        model_input[model_input]
        
        join --> transaction_features --> model_input
    end

    subgraph feature_validation [Feature Validation]
        time_consistency[time_consistency]
        distribution_shift[distribution_shift]
        redundancy[redundancy]
        feature_contract[feature_contract]
        
        time_consistency --> feature_contract
        distribution_shift --> feature_contract
        redundancy --> feature_contract
    end

    model_input -.-> time_consistency
    model_input -.-> distribution_shift
    model_input -.-> redundancy
    
    classDef cleansing fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;
    classDef engineering fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef selection fill:#fff3e0,stroke:#fb8c00,stroke-width:2px;
    
    class kaggle_source,schema_generation,ingestion,join cleansing;
    class transaction_features,model_input engineering;
    class time_consistency,distribution_shift,redundancy,feature_contract selection;
```

The split between the Feature Store and Feature Validation is worth defending: engineering answers "what could this column be", while the audit answers "should the model be allowed to see it". Collapsing them is how a feature ends up admitted because the person who built it also approved it.

## Code Location 2: Model Factory (`model_factory.py`)

Consumes exactly two artifacts from the feature platform and produces a validated, promotable model.

- `features.model_input` — depended on **by key**, never by value. The assets name the table, because a code location cannot receive another location's return value, only the fact that it materialized.
- `references/feature-contract.json` — read from disk. It decides which columns training is allowed to see. `model_features_admitted_check` asserts after the fit that the model and the contract still agree.

```mermaid
graph TD
    subgraph external_assets [External Assets]
        model_input[(model_input)]
        feature_contract[feature_contract.json]
    end

    subgraph dataset_preparation [Dataset Preparation]
        split_assignment[split_assignment]
    end
    
    model_input --> split_assignment
    feature_contract --> split_assignment

    subgraph model_training [Model Training]
        bqml_baseline[bqml_baseline]
        lightgbm_model[lightgbm_model]
    end
    
    split_assignment --> bqml_baseline
    split_assignment --> lightgbm_model

    subgraph model_registry [Model Registry]
        model_explanations[model_explanations]
        validation_gate[validation_gate]
    end
    
    bqml_baseline --> validation_gate
    lightgbm_model --> model_explanations
    model_explanations --> validation_gate
    
    classDef external fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,stroke-dasharray: 5 5;
    classDef prep fill:#fff8e1,stroke:#fbc02d,stroke-width:2px;
    classDef training fill:#e8eaf6,stroke:#3949ab,stroke-width:2px;
    classDef registry fill:#ffebee,stroke:#e53935,stroke-width:2px;
    
    class model_input,feature_contract external;
    class split_assignment prep;
    class bqml_baseline,lightgbm_model training;
    class model_explanations,validation_gate registry;
```

The seam is explicitly declared in `model_factory.py` via `EXTERNAL_ASSETS`. This is the complete list of what the Model Factory does not build for itself. If that list grows, the boundary is moving, and that should be a conscious architectural decision rather than an accident.
