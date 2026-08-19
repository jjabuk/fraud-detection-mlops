from fraud_detection.orchestration.assets.baseline import bqml_baseline
from fraud_detection.orchestration.assets.explainability import model_explanations
from fraud_detection.orchestration.assets.feature_audit import (
    audit_frame,
    feature_contract,
)
from fraud_detection.orchestration.assets.feature_contract_check import (
    feature_contract_freshness,
    feature_contract_integrity,
)
from fraud_detection.orchestration.assets.feature_engineering import transaction_features
from fraud_detection.orchestration.assets.gate import validation_gate
from fraud_detection.orchestration.assets.inference import (
    kaggle_submission,
    kaggle_test_joined,
    kaggle_test_model_input,
)
from fraud_detection.orchestration.assets.ingestion import (
    raw_identity_bigquery,
    raw_transactions_bigquery,
)
from fraud_detection.orchestration.assets.join import joined_transactions_identity
from fraud_detection.orchestration.assets.kaggle_source import (
    raw_identity_kaggle_to_gcs,
    raw_transaction_kaggle_to_gcs,
)
from fraud_detection.orchestration.assets.model_input import model_input
from fraud_detection.orchestration.assets.serving import best_model
from fraud_detection.orchestration.assets.splits import split_assignment
from fraud_detection.orchestration.assets.training import lightgbm_model

__all__ = [
    # Feature validation
    "audit_frame",
    # Serving
    "best_model",
    # Model training
    "bqml_baseline",
    "feature_contract",
    # Feature contract checks
    "feature_contract_freshness",
    "feature_contract_integrity",
    # Data joining
    "joined_transactions_identity",
    "kaggle_submission",
    # Inference
    "kaggle_test_joined",
    "kaggle_test_model_input",
    "lightgbm_model",
    # Model evaluation
    "model_explanations",
    "model_input",
    "raw_identity_bigquery",
    # Kaggle data sources
    "raw_identity_kaggle_to_gcs",
    "raw_transaction_kaggle_to_gcs",
    # Data ingestion
    "raw_transactions_bigquery",
    # Splits
    "split_assignment",
    # Feature audits
    # Feature engineering
    "transaction_features",
    "validation_gate",
]
