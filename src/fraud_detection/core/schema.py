"""Names the pipeline and the analysis layer both have to agree on.

These are facts about the dataset and about what feature engineering produces. They live
outside `assets/` because a notebook needs them without needing an orchestrator, and
because a constant owned by the layer that happens to write the SQL is a constant the
analysis layer has to reach backwards for.

Direction of dependency: `assets/` imports from here. Nothing here imports from `assets/`.
"""

from __future__ import annotations

LABEL_COLUMN = "isFraud"
AMOUNT_COLUMN = "TransactionAmt"
TIME_COLUMN = "TransactionDT"
ID_COLUMN = "TransactionID"

CARD_ENTITY_COLUMN = "card1"
DEVICE_ENTITY_COLUMN = "DeviceInfo"
CLIENT_ENTITY_COLUMN = "client_uid"

# What the reconstructed client is made of: `card1 + addr1 + (day - D1)`.
#
# One definition, two readers. `features.CLIENT_UID_EXPRESSION` builds the SQL from these,
# and the feature contract records them as its `entity` block. Naming them here keeps
# by hand in both places, so the contract could have gone on describing `card1 + addr1`
# after the SQL had moved on -- and the contract is the artefact somebody would trust to
# know what an entity is.
CLIENT_ENTITY_COMPONENTS = (CARD_ENTITY_COLUMN, "addr1")
CLIENT_ENTITY_ANCHOR = "D1"
"""Days since the client's first transaction, so `day - D1` is their account-start day."""

# Tables, by dataset.
RAW_DATASET = "raw"
FEATURES_DATASET = "features"
INFERENCE_DATASET = "inference"

RAW_TRANSACTION_TABLE = "ieee_train_transaction_raw"
RAW_IDENTITY_TABLE = "ieee_train_identity_raw"
JOINED_TABLE = "ieee_train_joined"
FEATURE_TABLE = "transaction_features"
MODEL_INPUT_TABLE = "model_input"
SPLIT_TABLE = "split_assignment"

# ---- batch scoring -------------------------------------------------------------
#
# The Kaggle test period, and the union that lets its windows see the training
# period. SCORING_HISTORY_TABLE is train ∪ test: the velocity aggregates are
# computed over it, and only the test rows are scored. Computing them over the
# test tables alone is training-serving skew -- a card with 300 transactions in
# the training period would arrive at its first test transaction with
# `card_txn_count_prior = 0`, a value that never occurs in training for that
# card. See docs/point-in-time.md and docs/MEASUREMENTS.md.
RAW_TEST_TRANSACTION_TABLE = "test_transaction"
RAW_TEST_IDENTITY_TABLE = "test_identity"
TEST_JOINED_TABLE = "test_joined"
SCORING_HISTORY_TABLE = "scoring_history"
SCORING_FEATURE_TABLE = "scoring_features"
TEST_MODEL_INPUT_TABLE = "test_model_input"
PREDICTION_LOG_TABLE = "prediction_logs"

# Which period a scoring_history row came from. Not a feature: it is how the
# scored set is selected back out of the union, and it is `EXCLUDED_COLUMNS`
# material for the same reason `split` is.
ORIGIN_COLUMN = "origin"
ORIGIN_TRAIN = "train"
ORIGIN_TEST = "test"


def qualified(project: str, dataset: str, table: str) -> str:
    """The fully-qualified name of a table.

    This exists so that an asset can name a table it depends on **without receiving the
    name as a value from the asset that built it**. Passing the string down the graph is
    fine inside one code location and impossible across two: the model factory cannot
    import the feature platform's asset function, only depend on its key. Deriving the
    name from a shared constant is what turns that edge into an ordering dependency rather
    than a data-passing one.
    """
    return f"{project}.{dataset}.{table}"

# The engineered (i.e. not passed through from raw) columns feature engineering produces.
# One source of truth: the SQL that builds them, the assembly that names them, and the
# contract that classifies them as retrieved-at-serving-time all read this list.
FEATURE_COLUMNS = [
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_txn_amt_avg_24h",
    "card_txn_amt_sum_24h",
    "card_amt_deviation_24h",
    "seconds_since_prev_txn_card",
    "device_txn_count_24h",
    "client_txn_count_prior",
    "client_txn_count_24h",
    "client_amt_avg_prior",
    "client_amt_deviation_prior",
    "seconds_since_prev_txn_client",
]

# The uid aggregates: the published solutions' aggregation family (ATTRIBUTION.md),
# recomputed under a
# point-in-time window. Built by fraud_detection.features rather than listed by hand,
# because the D-normalisations they aggregate are declared in config/feature-admission.toml
# and a hand-kept copy here would drift from it silently.
#
# Appended to FEATURE_COLUMNS at import so that everything reading "the engineered columns"
# -- the model input SQL, the contract's retrieved/request split -- sees one list.


def uid_aggregate_feature_columns() -> list[str]:
    from fraud_detection.core.feature_contract.admission import load_admission_rules
    from fraud_detection.feature_engineering.features import uid_aggregate_columns

    rules = load_admission_rules()
    wanted = set(rules.uid_std_of_derived)
    return uid_aggregate_columns(
        [d for d in rules.derivations if d.name in wanted],
        rules.uid_c_columns,
        rules.uid_m_columns,
    )

# Never features.
#
# TransactionID identifies a row. `split` and `card_seen_in_train` describe the
# experiment, not the transaction.
#
# TransactionDT is excluded on purpose and the reason is specific to a time-based split:
# the test window lies entirely after the training window, so every test value is outside
# the range the trees ever saw. A split on it can only send the whole test set down one
# branch. Time enters the model through the velocity features, which are relative to the
# transaction and therefore transfer.
#
# client_uid is an entity key, never a feature. It has 217,735 levels across 590,540 rows;
# as a categorical it would let the model memorise clients, and every client it memorised
# would be one it can never meet again.
EXCLUDED_COLUMNS = frozenset(
    {
        ID_COLUMN,
        TIME_COLUMN,
        LABEL_COLUMN,
        "split",
        "card_seen_in_train",
        "client_uid",
        ORIGIN_COLUMN,
    }
)
