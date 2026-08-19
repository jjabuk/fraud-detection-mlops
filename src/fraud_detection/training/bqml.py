# Where the validation gate looks up what "above baseline" currently means.
from fraud_detection.config import get_training_params
from fraud_detection.schema import FEATURE_COLUMNS

_bqml_cfg = get_training_params("training.bqml")
BASELINE_MODEL = _bqml_cfg["baseline_model"]
BASELINE_METRICS_PATH = _bqml_cfg["baseline_metrics_path"]

# Deliberately narrow: the amount plus the seven engineered velocity features.
# This is the number "PR-AUC above baseline" refers to, so it has to answer
# one question -- did the feature engineering buy anything -- and not be a
# second model competing with LightGBM. A wide BQML model over all ~440
# columns is a later ablation row, not this.
BASELINE_FEATURE_COLUMNS = ["TransactionAmt", *FEATURE_COLUMNS]

# No auto_class_weights. At a 3.5% positive rate that option would improve
# the decision threshold, and the threshold is not what this measures --
# PR-AUC scores the ranking, which class weighting leaves essentially alone.
# Keeping the baseline plain keeps it a baseline.
#
# The TRANSFORM clause is the reason this is worth doing in BQML at all.
# Preprocessing declared inside TRANSFORM is stored *with the model* and
# replayed automatically at prediction time, so training-serving skew in the
# preprocessing step becomes structurally impossible rather than something a
# test has to catch. ML.PREDICT below passes raw columns and never restates
# the scaling -- if it had to, that restatement would be the bug.
#
# model_registry/vertex_ai_model_id register the trained model straight into
# Vertex AI Model Registry, which exercises the seam between the data layer
# and the registry before a model anyone cares about depends on it.
CREATE_MODEL_SQL = """
CREATE OR REPLACE MODEL `{model_id}`
TRANSFORM(
  {transform_columns},
  {label_column}
)
OPTIONS(
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['{label_column}'],
  model_registry = 'vertex_ai',
  vertex_ai_model_id = '{vertex_model_id}'
) AS
SELECT
  {feature_columns},
  m.{label_column}
FROM `{model_input_table}` AS m
JOIN `{split_table}` AS s
  ON m.TransactionID = s.TransactionID
WHERE s.split = 'train'
"""

VERTEX_BASELINE_MODEL_ID = _bqml_cfg["vertex_baseline_model_id"]

# predicted_<label>_probs is an array of (label, prob) structs; the cast makes
# the lookup independent of whether BQML hands the label back as INT64 or
# STRING.
PREDICT_SQL = """
SELECT
  {label_column} AS y_true,
  (
    SELECT p.prob
    FROM UNNEST(predicted_{label_column}_probs) AS p
    WHERE CAST(p.label AS STRING) = '1'
  ) AS y_score
FROM ML.PREDICT(
  MODEL `{model_id}`,
  (
    SELECT
      {feature_columns},
      m.{label_column}
    FROM `{model_input_table}` AS m
    JOIN `{split_table}` AS s
      ON m.TransactionID = s.TransactionID
    WHERE s.split = '{split}'
  )
)
"""


def build_feature_column_list(columns: list[str] = BASELINE_FEATURE_COLUMNS) -> str:
    return ",\n  ".join(f"m.{column}" for column in columns)


def build_transform_column_list(columns: list[str] = BASELINE_FEATURE_COLUMNS) -> str:
    """Standardises every feature inside the model.

    Logistic regression is scale-sensitive, and these features span
    transaction counts in the single digits and amounts in the thousands.
    Declaring it here rather than in the SELECT is the whole point: BQML
    stores the fitted scaler with the model and reapplies it on every
    prediction, so nothing downstream can forget to.
    """
    return ",\n  ".join(f"ML.STANDARD_SCALER({column}) OVER() AS {column}" for column in columns)
