"""Loading a split of the model input out of BigQuery and into memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from fraud_detection.features.derivations import apply_derivations
from fraud_detection.schema import AMOUNT_COLUMN, EXCLUDED_COLUMNS, LABEL_COLUMN

# Read through the BigQuery Storage API (Arrow, columnar) rather than the REST
# download, which would hand back 590k rows across 442 columns as JSON. That
# needs bigquery.readsessions.create, granted to the workload service account
# in iaac/service_account.tf -- one IAM role instead of an export-to-Parquet
# detour through GCS.
SPLIT_QUERY = """
SELECT
  m.*,
  s.card_seen_in_train
FROM `{model_input_table}` AS m
JOIN `{split_table}` AS s
  ON m.TransactionID = s.TransactionID
WHERE s.split = '{split}'
"""


@dataclass(frozen=True)
class SplitFrame:
    features: pl.DataFrame
    labels: pl.Series
    amounts: pl.Series
    seen_in_train: pl.Series

    def __len__(self) -> int:
        return len(self.labels)


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXCLUDED_COLUMNS]


# Columns the dataset author declares categorical that arrive here as numbers.
#
# Vesta lists id_12 through id_38 as categorical. BigQuery's autodetect typed
# twelve of them as floats because they contain digits, and dtype is what
# prepare_features keys off -- so the model has been imposing an arithmetic
# ordering on codes. id_14 is a timezone, where that ordering is not merely
# unhelpful but meaningless.
#
# Measured cardinality: 4 to 522 levels, none continuous. See docs/eda.md §2.
DECLARED_CATEGORICAL_COLUMNS = frozenset(
    {
        "id_13", "id_14", "id_17", "id_18", "id_19", "id_20",
        "id_21", "id_22", "id_24", "id_25", "id_26", "id_32",
    }
)


def prepare_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Casts a raw BigQuery frame into what LightGBM wants.

    Object columns become pandas categoricals -- LightGBM splits on those
    natively, so no one-hot expansion of DeviceInfo's thousands of values.
    Floats drop to float32, which halves a ~1.3 GB training frame at a
    precision no split threshold in this data depends on.
    """
    features = frame.select(feature_columns(frame))
    
    exprs = []
    for column, dtype in zip(features.columns, features.dtypes):
        if column in DECLARED_CATEGORICAL_COLUMNS:
            exprs.append(pl.col(column).cast(pl.String).cast(pl.Categorical).alias(column))
        elif dtype == pl.Utf8 or dtype == pl.String:
            exprs.append(pl.col(column).cast(pl.Categorical).alias(column))
        elif dtype == pl.Float64:
            exprs.append(pl.col(column).cast(pl.Float32).alias(column))
        else:
            exprs.append(pl.col(column).alias(column))
            
    return features.select(exprs)


def to_lightgbm(frame: pl.DataFrame):
    """The one place a polars frame becomes something LightGBM will accept.

    LightGBM reads a polars frame through Arrow, and Arrow represents a categorical as a
    *dictionary* type, which LightGBM does not support: it aborts the process with
    "Unsupported Arrow type: dictionary" — `std::runtime_error` through `libc++abi`, so
    not an exception any caller could catch. Every fit died there.

    pandas is the supported path: LightGBM handles `category` dtype natively, splitting on
    the categories rather than on their codes, which is the behaviour `align_categories`
    exists to make consistent between splits. `to_pandas` maps polars Enum onto exactly
    that dtype, so the vocabulary and its order survive the crossing.

    One function rather than a `.to_pandas()` at each of the eight call sites: training,
    scoring, explainability and the serving container all have to hand LightGBM the same
    shape, and eight conversions are eight chances for one of them to drift.

    Booleans are cast to float first, and that is not cosmetic. numpy's `bool` has no
    representation for missing, so a nullable polars Boolean lands in pandas as `object`
    -- Python `True`/`False`/`None` -- and LightGBM rejects the frame with "pandas dtypes
    must be int, float or bool". Five columns here are exactly that shape (`M5`, `M6`,
    `id_35`, `id_36`, `id_37`), each missing on a quarter to three quarters of rows. As
    floats the values are 1.0/0.0 with NaN for missing, which is what a tree does with a
    boolean anyway: it splits at 0.5 and routes missing down its own branch.
    """
    booleans = [name for name, dtype in frame.schema.items() if dtype == pl.Boolean]
    if booleans:
        frame = frame.with_columns(pl.col(booleans).cast(pl.Float32))
    return frame.to_pandas()


def align_categories(frame: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """Gives `frame` the exact dtypes and categorical vocabularies `reference` was fitted with."""
    exprs = []
    for column, ref_dtype in zip(reference.columns, reference.dtypes):
        if column not in frame.columns:
            exprs.append(pl.lit(None, dtype=ref_dtype).alias(column))
        else:
            if ref_dtype == pl.Categorical or isinstance(ref_dtype, pl.Enum):
                # `drop_nulls` before the vocabulary is built. A null is a *value* an Enum
                # column may hold, never one of its categories, and `pl.Enum` raises
                # "Enum categories must not contain null values" rather than ignoring it.
                # Nearly every categorical here has missing values -- card4, card6, M4,
                # DeviceInfo, both email domains -- so this raised on the first split it
                # aligned and no model could be fitted at all. Missing stays missing:
                # LightGBM routes it down its own branch, which is the behaviour the
                # feature engineering already relies on.
                cats = (
                    reference.get_column(column).drop_nulls().unique().to_list()
                    if ref_dtype == pl.Categorical
                    else ref_dtype.categories
                )
                exprs.append(
                    pl.when(pl.col(column).cast(pl.String).is_in(cats))
                    .then(pl.col(column).cast(pl.String))
                    .otherwise(None)
                    .cast(pl.Enum(cats))
                    .alias(column)
                )
            elif ref_dtype == pl.Boolean and frame.schema[column] in (pl.Utf8, pl.String):
                exprs.append(
                    pl.when(pl.col(column).str.to_lowercase().is_in(["true", "t", "1"]))
                    .then(True)
                    .when(pl.col(column).str.to_lowercase().is_in(["false", "f", "0"]))
                    .then(False)
                    .otherwise(None)
                    .alias(column)
                )
            elif ref_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64) and frame.schema[column] not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64):
                exprs.append(pl.col(column).cast(ref_dtype, strict=False))
            else:
                exprs.append(pl.col(column).cast(ref_dtype))
    return frame.select(exprs)


def split_frame_from_dataframe(
    frame: pl.DataFrame, *, label: str, amount: str, seen: str
) -> SplitFrame:
    """Build a :class:`SplitFrame` from an in-memory frame.

    For notebooks: the pipeline gets its splits from BigQuery, but experimenting on a
    sampled CSV should not require a warehouse.
    """
    return SplitFrame(
        features=prepare_features(frame),
        labels=frame.get_column(label).cast(pl.Int8),
        amounts=frame.get_column(amount),
        seen_in_train=frame.get_column(seen).cast(pl.Boolean),
    )


import os
import time
from pathlib import Path


def _fetch_cached_split(client, project: str, split: str, model_input_table: str, split_table: str) -> pl.DataFrame:
    """Fetches a split from BigQuery or local Parquet cache."""
    cache_dir = Path("data/local/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model_input_table}_{split_table}_{split}.parquet"
    
    ttl_seconds = int(os.getenv("DAGSTER_CACHE_TTL_SECONDS", "86400")) # 1 day default
    
    if cache_file.exists():
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < ttl_seconds:
            return pl.read_parquet(cache_file)

    query = SPLIT_QUERY.format(
        model_input_table=f"{project}.features.{model_input_table}",
        split_table=f"{project}.features.{split_table}",
        split=split,
    )
    frame = pl.from_arrow(client.query(query).result().to_arrow())
    frame.write_parquet(cache_file)
    return frame

def load_split(client, project: str, split: str, *, model_input_table: str, split_table: str):
    """Returns one split as a SplitFrame. `client` is a BigQuery client."""
    frame = _fetch_cached_split(client, project, split, model_input_table, split_table)

    return SplitFrame(
        features=prepare_features(frame),
        labels=frame.get_column(LABEL_COLUMN).cast(pl.Int8),
        amounts=frame.get_column(AMOUNT_COLUMN).cast(pl.Float64),
        seen_in_train=frame.get_column("card_seen_in_train").cast(pl.Boolean),
    )


def load_raw_split(client, project: str, split: str, *, model_input_table: str, split_table: str) -> pl.DataFrame:
    """Returns the raw un-filtered frame for a split."""
    return _fetch_cached_split(client, project, split, model_input_table, split_table)



def split_with_contract(
    frame: pl.DataFrame,
    contract,
    *,
    seen_in_train: pl.Series,
    derivations: Sequence = (),
) -> SplitFrame:
    """Compute the contract's derived columns, then project onto its admitted set.

    The contract decides what the model sees. Columns it rejected — for time
    inconsistency, redundancy, or shift — never enter training, so the model
    cannot silently reintroduce them.

    This is a transform, not a load: the caller supplies the frame, because the
    same frame is needed unfiltered to derive the cold-entity flag. Splitting it
    into "fetch" and "project" is what keeps the second query from happening.

    ``derivations`` are the policy's declared derived columns. They are computed here, from
    the same function the audit used and the serving path will use, because a derivation
    applied in training and reimplemented at serving time is two definitions of one feature
    — the failure the contract exists to prevent.

    ``contract`` is a :class:`~fraud_detection.feature_contract.FeatureContract`.
    """
    features = prepare_features(apply_derivations(frame, derivations))

    # A column the contract admits but the table does not carry is a real
    # divergence, not something to route around. Dropping it quietly would
    # train a model on a smaller set than the one whose metrics get published;
    # assert_model_features_admitted catches it downstream, but only if the
    # frame is not silently patched up first.
    admitted = contract.training_features()
    if missing := [c for c in admitted if c not in features.columns]:
        raise KeyError(
            f"{len(missing)} admitted column(s) absent from the model input, "
            f"e.g. {missing[:5]} — the contract and the table have drifted apart"
        )

    return SplitFrame(
        features=features.select(admitted),
        labels=frame.get_column(LABEL_COLUMN).cast(pl.Int8),
        amounts=frame.get_column(AMOUNT_COLUMN).cast(pl.Float64),
        seen_in_train=seen_in_train,
    )
