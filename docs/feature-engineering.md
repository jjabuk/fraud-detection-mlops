# Feature Engineering

> What the feature store holds, what it deliberately does not, and how the model's input is
> assembled from two tables rather than one.
> Leakage guarantee: [point-in-time.md](point-in-time.md) · Dataset shape: [../eda/README.md](../eda/README.md) ·
> Implementation: [`feature_engineering.py`](../src/fraud_detection/assets/feature_engineering.py)

---

## 1. The feature store is a table of *entity state*, not of model input

`features.transaction_features` holds nineteen columns: seven passed through from raw and
twelve engineered. It does not hold the roughly four hundred predictive columns that the
model actually trains on. That is not an omission — it is the design, and this document
exists because the reason is not obvious from looking at the table.

---

## 2. Decision: the model input is assembled from two tables

**Decision.** The training matrix is `raw.ieee_train_joined` joined to `features.transaction_features` on `TransactionID`.

**Why:** Serving time input is `request fields + retrieved entity state`. Therefore, training must assemble the exact same input by joining the raw transaction table with the retrieved feature table. Widening the feature table to hold raw predictors or pushing velocity aggregates into the raw table creates divergent code paths (training-serving skew).

---

## 3. What is in each table

### `raw.ieee_train_joined` — 435 columns

`transaction` left-joined to `identity` on `TransactionID`, plus `null_count_V_block`.
Every column is a property of the transaction itself. Nothing here is derived from other
rows, so nothing here can leak. See [architecture.md](architecture.md), Module 1.

### `features.transaction_features` — 19 columns

Twelve engineered aggregates over three entities. The list lives in one place,
[`schema.FEATURE_COLUMNS`](../src/fraud_detection/schema.py), which the SQL that builds
them, the assembly that names them, and the contract that classifies them as
retrieved-at-serving-time all read.

| Column | Definition | Entity |
| --- | --- | --- |
| `card_txn_count_1h` | Transactions on this card in the preceding hour | `card1` |
| `card_txn_count_24h` | Transactions on this card in the preceding 24h | `card1` |
| `card_txn_amt_avg_24h` | Mean amount on this card over the preceding 24h | `card1` |
| `card_txn_amt_sum_24h` | Total amount on this card over the preceding 24h | `card1` |
| `card_amt_deviation_24h` | `TransactionAmt` minus the 24h mean above | `card1` |
| `seconds_since_prev_txn_card` | Gap to the card's previous transaction | `card1` |
| `device_txn_count_24h` | Transactions on this device in the preceding 24h | `DeviceInfo` |
| `client_txn_count_prior` | Transactions by this client, ever, before this one | `client_uid` |
| `client_txn_count_24h` | Transactions by this client in the preceding 24h | `client_uid` |
| `client_amt_avg_prior` | Mean amount over the client's prior transactions | `client_uid` |
| `client_amt_deviation_prior` | `TransactionAmt` minus that mean | `client_uid` |
| `seconds_since_prev_txn_client` | Gap to the client's previous transaction | `client_uid` |

Plus `TransactionID` (the join key), `TransactionDT`, `TransactionAmt`, `card1`,
`DeviceInfo`, `client_uid` and `isFraud`, carried so the table is independently
inspectable. `client_uid` is an entity key and never a feature — it is excluded by
[`schema.EXCLUDED_COLUMNS`](../src/fraud_detection/schema.py) and the exclusion is pinned
by a test.

**The client entity.** `card1 + addr1 + (day − D1)`, where `D1` is days since the card
began, so `day − D1` recovers the day it began — a constant across that client's history.
It reaches 98.5% label purity against 84.8% for `card1`; the reconstruction, and the reason
its real value is an honest cold-entity denominatorThis is the Entity Purity claim. It is measured and pinned by a test: see
[../src/fraud_detection/evaluation/README.md](../src/fraud_detection/evaluation/README.md). Every client aggregate is
computed over this reconstructed grouping rather than over `card1`.econstructable id.

### Computing them yourself

The statement lives in [`fraud_detection.features`](../src/fraud_detection/features.py),
not in the Dagster asset, and runs on two engines: BigQuery in the pipeline, DuckDB in a
notebook. Same SQL, so there is no second definition of a feature to drift.

```python
import polars as pl
from fraud_detection.features import compute_locally

tx = pl.read_csv("kaggle/raw/train_transaction.csv",
                 columns=["TransactionID", "TransactionDT", "TransactionAmt",
                          "isFraud", "card1", "addr1", "D1"])
identity = pl.read_csv("kaggle/raw/train_identity.csv",
                       columns=["TransactionID", "DeviceInfo"])

features = compute_locally(tx.join(identity, on="TransactionID", how="left"))
```

590,540 rows in **1.7 s**, no cloud. It reproduces the pipeline's numbers exactly: 11.3%
null `client_uid`, 199,070 distinct clients, 79.9% null `device_txn_count_24h`, and
`seconds_since_prev_txn_card` null on exactly 13,553 rows — the number of distinct `card1`
values, which is the consistency check that the windows mean what they claim.

One caveat the docstring repeats: velocity aggregates count an entity's *neighbours*, so a
random row sample undercounts them. Slice a contiguous period, or read the whole file.

Every one of the twelve is computed over a `RANGE … 1 PRECEDING` frame. "Preceding" means
strictly earlier — never the current transaction, never a peer sharing its exact
timestamp. That guarantee, the failure modes it defends against, and the leak that was
found and fixed on 2026-08-10 are documented in [point-in-time.md](point-in-time.md).

**The asymmetry in the entity columns, and the bug it caused.** `card1` has zero nulls. `DeviceInfo` is present on only 118,666 rows.

Defect found 2026-08-10: SQL places every NULL in the same window partition. `COUNT(*) OVER device_24h` counted other device-less transactions on the 471,874 rows with no device, creating a false signal. Fixed by making the aggregate `NULL` when `DeviceInfo` is `NULL`.

---

## 4. Assembly

`features.model_input` is the join: every column of the raw joined table, plus the seven
engineered columns named explicitly. The overlapping pass-through columns come from the
raw side only, so nothing is duplicated.

The engineered column names live in one constant, `FEATURE_COLUMNS`, so the assembly can
never silently drop a feature that feature engineering started producing. A `SELECT *` on
the feature side would have been shorter and would have re-introduced the six duplicated
pass-through columns.

---

## 5. Deliberately absent

**Reduction of `V1–V339`.** 129 representatives out of 339 (62% cut) were found. See [../src/fraud_detection/evaluation/README.md](../src/fraud_detection/evaluation/README.md). Full block stays until retraining cost is measured.

**Online-shaped features.** `features.transaction_features` is keyed by `TransactionID`. Serving needs one row per entity, current state.

**A cold-entity denominator in the gate.** The gate segments by `card1`. 98.9% of holdout rows sit on a card already seen in training. Repointing the gate at `client_uid` drops the seen share to 51.5%.
