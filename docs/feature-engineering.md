# Feature engineering

What the feature table holds, what it does not, and how the model input is assembled from two
tables rather than one. The leakage guarantee behind every aggregate here is in
[point-in-time.md](point-in-time.md); the SQL is in
[`features/features.py`](../src/fraud_detection/features/features.py).

## 1. Two tables, joined at training time

The training matrix is `raw.ieee_train_joined` joined to `features.transaction_features` on
`TransactionID`.

Serving input is `request fields + retrieved entity state`, so training assembles the same
shape: the raw transaction on one side, the retrieved entity state on the other. Widening the
feature table to hold raw predictors, or pushing velocity aggregates into the raw table, would
create a second code path that only training uses.

That is also why `features.transaction_features` does not hold the ~400 predictive columns the
model trains on. It is a table of entity state, not of model input.

## 2. What is in each table

### `raw.ieee_train_joined`, 435 columns

`transaction` left-joined to `identity` on `TransactionID`, plus `null_count_V_block`. Every
column is a property of the transaction itself, so nothing here can leak.

### `features.transaction_features`, 47 columns

Seven columns carried through (`TransactionID`, `TransactionDT`, `TransactionAmt`, `card1`,
`DeviceInfo`, `client_uid`, `isFraud`) so the table is independently inspectable, twelve core
aggregates, and a configurable block of client-level uid aggregates.

The twelve are listed in
[`schema.py`](../src/fraud_detection/schema.py) as `FEATURE_COLUMNS`, which the SQL
that builds them, the assembly that names them and the contract that classifies them as
retrieved-at-serving-time all read from.

| Column | Definition | Entity | In the contract |
| --- | --- | --- | --- |
| `card_txn_count_1h` | Transactions on this card in the preceding hour | `card1` | admitted |
| `card_txn_count_24h` | Transactions on this card in the preceding 24h | `card1` | admitted |
| `card_txn_amt_avg_24h` | Mean amount on this card over the preceding 24h | `card1` | admitted |
| `card_txn_amt_sum_24h` | Total amount on this card over the preceding 24h | `card1` | admitted |
| `card_amt_deviation_24h` | `TransactionAmt` minus the 24h mean above | `card1` | admitted |
| `seconds_since_prev_txn_card` | Gap to the card's previous transaction | `card1` | admitted |
| `device_txn_count_24h` | Transactions on this device in the preceding 24h | `DeviceInfo` | rejected |
| `client_txn_count_prior` | Transactions by this client, ever, before this one | `client_uid` | rejected |
| `client_txn_count_24h` | Transactions by this client in the preceding 24h | `client_uid` | admitted |
| `client_amt_avg_prior` | Mean amount over the client's prior transactions | `client_uid` | admitted |
| `client_amt_deviation_prior` | `TransactionAmt` minus that mean | `client_uid` | rejected |
| `seconds_since_prev_txn_client` | Gap to the client's previous transaction | `client_uid` | rejected |

Admitted or rejected is read off `references/feature-contract.json`. Which check rejected a
column, and the number behind it, are recorded there too; the analysis that produced them is
in [`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda).

The uid block is declared in [`config/feature-admission.toml`](../config/feature-admission.toml)
and generated into the same statement: the standard deviation of six derived `D` columns and
the mean of fourteen `C` and eight `M` columns over the client's prior transactions, 28 columns
at current settings. The contract judges them like anything else and currently admits seven.
Nineteen of the rejected ones were readmitted by a policy override until that override was
measured at −0.0002 ROC-AUC and retired ([MEASUREMENTS.md](MEASUREMENTS.md)).

Computed but rejected columns still cost a BigQuery statement and nothing else: they never
reach the model, and the contract records why.

### The client entity

`client_uid` is `card1 + addr1 + (day − D1)`, where `D1` is days since the card began, so
`day − D1` recovers the day it began and is constant across that client's history. Whether
that reconstruction identifies a real customer is a question about the data, answered in
[`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda); the contract carries the verdict as its
`entity` block. The expression is built by
[`features/features.py`](../src/fraud_detection/features/features.py) as `CLIENT_UID_EXPRESSION`, carried into `model_input` as a column, and pinned
by a test. Every client aggregate is computed over this grouping rather than over `card1`.

`client_uid` is an entity key and never a feature; it is listed in `schema.EXCLUDED_COLUMNS`
and the exclusion is pinned by a test.

### Null entities

`card1` has no nulls. `DeviceInfo` is present on 118,666 rows of 590,540.

A defect found on 2026-08-10: SQL places every NULL in the same window partition, so
`COUNT(*) OVER device_24h` was counting other device-less transactions on the 471,874 rows
without a device, which produced a feature that looked informative and measured nothing. Every
device and client aggregate is now `NULL` when its entity key is `NULL`.

## 3. Assembly

`features.model_input` is the join: every column of the raw joined table, plus the engineered
columns named explicitly from `FEATURE_COLUMNS` and the uid block. Overlapping pass-through
columns come from the raw side only.

Naming the columns from one constant means the assembly cannot silently drop a feature that
feature engineering started producing. A `SELECT *` on the feature side would have been shorter
and would have reintroduced the six duplicated pass-through columns.

## 4. Where the SQL lives

The statement is built by `build_sql(...)` in `features/features.py`, not embedded
in the Dagster asset. The asset supplies table names and calls it, which is what lets the tests
assert on the generated SQL without a warehouse and lets the uid block be generated from
config rather than hand-maintained.

Shape of the output on the full dataset: 590,540 rows, 11.3% null `client_uid`, 199,070
distinct clients, 79.9% null `device_txn_count_24h`, and `seconds_since_prev_txn_card` null on
exactly 13,553 rows, which is the number of distinct `card1` values.

One caveat when sampling: velocity aggregates count an entity's neighbours, so a random row
sample undercounts them. Slice a contiguous period, or read the whole file.

## 5. Deliberately absent

**Reduction of `V1–V339`.** The audits identify a far smaller set of representatives that
carries the block's information ([`ieee-cis-fraud-detection-eda`](https://github.com/jjabuk/ieee-cis-fraud-detection-eda)). The full block
stays here until the retraining cost of dropping it is measured, which is this side's job.

**Online-shaped features.** This table is keyed by `TransactionID`. Serving would need one row
per entity holding current state, which is the lookup described in
[architecture.md](architecture.md) §5.

**A cold-entity denominator on `card1`.** The gate segments by client rather than card: 98.9%
of holdout rows sit on a card already seen in training, so a per-card version of that check
would ask almost nothing. On `client_uid` the seen share drops to 51.5%.
