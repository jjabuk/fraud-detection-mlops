# Point-in-time correctness

How this pipeline keeps lookahead bias out of the feature table, and where the guarantee
stops.

## 1. The guarantee

For every row in `features.transaction_features`, each engineered feature is computed only
from transactions whose `TransactionDT` is strictly less than that row's own `TransactionDT`.

"Strictly" is load-bearing: a transaction at the same instant is not earlier and is excluded.
Offline metrics cannot catch a violation, because a leaking model looks better rather than
worse, so the guarantee has to be a property of the SQL rather than something the metrics
would reveal.

## 2. Three ways a window function sees the future

### 2.1 The frame includes the current row

The default frame of an ordered window in SQL is
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Written without an explicit frame,
`COUNT(*) OVER (PARTITION BY card1 ORDER BY TransactionDT)` counts the row itself, so every
transaction knows that at least one transaction exists on its card, and the count is off by
one in a way that correlates with the label.

Every frame here ends at `1 PRECEDING`, never `CURRENT ROW`.

### 2.2 `ROWS` instead of `RANGE`

`ROWS BETWEEN … 1 PRECEDING` frames on row position: the row before this one in the sort
order. `RANGE BETWEEN … 1 PRECEDING` frames on the `ORDER BY` value: every row whose
`TransactionDT` is at most `current − 1`.

They diverge when two transactions on the same card share a `TransactionDT`. Under `ROWS` the
sort order breaks the tie arbitrarily and one peer sees the other. Under `RANGE` both are
excluded from each other's frame.

That matters on this dataset specifically: two simultaneous transactions on one card is among
the strongest fraud signals in it, since card testing looks exactly like that. A `ROWS` frame
hands the model a look at the pattern it is supposed to infer.

`1 PRECEDING` means "strictly earlier" here because `TransactionDT` is an integer count of
seconds, so the smallest gap between two distinct timestamps is 1. Over a `FLOAT64` or
`TIMESTAMP` ordering key the same bound would also discard genuinely earlier rows inside the
last unit. Anyone changing the time column has to revisit it.

### 2.3 Positional navigation functions

`LAG(TransactionDT) OVER (PARTITION BY card1 ORDER BY TransactionDT)` is the obvious way to
compute "seconds since the previous transaction", and it has the `ROWS` problem with no frame
clause available to fix it. For tied timestamps it returns a peer and the computed gap is `0`.

`LAG` and `LEAD` are banned. The gap is an aggregate over a `RANGE` frame instead:

```sql
TransactionDT - MAX(TransactionDT) OVER (
  PARTITION BY card1 ORDER BY TransactionDT
  RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
) AS seconds_since_prev_txn_card
```

`MAX` over everything strictly earlier is the previous transaction's timestamp, and it
inherits the frame's peer exclusion.

## 3. The failure that actually happened

Two of the three mechanisms above were in place from the start. The third was missing and the
pipeline shipped without it.

`seconds_since_prev_txn_card` used `LAG`. Every other aggregate used a correct `RANGE` frame,
the tests asserted point-in-time correctness, and the docs stated the guarantee in absolute
terms. All of that was true of the other features and false of this one.

It surfaced on 2026-08-10, from running the pipeline against the full dataset and querying the
output rather than reading the code:

```sql
SELECT COUNTIF(seconds_since_prev_txn_card = 0) FROM features.transaction_features
-- 166
```

166 rows out of 590,540 had learned that another transaction was hitting the same card at the
same instant. After the fix the same query returns `0` and the minimum gap is `1`.

Two things came out of it. The leak was in the one feature that used a different SQL construct
from all the others, which makes uniformity a correctness property here rather than a style
preference. And the unit tests could not have caught it, because they asserted on the shape of
the generated SQL and the SQL was shaped exactly as its author intended.

## 4. What the guarantee does not cover

**The entity definition.** Velocity aggregates partition by `card1`, `DeviceInfo` and the
reconstructed client. Choosing a different entity changes which rows share a partition and
therefore the feature values, but not the guarantee: whatever the partition, the frame admits
only strictly earlier rows.

**The split.** Point-in-time features remove leakage within a row, not across the split. A
random K-fold would put a card's later transactions in train and its earlier ones in test,
which no window frame can undo. The split is therefore time-based on `TransactionDT`, its
boundaries use `<=` so peers of a timestamp always land on the same side, and it labels each
row with `card_seen_in_train` rather than dropping returning cards, since returning cards are
the production reality and the useful thing is to report metrics for both groups separately.

**Serving time.** The training table answers "what were this entity's aggregates as of this
transaction". A live request asks "what are they right now", and an entity-keyed snapshot
answers as of the entity's last observed transaction instead. For a card quiet for three hours
the true `card_txn_count_1h` is 0 and a snapshot says otherwise. That is staleness rather than
lookahead, but it breaks the same contract that training and serving compute the same
function. The batch path avoids it by computing the windows over `raw.scoring_history`
(train ∪ test); an online path would have to solve it per request, which is why online serving
is out of scope in [architecture.md](architecture.md).

**Ingestion and join.** `joined_transactions_identity` is a left join plus a null count. It
has no time semantics and no ordering, so it cannot leak.

## 5. Enforcement

**In the SQL.** All six frames are declared in one `WINDOW` clause, so a new feature reuses a
named window instead of inventing a frame:

| Window | Frame | Feeds |
| --- | --- | --- |
| `card_1h` | `RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING` | `card_txn_count_1h` |
| `card_24h` | `RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING` | `card_txn_count_24h`, `card_txn_amt_avg_24h`, `card_txn_amt_sum_24h`, `card_amt_deviation_24h` |
| `card_prior` | `RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` | `seconds_since_prev_txn_card` |
| `client_24h` | `RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING` | `client_txn_count_24h` |
| `client_prior` | `RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` | `client_txn_count_prior`, `client_amt_avg_prior`, `client_amt_deviation_prior`, `seconds_since_prev_txn_client` |
| `device_24h` | `RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING` | `device_txn_count_24h` |

**In the tests.** Structural assertions over the generated SQL, which is what a mocked
BigQuery client allows: every frame ends at `1 PRECEDING`, the count of `RANGE BETWEEN` equals
the count of `PRECEDING AND 1 PRECEDING` so no bare `OVER (…)` can slip in with a default
frame, and `LAG(`, `LEAD(` and `ROWS BETWEEN` appear nowhere. These are regression guards, not
proofs; they would not have caught §3, and the tests say so in their docstrings.

**Against the data.** The checks that actually establish it, run after materialization
(590,540 rows, 2026-08-10):

| Query | Result | What it establishes |
| --- | --- | --- |
| `COUNTIF(seconds_since_prev_txn_card = 0)` | `0` | No row sees a peer |
| `MIN(seconds_since_prev_txn_card)` | `1` | The gap is strictly positive |
| `COUNTIF(seconds_since_prev_txn_card IS NULL)` | `13553` | Rows with no earlier transaction on their card |
| `COUNT(DISTINCT card1)` | `13553` | Equal to the line above: every card has exactly one first transaction and no card has a tie at its earliest timestamp |
| `COUNTIF(card_txn_count_24h = 0)` | `127607` | Rows whose card was idle for the preceding 24h |
| `COUNTIF(card_txn_amt_avg_24h IS NULL)` | `127607` | Identical by construction: `COUNT` over an empty frame is 0, `AVG` is NULL |

The last two pairs are the useful ones. Each is two independently computed numbers that have
to agree if the frames mean what they claim.

## 6. Adding a feature

1. Reuse a named window. A genuinely new frame is `RANGE` and ends at `1 PRECEDING`.
2. No `LAG`, no `LEAD`, no `ROWS` frames. A positional function can always be rewritten as an
   aggregate over a `RANGE` frame.
3. Never reference a column of the current row other than fields passed through unchanged from
   raw.
4. Ship the empirical check with the feature, not only the unit test. Find the pair of numbers
   that must agree and add it to §5.
