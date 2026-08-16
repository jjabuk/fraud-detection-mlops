# Point-in-Time Correctness

> How this pipeline prevents lookahead bias.

---

## 1. The guarantee, stated exactly

> For every row in `features.transaction_features`, each engineered feature is computed
> **only** from transactions whose `TransactionDT` is strictly less than that row's own
> `TransactionDT`.

Three words carry the weight:

- **only** — no feature reads any column of the row it describes beyond the raw fields
  passed straight through.
- **strictly** — a transaction at the same instant is not "earlier". It is excluded.
- **less** — nothing later, obviously, but this is the part a naive implementation gets
  right by accident and the rest by luck.

Fraud detection is a forecasting problem wearing a classification costume. The label is
known only after the fact, so any feature that quietly encodes "what happened next"
produces a model that scores beautifully offline and collapses in production. Offline
metrics cannot detect this — a leaking model looks *better*, not worse.

---

## 2. Why the naive version leaks

The features are per-entity velocity aggregates: how many transactions has this card made
in the last hour, what did it usually spend, how long since it was last used. Every one is
a window function, and a window function has three ways to see the future.

### 2.1 Failure mode: the frame includes the current row

The default frame of an ordered window in SQL is
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Written without an explicit frame,
`COUNT(*) OVER (PARTITION BY card1 ORDER BY TransactionDT)` counts the row itself. Every
transaction then knows that at least one transaction exists on its card — itself — and
`card_txn_count_24h` is silently off by one in a way that correlates with the label.

**Mechanism:** every frame ends at `1 PRECEDING`, never `CURRENT ROW`.

### 2.2 Failure mode: `ROWS` instead of `RANGE`

This is the subtle one, and it is the reason this document exists.

- `ROWS BETWEEN … 1 PRECEDING` frames on **row position**: "the row before this one in the
  sort order".
- `RANGE BETWEEN … 1 PRECEDING` frames on the **`ORDER BY` value**: "every row whose
  `TransactionDT` is at most `current − 1`".

They diverge exactly when two transactions on the same card share a `TransactionDT`. Under
`ROWS`, the sort order breaks the tie arbitrarily and one peer sees the other. Under
`RANGE`, both are excluded from each other's frame, because neither has a `TransactionDT`
strictly below the other's.

That difference is not academic here. Two simultaneous transactions on one card is among
the strongest fraud signals in the dataset — card testing looks precisely like that. A
`ROWS` frame hands the model a peek at the very pattern it is supposed to infer.

**Mechanism:** every frame is `RANGE`. `ROWS` frames are banned.

> **Why `1 PRECEDING` cleanly means "strictly earlier" here.** `TransactionDT` is an
> integer count of seconds from a fixed reference, so the smallest gap between two
> distinct timestamps is 1 and `RANGE … 1 PRECEDING` excludes peers and nothing else. This
> is a property of the column's type, not a universal truth: over a `FLOAT64` or
> `TIMESTAMP` ordering key, `1 PRECEDING` would also discard genuinely-earlier rows inside
> the last unit. Anyone changing the time column has to revisit the bound.

### 2.3 Failure mode: positional navigation functions

`LAG(TransactionDT) OVER (PARTITION BY card1 ORDER BY TransactionDT)` looks like the
obvious way to compute "seconds since the previous transaction". It has the `ROWS` problem
with no frame clause to fix: `LAG` is positional by definition. For tied timestamps it
returns a peer, and the computed gap is `0`.

**Mechanism:** `LAG` and `LEAD` are banned. The gap is an aggregate over a `RANGE` frame:

```sql
TransactionDT - MAX(TransactionDT) OVER (
  PARTITION BY card1 ORDER BY TransactionDT
  RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
) AS seconds_since_prev_txn_card
```

`MAX` over "everything strictly earlier" *is* the previous transaction's timestamp, and it
inherits the frame's peer exclusion for free.

---

## 3. This failure actually happened here

The three mechanisms above were not derived in the abstract. Two of them were in place
from the start; the third was missing, and the pipeline shipped with it missing.

`seconds_since_prev_txn_card` used `LAG`. Every other aggregate used a correct `RANGE`
frame, the tests asserted point-in-time correctness, and `MEASUREMENTS.md` stated the
guarantee in absolute terms — *"excludes both the current row and every peer sharing its
exact timestamp"*. All of that was true of the other six features and false of this one.

It surfaced on **2026-08-10**, from running the pipeline against the full dataset and
querying the output rather than reading the code:

```sql
SELECT COUNTIF(seconds_since_prev_txn_card = 0) FROM features.transaction_features
-- 166
```

166 rows out of 590,540 — 0.03% — had learned that another transaction was hitting the
same card at the same instant. After the fix, the same query returns `0` and the minimum
gap is `1`.

Two things are worth taking from this. First, the leak was in the one feature that used a
different SQL construct from all the others; uniformity is a correctness property, not a
style preference. Second, the unit tests could not have caught it, because they asserted
on the shape of the generated SQL and the SQL was shaped exactly as its author intended.
Only running it and interrogating the result found it.

---

## 4. What is deliberately *not* part of the guarantee

### The entity definition

Velocity aggregates group by `card1` (and `DeviceInfo` for the device features). `card1` is
a proxy — a tighter entity would combine several card and address fields. Choosing a
different entity changes which rows share a partition, and therefore changes the feature
values. It does not change the guarantee: whatever the partition, the frame still admits
only strictly-earlier rows. Entity selection is a modelling question; leakage is not.

### The train/test split

Point-in-time features remove leakage *within* a row. They do not remove it *across* the
split. A random K-fold would place a card's later transactions in train and earlier ones in
test, which leaks in a way no window frame can prevent. The split is therefore time-based
on `TransactionDT`, with an additional guard against a card proxy appearing on both sides.
That work is Plane 3 and is **not built yet** — until it is, this document describes only
half of the leakage story.

### Serving time

This is the largest open gap. The guarantee above is about the *training table*, which has
one row per transaction with features as of that transaction. At serving time the question
is different: what are this card's aggregates *right now*, at request time `T`?

An entity-keyed snapshot answers as of the entity's last observed transaction, not as of
`T`. For a card that has been quiet for three hours, the true `card_txn_count_1h` is 0 and
a snapshot will say otherwise. That is not lookahead bias — it is staleness, its mirror
image — but it breaks the same contract, that training and serving compute the same
function. Plane 3 settles it by measurement; see `MEASUREMENTS.md`, 2026-08-10.

### The ingestion and join layer

`joined_transactions_identity` is a left join on `TransactionID` plus a null count. It has
no time semantics and no ordering, so it cannot leak. `null_count_V_block` is a property of
the row itself and of nothing else.

---

## 5. How the guarantee is enforced

**In the SQL.** All four frames are declared in one `WINDOW` clause, so they are read
together and a new feature reuses a named window rather than inventing a frame:

| Window | Frame | Feeds |
| --- | --- | --- |
| `card_1h` | `RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING` | `card_txn_count_1h` |
| `card_24h` | `RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING` | `card_txn_count_24h`, `card_txn_amt_avg_24h`, `card_txn_amt_sum_24h`, `card_amt_deviation_24h` |
| `card_prior` | `RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` | `seconds_since_prev_txn_card` |
| `device_24h` | `RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING` | `device_txn_count_24h` |

**In the tests.** Structural assertions over the generated SQL, which is what a mocked
BigQuery client permits:

- every frame ends at `1 PRECEDING`, and the count of `RANGE BETWEEN` equals the count of
  `PRECEDING AND 1 PRECEDING` — so no bare `OVER (…)` can slip in with a default frame;
- `LAG(`, `LEAD(` and `ROWS BETWEEN` appear nowhere in the query.

These are guards against regression, not proofs of correctness. They would not have caught
the `LAG` leak, and the tests say so in their own docstrings.

**Against the data.** The checks that actually prove it, run after materialization
(590,540 rows, 2026-08-10):

| Query | Result | What it establishes |
| --- | --- | --- |
| `COUNTIF(seconds_since_prev_txn_card = 0)` | `0` | No row sees a peer |
| `MIN(seconds_since_prev_txn_card)` | `1` | The gap is strictly positive — "strictly earlier" is real |
| `COUNTIF(seconds_since_prev_txn_card IS NULL)` | `13553` | Rows with no earlier transaction on their card |
| `COUNT(DISTINCT card1)` | `13553` | **Exactly equal** to the line above: every card has exactly one first-ever transaction, and no card has a tie at its earliest timestamp |
| `COUNTIF(card_txn_count_24h = 0)` | `127607` | Rows whose card was idle for the preceding 24h |
| `COUNTIF(card_txn_amt_avg_24h IS NULL)` | `127607` | Identical by construction: `COUNT` over an empty frame is 0, `AVG` is NULL |

The last two pairs are the useful ones. Each is two independently computed numbers that
must agree if the frames mean what they claim, and they agree exactly.

---

## 6. Rules for adding a feature

1. Reuse a named window from the `WINDOW` clause. If a new frame is genuinely needed, it is
   `RANGE`, and it ends at `1 PRECEDING`.
2. No `LAG`, no `LEAD`, no `ROWS` frames. If a positional function seems necessary, it can
   be rewritten as an aggregate over a `RANGE` frame — `MAX` for the previous value, and so
   on.
3. Never reference a column of the current row other than fields passed through unchanged
   from raw.
4. Ship the empirical check with the feature, not just the unit test. Find the pair of
   numbers that must agree, and put it in section 5.
