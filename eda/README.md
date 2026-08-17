# Exploratory analysis

One notebook per question, named after the question. Each runs on its own: the setup it needs
is carried into it, so no notebook depends on another having been run first.

Notebooks import the same modules the pipeline runs (`fraud_detection.evaluation`,
`fraud_detection.training`), which is enforced by
[`tests/test_layering.py`](../tests/test_layering.py). An analysis here cannot measure a
different implementation from the one that reaches production.

Where a technique comes from published Kaggle work, the notebook says so in its first cell and
[ATTRIBUTION.md](../ATTRIBUTION.md) records what was taken. Data is not committed, see
[docs/setup.md](../docs/setup.md).

## The data, before any modelling

- [Do the declared column types match what the data actually holds?](notebooks/do-the-declared-column-types-match-the-data.ipynb)
- [How many distinct values does each column take, and what does that imply about its type?](notebooks/how-many-distinct-values-does-each-column-take.ipynb)
- [What is the fraud base rate, and does it drift across the time axis?](notebooks/what-is-the-fraud-base-rate-and-does-it-drift.ipynb)
- [Does fraud follow a daily cycle?](notebooks/does-fraud-follow-a-daily-cycle.ipynb)
- [Does transaction amount predict fraud?](notebooks/does-transaction-amount-predict-fraud.ipynb)
- [Is missingness itself a signal, or just absence?](notebooks/is-missingness-itself-a-signal.ipynb)
- [Do the V columns fall into families that share a missingness pattern?](notebooks/do-v-columns-share-a-missingness-pattern.ipynb)

## Who is the customer, when the dataset does not say

- [Can a client be reconstructed when the dataset has no customer id?](notebooks/can-a-client-be-reconstructed-without-a-customer-id.ipynb)
- [Which candidate identifier groups transactions most purely by label?](notebooks/which-identifier-groups-transactions-most-purely.ipynb)
- [Are the pure groups just singletons?](notebooks/are-pure-groups-just-singletons.ipynb)
- [What happens to rows with a missing identifier component?](notebooks/what-happens-to-rows-with-a-missing-id-component.ipynb)
- [How many rows does each identifier actually cover?](notebooks/how-many-rows-does-each-identifier-cover.ipynb)
- [Which segments carry the most variance in fraud rate?](notebooks/which-segments-have-the-highest-fraud-variance.ipynb)
- [What do those segments imply for the model downstream?](notebooks/what-do-those-segments-imply-downstream.ipynb)

## Does a feature still mean the same thing later?

- [Which single features rank one way early and the other way late?](notebooks/which-single-features-invert-between-windows.ipynb)
- [Does a whole V block invert together?](notebooks/does-a-whole-v-block-invert-together.ipynb)
- [How many of the 377 raw features invert?](notebooks/how-many-of-377-features-invert.ipynb)
- [Does the inversion verdict survive a wider window?](notebooks/does-the-verdict-survive-a-wider-window.ipynb)
- [How does the label itself move over time?](notebooks/how-does-the-label-itself-move-over-time.ipynb)
- [Is the fraud label a property of the client or the transaction?](notebooks/is-fraud-a-property-of-the-client.ipynb)

## Has the population moved?

- [Which columns shift between an early and a late window?](notebooks/which-columns-shift-between-early-and-late-windows.ipynb)
- [Is the measured shift driven by missingness rather than by values?](notebooks/is-the-shift-driven-by-missingness.ipynb)
- [Can a model tell early rows from late ones at all?](notebooks/can-a-model-tell-early-rows-from-late-ones.ipynb)
- [Do columns separate the windows in combination that do not separate them alone?](notebooks/do-columns-separate-the-windows-in-combination.ipynb)
- [Do PSI and adversarial validation agree on which columns moved?](notebooks/do-psi-and-adversarial-validation-agree.ipynb)
- [Why do unpinned bins give a different answer?](notebooks/why-do-unpinned-bins-give-a-different-answer.ipynb)

## Which columns are saying the same thing twice?

- [Do the missingness families hold on this data?](notebooks/do-the-missingness-families-hold-on-this-data.ipynb)
- [Which columns belong to no family?](notebooks/which-columns-belong-to-no-family.ipynb)
- [How should a representative be chosen within each group?](notebooks/how-should-a-representative-be-chosen-per-group.ipynb)
- [Is each group more tightly bound inside than to its neighbours?](notebooks/is-each-group-tighter-inside-than-to-its-neighbours.ipynb)
- [Does the V block split into two global halves?](notebooks/does-the-v-block-split-into-two-global-halves.ipynb)
- [What does PCA per family cost, and why is it not the default?](notebooks/what-does-pca-per-family-cost.ipynb)
- [How much variance does one component carry per family?](notebooks/how-much-variance-does-one-component-carry.ipynb)
- [Does forward selection find a smaller set that performs as well?](notebooks/does-forward-selection-find-a-smaller-set.ipynb)

## Encoding categoricals

- [Does rarity predict fraud, and in which columns?](notebooks/does-rarity-predict-fraud.ipynb)
- [What shape do one-hot and frequency encoding actually have on this data?](notebooks/what-shape-do-the-two-encodings-have.ipynb)
- [Does either encoding move the model beyond the noise band?](notebooks/does-either-encoding-move-the-model.ipynb)
- [Where does an encoding help, on seen clients or unseen ones?](notebooks/where-does-an-encoding-help-seen-or-unseen-clients.ipynb)

## What the contract decided, and whether it was worth it

- [How many declared columns survive the audits?](notebooks/how-many-columns-survive-the-audits.ipynb)
- [Where in the funnel are columns lost?](notebooks/where-in-the-funnel-are-columns-lost.ipynb)
- [Which audit rejects the most, and does that make it the strictest?](notebooks/which-audit-rejects-the-most.ipynb)
- [Which specific columns were rejected, and on what number?](notebooks/which-specific-columns-were-rejected-and-why.ipynb)
- [Is the admitted feature set better than a random set of the same size?](notebooks/is-the-admitted-set-better-than-a-random-one.ipynb)
- [Is the admitted set stable across nearby policy settings?](notebooks/is-the-admitted-set-stable-across-thresholds.ipynb)
- [What does the contract buy that the metric cannot see?](notebooks/what-does-the-contract-buy-that-the-metric-cannot-see.ipynb)

## What the model gets wrong

- [Which amount band is the model worst at?](notebooks/which-amount-band-is-the-model-worst-at.ipynb)
- [Which product is the model worst at?](notebooks/which-product-is-the-model-worst-at.ipynb)
- [Does model quality depend on how old the card is?](notebooks/does-model-quality-depend-on-entity-age.ipynb)
- [Does the model miss more fraud by value than by count?](notebooks/does-the-model-miss-more-by-value-than-by-count.ipynb)
- [How fast does performance decay as the scoring window moves away from training?](notebooks/how-fast-does-performance-decay-after-training.ipynb)
- [Is that decay time, or is it population mix?](notebooks/is-that-decay-time-or-population-mix.ipynb)

51 notebooks.
