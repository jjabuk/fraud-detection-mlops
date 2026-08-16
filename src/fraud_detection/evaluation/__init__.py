"""Feature-selection and data-understanding techniques, each as testable code.

The IEEE-CIS competition's published solutions describe a set of checks their authors ran
by hand, once. Each module here implements one of them as a reusable audit whose output is
a pinned artefact rather than a notebook to re-read:

``time_consistency``    does a feature still rank the same way months later?
``distribution_shift``  did a column move, and can a model tell the periods apart?
``entity_purity``       rebuild the customer the data does not name, and prove it is real.
``redundancy``          collapse families of columns that carry the same signal.
``selection``           reduce by PCA, or grow a feature set one column at a time.

Every module produces a ``Fragment`` for ``fraud_detection.feature_contract``, so the
result of an audit becomes a decision the training pipeline and the serving schema both
read from one place.
"""
