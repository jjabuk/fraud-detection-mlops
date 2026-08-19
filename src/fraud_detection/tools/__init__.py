"""Commands that compose the pure layer with the warehouse.

The criterion is exact and is enforced, not described: **a module belongs here
if it reaches outside the process** — BigQuery, GCS, the registry — while still
being something a person runs by hand rather than something the orchestrator
materialises.

That is why these are not in `features/` or `training/` even though that is what
they are *about*. Those packages are the pure layer: a notebook imports them, the
serving path reuses them, and the moment one of them opens a BigQuery client,
importing it drags a cloud SDK in with it. `tests/test_layering.py` asserts that
nothing in the pure layer imports `dagster` or `google`, and these two do.

The dependency runs one way. These may import the pure layer and the
orchestrator's resources; nothing in the pure layer may import these.

What lives here now:

``noise_band``      how much a metric moves between seeds, which is the only
                    thing that makes a promotion gate's threshold meaningful.
``frequency_maps``  the pinned frequency encodings, read out of the warehouse
                    once and committed rather than recomputed per run.
"""
