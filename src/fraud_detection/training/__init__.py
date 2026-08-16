"""Pure training logic: no BigQuery, no Dagster, no cloud clients.

Everything here takes arrays and returns numbers, so the parts of the model
pipeline that are easy to get subtly wrong -- calibration, threshold choice,
the cost model -- are unit-testable against hand-built inputs rather than
only observable after a full training run.
"""
