"""What is recorded about a model, and what lets one be promoted.

The promotion marker and the provenance record are the two artefacts that
survive a training run and are read by something that was not part of it: the
scoring job checks the marker, and an auditor reads the provenance. Both are
plain files with no cloud SDK behind them, so they can be written by a Dagster
asset and read by a notebook without either importing the other.
"""

from fraud_detection.registry.promotion import *
from fraud_detection.registry.provenance import *
