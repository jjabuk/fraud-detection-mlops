"""The serving container's HTTP surface. Kept out of the package root's imports on
purpose: importing `fraud_detection.serving` pulls in FastAPI, and the batch job has no
use for it."""
