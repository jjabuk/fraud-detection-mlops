"""Fetch the IEEE-CIS Fraud Detection CSVs into ``kaggle/raw/``.

    uv run python kaggle/download.py

Needs a Kaggle account that has accepted the competition rules, and credentials in
``~/.kaggle/kaggle.json`` or in ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.

``kagglehub`` keeps its own cache, so the files are copied out of it into
``kaggle/raw/`` — a fixed path the notebook and the scripts can rely on, and one
that mirrors the ``../input/`` layout a Kaggle kernel sees.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
DEST = Path(__file__).resolve().parent / "raw"
WANTED = ("train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv")


def main() -> int:
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed — `uv sync --extra data`", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    missing = [name for name in WANTED if not (DEST / name).exists()]
    if not missing:
        print(f"already present in {DEST}: {', '.join(WANTED)}")
        return 0

    print(f"downloading {COMPETITION} (~600 MB on first run)…")
    cache = Path(kagglehub.competition_download(COMPETITION))

    for name in missing:
        found = next(cache.rglob(name), None)
        if found is None:
            print(f"{name} not found under {cache}", file=sys.stderr)
            return 1
        shutil.copy2(found, DEST / name)
        print(f"  {name}  {(DEST / name).stat().st_size / 1e6:.0f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
