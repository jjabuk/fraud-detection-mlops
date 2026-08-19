import tomllib
from pathlib import Path
from typing import Any

# Two levels up from src/fraud_detection/config.py, then out of `src/`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def resolve_repo_path(relative: str | Path) -> Path:
    """Find a repo-relative data file from wherever the caller happens to be running.

    Three consumers run from three different working directories:

    * the **container** runs with `WORKDIR /app` and `config/`, `references/`, `schemas/`
      copied in beside the package, so a plain relative path resolves — the Dockerfile
      says so explicitly and that behaviour has to keep working;
    * the **pipeline and tests** run from the repository root, where it also resolves;
    * an **analysis** runs from `analysis/` or `analysis/notebooks/`, where it does not:
      `load_admission_rules()` raises "no admission file at config/feature-admission.toml"
      with the file sitting one or two directories up.

    So: honour the working directory first, because that is the container's contract, and
    fall back to the package's own location, which is where a notebook needs it. Anchoring
    only on `__file__` would have broken the container, where the package lives under
    `/app/.venv/...` and the data files do not.
    """
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    anchored = PROJECT_ROOT / candidate
    return anchored if anchored.exists() else candidate

def _load_toml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)

# Load configurations once at module import
_feature_admission_config = _load_toml("feature-admission.toml")
_training_config = _load_toml("training.toml")
_orchestration_config = _load_toml("orchestration.toml")

# Expose configuration dictionaries
def get_feature_admission_config() -> dict[str, Any]:
    return _feature_admission_config

def get_training_config() -> dict[str, Any]:
    return _training_config

def get_orchestration_config() -> dict[str, Any]:
    return _orchestration_config

# Helper to get specific training sections
def get_training_params(section: str = "training") -> dict[str, Any]:
    keys = section.split(".")
    val = _training_config
    for key in keys:
        val = val.get(key, {})
    return val

# Helper to get specific orchestration sections
def get_orchestration_params(section: str) -> dict[str, Any]:
    keys = section.split(".")
    val = _orchestration_config
    for key in keys:
        val = val.get(key, {})
    return val
