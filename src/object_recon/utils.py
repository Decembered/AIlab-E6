"""Shared utilities for object reconstruction pipeline."""
import json
import time
from pathlib import Path
from typing import Optional


def get_timestamp() -> str:
    """Return ISO timestamp for logging."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(filepath: Path, metrics: dict) -> None:
    """Save metrics dict as JSON."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_config(config_path: Path) -> dict:
    """Load a YAML config file. Falls back to JSON if yaml not available."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        import json as _json
        with open(config_path) as f:
            return _json.load(f)
