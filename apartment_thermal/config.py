"""Versioned runtime configuration loading."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path


def load_config(path: Path | None = None) -> dict[str, object]:
    text = (
        path.read_text()
        if path is not None
        else files("apartment_thermal").joinpath("default_config.json").read_text()
    )
    config = json.loads(text)
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported configuration schema: {config.get('schema_version')}")
    return config
