"""
T0.2 — Config loader.
Loads YAML config for the active environment (YAAZHI_ENV=local|azure).
All module code must import settings from here; never hardcode values.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR = "YAAZHI_ENV"
_VALID_ENVS = ("local", "azure")
_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_raw(env: str) -> dict[str, Any]:
    base_path = _CONFIG_DIR / "local.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with base_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    if env != "local":
        override_path = _CONFIG_DIR / f"{env}.yaml"
        if override_path.exists():
            with override_path.open() as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, override)

    return cfg


class _Namespace:
    """Dot-access wrapper around a plain dict, with missing-key error."""

    def __init__(self, data: dict, path: str = "") -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        path = object.__getattribute__(self, "_path")
        if key not in data:
            full = f"{path}.{key}" if path else key
            raise KeyError(
                f"Required config key '{full}' is missing. "
                f"Add it to configs/local.yaml."
            )
        val = data[key]
        if isinstance(val, dict):
            sub_path = f"{path}.{key}" if path else key
            return _Namespace(val, sub_path)
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Config is read-only.")

    def get(self, key: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        val = data.get(key, default)
        if isinstance(val, dict):
            return _Namespace(val)
        return val

    def as_dict(self) -> dict:
        return dict(object.__getattribute__(self, "_data"))


def load_config(env: str | None = None) -> _Namespace:
    """
    Load config for the given environment.
    Falls back to YAAZHI_ENV env var, then 'local'.
    Raises KeyError on missing required fields when accessed.
    """
    if env is None:
        env = os.environ.get(_ENV_VAR, "local").lower()
    if env not in _VALID_ENVS:
        raise ValueError(f"Unknown YAAZHI_ENV '{env}'. Must be one of {_VALID_ENVS}.")
    raw = _load_raw(env)
    return _Namespace(raw)


# Module-level singleton — import this in all other modules.
settings = load_config()
