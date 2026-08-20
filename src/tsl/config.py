"""YAML config loading and deep-merge for Transformer Surgery Lab.

Design goals:
- load a base config and one or more override configs
- deep-merge nested dicts (later overrides win)
- lightly validate required top-level sections
- return a plain dict that is easy to save back out

This intentionally avoids frameworks so it stays whiteboard-explainable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from tsl.constants import REQUIRED_CONFIG_SECTIONS


class ConfigError(ValueError):
    """Raised when a config file is missing, invalid, or incomplete."""


def _deep_merge(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Recursively merge *override* into *base* (in place)."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dict."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a mapping to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            dict(data),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def validate_config(cfg: Mapping[str, Any], sections: Sequence[str] | None = None) -> None:
    """Ensure required top-level sections exist."""
    required = sections if sections is not None else REQUIRED_CONFIG_SECTIONS
    missing = [s for s in required if s not in cfg]
    if missing:
        raise ConfigError(f"Config missing required section(s): {', '.join(missing)}")


def load_config(
    *paths: str | Path,
    validate: bool = True,
    required_sections: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load and deep-merge one or more YAML configs.

    Earlier files are the base; later files override. Example::

        cfg = load_config("configs/default.yaml", "configs/data/tinystories.yaml")
    """
    if not paths:
        raise ConfigError("load_config requires at least one config path")

    merged: dict[str, Any] = {}
    for path in paths:
        override = load_yaml(path)
        _deep_merge(merged, override)

    if validate:
        validate_config(merged, sections=required_sections)
    return merged


def resolve_config(
    base: str | Path = "configs/default.yaml",
    overrides: Sequence[str | Path] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper: merge *base* with optional override paths."""
    paths: list[str | Path] = [base]
    if overrides:
        paths.extend(overrides)
    return load_config(*paths, validate=validate)
