"""YAML load + deep merge. Later files override earlier ones."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from tsl.constants import REQUIRED_CONFIG_SECTIONS


class ConfigError(ValueError):
    pass


def _deep_merge(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> MutableMapping[str, Any]:
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
    required = sections if sections is not None else REQUIRED_CONFIG_SECTIONS
    missing = [s for s in required if s not in cfg]
    if missing:
        raise ConfigError(f"Config missing required section(s): {', '.join(missing)}")


def load_config(
    *paths: str | Path,
    validate: bool = True,
    required_sections: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge one or more YAML files.

    Each file may list ``includes``; those are merged first, depth-first.
    ``includes`` itself is dropped from the dict you get back.
    """
    if not paths:
        raise ConfigError("load_config requires at least one config path")

    merged: dict[str, Any] = {}
    for path in paths:
        _merge_file(merged, Path(path), seen=set())

    merged.pop("includes", None)

    if validate:
        validate_config(merged, sections=required_sections)
    return merged


def _merge_file(
    merged: MutableMapping[str, Any],
    path: Path,
    *,
    seen: set[Path],
) -> None:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular config include detected: {path}")
    seen.add(path)

    data = load_yaml(path)
    includes = data.pop("includes", None) or []
    if not isinstance(includes, list):
        raise ConfigError(f"'includes' must be a list in {path}")

    base_dir = path.parent
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            # try CWD first so experiment YAMLs can include configs/... from the repo root
            cwd_candidate = Path.cwd() / inc_path
            file_candidate = base_dir / inc_path
            if cwd_candidate.is_file():
                inc_path = cwd_candidate
            elif file_candidate.is_file():
                inc_path = file_candidate
            else:
                inc_path = cwd_candidate
        _merge_file(merged, inc_path, seen=seen)

    _deep_merge(merged, data)


def resolve_config(
    base: str | Path = "configs/default.yaml",
    overrides: Sequence[str | Path] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    paths: list[str | Path] = [base]
    if overrides:
        paths.extend(overrides)
    return load_config(*paths, validate=validate)
