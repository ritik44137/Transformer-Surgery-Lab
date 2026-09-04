"""Config loading, deep-merge, validation, and override precedence."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tsl.config import ConfigError, load_config, load_yaml, validate_config


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_yaml_roundtrip(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "cfg.yaml", {"run": {"name": "x"}, "train": {"max_steps": 3}})
    data = load_yaml(path)
    assert data["run"]["name"] == "x"
    assert data["train"]["max_steps"] == 3


def test_load_yaml_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "missing.yaml")


def test_deep_merge_and_override_precedence(tmp_path: Path) -> None:
    base = _write_yaml(
        tmp_path / "base.yaml",
        {
            "run": {"name": "base", "seed": 1},
            "data": {"seq_len": 64},
            "model": {"hidden_size": 32, "norm": "layernorm"},
            "train": {"max_steps": 10, "batch_size": 4},
        },
    )
    override = _write_yaml(
        tmp_path / "override.yaml",
        {
            "run": {"name": "override"},
            "model": {"norm": "rmsnorm"},
            "train": {"max_steps": 2},
        },
    )
    cfg = load_config(base, override)
    assert cfg["run"]["name"] == "override"
    assert cfg["run"]["seed"] == 1
    assert cfg["model"]["hidden_size"] == 32
    assert cfg["model"]["norm"] == "rmsnorm"
    assert cfg["train"]["max_steps"] == 2
    assert cfg["train"]["batch_size"] == 4


def test_includes_merge(tmp_path: Path) -> None:
    fragment = _write_yaml(
        tmp_path / "frag.yaml",
        {"model": {"hidden_size": 16}, "train": {"max_steps": 1}},
    )
    # Relative include resolved from CWD; write under tmp and pass absolute paths.
    main = _write_yaml(
        tmp_path / "main.yaml",
        {
            "includes": [str(fragment)],
            "run": {"name": "inc"},
            "data": {"seq_len": 8},
            "model": {"num_layers": 2},
            "train": {"batch_size": 2},
        },
    )
    cfg = load_config(main)
    assert cfg["model"]["hidden_size"] == 16
    assert cfg["model"]["num_layers"] == 2
    assert cfg["train"]["max_steps"] == 1
    assert "includes" not in cfg


def test_validate_required_sections() -> None:
    with pytest.raises(ConfigError, match="missing required"):
        validate_config({"run": {}, "data": {}})


def test_load_config_validate_flag(tmp_path: Path) -> None:
    incomplete = _write_yaml(tmp_path / "bad.yaml", {"run": {"name": "x"}})
    with pytest.raises(ConfigError, match="missing required"):
        load_config(incomplete, validate=True)
    cfg = load_config(incomplete, validate=False)
    assert cfg["run"]["name"] == "x"
