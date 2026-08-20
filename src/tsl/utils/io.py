"""Small file I/O helpers for configs and experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write *data* as pretty JSON. Creates parent directories."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    return path


def append_jsonl(record: Mapping[str, Any], path: str | Path) -> Path:
    """Append one JSON object as a line to a JSONL file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Overwrite a JSONL file with *records*."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return path


def read_yaml(path: str | Path) -> Any:
    """Read a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: Any, path: str | Path) -> Path:
    """Write *data* as YAML. Creates parent directories."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return path


def run_dir(output_dir: str | Path, run_name: str) -> Path:
    """Return ``output_dir / run_name``, creating it if needed."""
    return ensure_dir(Path(output_dir) / run_name)
