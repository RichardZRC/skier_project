"""Small file helpers shared by every tool: JSON, CSV, ID lists, hashing."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` with ``indent=2, sort_keys=True``. Dict keys must all be strings."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def read_json_with_duplicates(path: str | Path) -> tuple[Any, list[str]]:
    """Parse JSON and also return every key that was repeated inside one object.

    Python's ``json`` silently keeps the last value of a repeated key; this keeps
    that behaviour but reports the repeats so an audit can flag them.
    """
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    with Path(path).open(encoding="utf-8") as f:
        obj = json.load(f, object_pairs_hook=hook)
    return obj, duplicates


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    return str(value)


def write_csv(rows: Iterable[dict], path: str | Path, columns: list[str]) -> None:
    """Write rows with a header. Missing or ``None`` values become empty cells.

    A row with a key that is not in ``columns`` raises, so typos don't vanish silently.
    """
    path = Path(path)
    ensure_dir(path.parent)
    allowed = set(columns)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            extra = set(row) - allowed
            if extra:
                raise ValueError(f"row has unknown columns {sorted(extra)}")
            writer.writerow([_cell(row.get(c)) for c in columns])


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_id_list(path: str | Path) -> list[str]:
    """One ID per line; ``#`` starts a comment; blank lines are ignored."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()
