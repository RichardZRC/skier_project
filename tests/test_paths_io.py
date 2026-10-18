import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from skier import paths
from skier.io_utils import (
    read_csv,
    read_id_list,
    read_json,
    read_json_with_duplicates,
    sha256_file,
    write_csv,
    write_json,
)


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIER_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("SKIER_RESULTS", str(tmp_path / "r"))
    assert paths.data_root() == tmp_path / "d"
    assert paths.results_root() == tmp_path / "r"


def test_defaults_under_home(monkeypatch):
    monkeypatch.delenv("SKIER_DATA", raising=False)
    monkeypatch.delenv("SKIER_RESULTS", raising=False)
    assert paths.data_root() == Path.home() / "skier_data"
    assert paths.results_root() == Path.home() / "skier_results"


def test_repo_root():
    assert (paths.repo_root() / "pyproject.toml").exists()
    assert (paths.configs_dir() / "ski24_schema.json").exists()


def test_json_round_trip(tmp_path):
    obj = {"b": [1, 2.5, None], "a": {"x": "y"}, "n": np.int64(3), "f": np.float32(0.5), "arr": np.arange(3)}
    write_json(obj, tmp_path / "sub" / "o.json")
    assert read_json(tmp_path / "sub" / "o.json") == {"a": {"x": "y"}, "b": [1, 2.5, None], "n": 3, "f": 0.5, "arr": [0, 1, 2]}


def test_csv_round_trip_and_none(tmp_path):
    write_csv([{"a": 1, "b": None}, {"a": "x"}], tmp_path / "t.csv", ["a", "b"])
    assert read_csv(tmp_path / "t.csv") == [{"a": "1", "b": ""}, {"a": "x", "b": ""}]


def test_csv_unknown_column_raises(tmp_path):
    with pytest.raises(ValueError):
        write_csv([{"a": 1, "typo": 2}], tmp_path / "t.csv", ["a"])


def test_duplicate_keys_reported(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"a": 1, "a": 2, "b": {"c": 1, "c": 3}}')
    obj, dupes = read_json_with_duplicates(p)
    assert obj == {"a": 2, "b": {"c": 3}}
    assert sorted(dupes) == ["a", "c"]


def test_read_id_list(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# pilot videos\nv1\n\n  v2  # comment\n")
    assert read_id_list(p) == ["v1", "v2"]


def test_sha256(tmp_path):
    data = b"skier" * 1000
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert sha256_file(p, chunk=7) == hashlib.sha256(data).hexdigest()
