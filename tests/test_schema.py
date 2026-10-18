import json

import pytest

from skier.paths import configs_dir
from skier.schema import load_schema

COCO_SHARED = {5: 6, 6: 2, 7: 7, 8: 3, 11: 13, 12: 10, 13: 14, 14: 11, 15: 15, 16: 12}


def test_names_and_groups(schema):
    assert schema.num_keypoints == 24
    assert len(set(schema.names)) == 24
    assert {g: len(m) for g, m in schema.groups.items()} == {"body": 14, "ski": 8, "pole": 2}
    assert sorted(i for m in schema.groups.values() for i in m) == list(range(24))
    assert schema.names[16] == "ski_tip_right"
    assert schema.names[9] == "pole_basket_left"
    assert schema.group_of(5) == "pole"


def test_flip(schema):
    f = schema.flip_index
    assert all(f[f[i]] == i for i in range(24))
    for r, l in schema.flip_pairs:
        assert schema.side[r] == "right" and schema.side[l] == "left"
        assert schema.names[r].replace("right", "") == schema.names[l].replace("left", "")
    assert f[0] == 0 and f[1] == 1
    assert len(schema.flip_pairs) == 11


def test_edges(schema):
    assert len(schema.edges) == 22
    counts = {g: sum(e.group == g for e in schema.edges) for g in ("body", "ski", "pole")}
    assert counts == {"body": 14, "ski": 6, "pole": 2}
    assert all(0 <= e.a < 24 and 0 <= e.b < 24 for e in schema.edges)


def test_coco_maps(schema):
    assert schema.coco_shared == COCO_SHARED
    assert schema.coco_approx == {9: 8, 10: 4}


def _corrupt(tmp_path, mutate):
    raw = json.loads((configs_dir() / "ski24_schema.json").read_text())
    mutate(raw)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw))
    return p


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["names"].__setitem__(1, "head"),  # duplicate name
        lambda r: r["flip_pairs"].__setitem__(0, [2, 3]),  # right -> right
        lambda r: r["edges"].append({"a": 0, "b": 24, "group": "body"}),  # bad index
        lambda r: r["groups"]["pole"].append(0),  # groups overlap
        lambda r: r["coco_shared"].__setitem__("5", 16),  # COCO target not a body joint
    ],
    ids=["dup_name", "right_right_pair", "edge_24", "overlap", "coco_to_ski"],
)
def test_corrupt_schema_raises(tmp_path, mutate):
    with pytest.raises(ValueError):
        load_schema(_corrupt(tmp_path, mutate))
