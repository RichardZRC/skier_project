import numpy as np
import synth

from skier.data import audit, overlays
from skier.io_utils import read_csv
from skier.viz import COLOR, contact_sheet, draw_dashed_line, draw_points, marker_radius, render_gt_overlay


def _blank(w=200, h=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_point_styles_and_colors(schema):
    img = _blank()
    xy = np.full((24, 2), -100.0)
    reliable = np.ones(24, dtype=bool)
    xy[2] = (50, 50)  # shoulder_right, reliable
    xy[6] = (150, 50)  # shoulder_left, occluded guess
    reliable[6] = False
    xy[0] = (100, 20)  # head, reliable
    draw_points(img, xy, reliable, schema)  # points only: no edge can cover a hollow center
    r = marker_radius(img)
    assert tuple(img[50, 50]) == COLOR["right"]
    assert tuple(img[50, 150]) == (0, 0, 0)  # hollow center stays background
    assert img[50, 150 + r].any()  # ring is drawn
    assert tuple(img[20, 100]) == COLOR["center"]


def test_overlay_uses_real_image_size(schema):
    ann = np.full((24, 3), np.nan)
    ann[2] = (0.25, 0.5, 1)
    ann[6] = (0.75, 0.5, 1)
    out = render_gt_overlay(_blank(200, 100), ann, schema, header_lines=[])
    assert tuple(out[50, 50]) == COLOR["right"]
    assert tuple(out[50, 150]) == COLOR["left"]


def test_dashed_line_has_gaps():
    img = _blank()
    draw_dashed_line(img, (0, 50), (199, 50), (255, 255, 255), 1)
    row = img[50, :, 0]
    assert (row == 255).any() and (row == 0).any()


def test_contact_sheet_shape():
    sheet = contact_sheet([_blank(1920, 1080)] * 7, cols=6, thumb_w=320)
    assert sheet.shape == (2 * 180, 6 * 320, 3)


def _inventory_rows():
    rows = []
    for v in ("v0", "v1"):
        for s in ("s0", "s1", "s2"):
            for i in range(4):
                rows.append({"sample_id": f"{v}/{s}/i{i}", "video_id": v, "segment_id": s,
                             "width": "64", "n_reliable_body": "14", "view_angle": ""})
    rows.append({"sample_id": "v0/s0/bad", "video_id": "v0", "segment_id": "s0", "width": "", "n_reliable_body": ""})
    return rows


def test_selection_deterministic_and_spread():
    rows = _inventory_rows()
    a = overlays.select_overlay_samples(rows, ["v0"], 5, seed=230)
    assert a == overlays.select_overlay_samples(rows, ["v0"], 5, seed=230)
    assert len(a) == 5 and all(s.startswith("v0/") for s in a)
    assert "v0/s0/bad" not in a
    assert {s.split("/")[1] for s in a} == {"s0", "s1", "s2"}  # every group appears
    assert len(overlays.select_overlay_samples(rows, ["v0"], 100, seed=1)) == 12


def test_overlay_cli_end_to_end(tmp_path):
    labels = synth.make_labels(videos=2, segments=2, images=3)
    images_root = tmp_path / "images"
    template = "{video_id}/{segment_id}/{image_id}.png"
    synth.make_blank_images(images_root, labels, size=(320, 180), template=template, value=60)
    labels_path = synth.write_labels(labels, tmp_path / "labels.json")
    inv = tmp_path / "inventory.csv"
    audit.main([str(labels_path), "--images-root", str(images_root), "--path-template", template,
                "--inventory", str(inv), "--out", str(tmp_path / "audit")])
    pilots = tmp_path / "pilot_videos.txt"
    pilots.write_text("# pilots\nv0\n")
    out = tmp_path / "overlays"
    assert overlays.main(["--inventory", str(inv), "--labels", str(labels_path), "--images-root", str(images_root),
                          "--videos-file", str(pilots), "--n", "4", "--out", str(out)]) == 0
    ids = (out / "overlay_ids.txt").read_text().split()
    assert len(ids) == 4 and all(s.startswith("v0/") for s in ids)
    assert len(list(out.glob("v0__*.png"))) == 4
    assert (out / "contact_sheet.png").exists()
    assert len(read_csv(inv)) == 12
