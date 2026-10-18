# Step 1 code spec: scaffold and offline scripts (1A + 1B)

**Owner:** Ruichao Zhang — ruichaoz@stanford.edu
**Spec version:** v1, October 17, 2026
**Companions:** [Step 1 guide](skier_step_1_guide.md) (run order, AWS, data, review) · [Six-week plan](skier_implementation_plan.md) · [PROJECT_NOTES.md](PROJECT_NOTES.md)

This is the single source of truth for **how** to write the Step 1 code. The guide says **when** and **on what data** to run it. Everything here is `[CODE]`: it runs in the light dev environment (§2.3) with synthetic inputs, without the dataset, a GPU or internet after the one-time install.

---

## 0. How to use this spec

- Implement §2 → §11 in order. Each section gives files, signatures, behaviour, tests and a done condition.
- A function is done when its tests pass. A script is done when its tests pass and `--help` works in the light environment.
- Facts only the real data can answer live in marked `FILL IN` blocks (§13). Don't guess them. Anything else that turns out wrong goes in `docs/decisions.md` with the fix.

---

## 1. Conventions (all Step 1 code)

- **Logic in the package, scripts thin.** Each tool lives in a module under `src/skier/` that exposes `build_parser() -> argparse.ArgumentParser` and `main(argv: list[str] | None = None) -> int`. `scripts/<tool>.py` is only:

  ```python
  from skier.data.audit import main

  if __name__ == "__main__":
      raise SystemExit(main())
  ```

  Tests call `main([...])` directly. No `sys.path` hacks: scripts rely on `pip install -e .`.
- **Heavy imports are lazy.** `torch`, `mmcv`, `mmengine`, `mmpose`, `siamfc` and `got10k` are imported only inside the functions that need them. At module top level, use only the standard library, `numpy`, `cv2`, `PIL` and `yaml`.
- **Outputs.** Every tool takes `--out DIR`, creates it with `mkdir(parents=True, exist_ok=True)`, overwrites its own files on rerun, and never modifies its inputs. Default output directories, under `results_root() / "step1"`:

  | Tool | Default `--out` |
  | --- | --- |
  | audit | `audit/` |
  | overlays | `overlays/` |
  | alignment | `alignment/<safe video_id>__<safe segment_id>/` |
  | env check | `env/` |
  | pose smoke test | `smoke_pose/` |
  | tracker smoke test | `smoke_tracker/` |

  The single exception is the audit's inventory manifest, which goes to the repo via `--inventory` (§6).
- **What reaches git.** Only small text outputs (`.json`, `.csv`, `.txt`) are synced into `results/step1/` (guide 1I). Never write label coordinates into those files. Images, videos and `.npz` files stay under `$SKIER_RESULTS`.
- **IDs.** Treat every JSON key as `str`; never cast to int. `sample_id = f"{video_id}/{segment_id}/{image_id}"`. `safe_name(s)` replaces `/` with `__` for filenames.
- **Coordinates.** Labels are normalized: `x_px = x * W`, `y_px = y * H`, where W and H come from the image or frame being drawn on. Boxes are `xyxy` float arrays.
- **Randomness.** `--seed` defaults to 230. Use a local `random.Random(seed)` or `np.random.default_rng(seed)`; library code never seeds globally.
- **Files.** CSV through the stdlib `csv` module with a header row; empty cell for unknown values. JSON with `indent=2, sort_keys=True`. No pandas.
- **Drawing.** Markers and lines use `cv2.LINE_8`, so pixel colors are exact and testable. Text may use `cv2.LINE_AA`. Colors are BGR.
- **Exit codes.** 0 = ran (even if it found data problems); 1 = could not run (bad structure, a failed strict check); 2 = usage error (argparse).
- **Style.** Python 3.10, `pathlib` everywhere (the coding machine may be Windows), type hints on public functions.

---

## 2. 1A — Repo scaffold

### 2.1 Tree

```text
.gitignore  pyproject.toml  requirements-pose.txt
configs/ski24_schema.json  configs/pose/.gitkeep  configs/replay/.gitkeep
data/manifests/.gitkeep
docs/scope.md  docs/decisions.md
results/.gitkeep
scripts/audit_pose_labels.py  draw_gt_overlays.py  check_alignment.py
        check_env.py  smoke_pose.py  smoke_tracker.py
src/skier/__init__.py  paths.py  io_utils.py  schema.py  geometry.py  viz.py  envcheck.py
src/skier/data/__init__.py  ski2dpose_raw.py  audit.py  overlays.py  alignment.py
src/skier/pose/__init__.py  smoke.py
src/skier/tracking/__init__.py  smoke.py
src/skier/replay/__init__.py
src/skier/eval/__init__.py
tests/conftest.py  tests/synth.py  tests/test_*.py  tests/fixtures/.gitkeep
```

### 2.2 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "skier"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy<2", "opencv-python", "pillow", "pyyaml"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "stack: needs the pinned torch/mmcv/mmpose stack (skipped otherwise)",
    "gpu: needs a CUDA GPU (skipped otherwise)",
]
```

torch, mmcv and mmpose are deliberately **not** dependencies; they come from special index URLs (guide 1C.7). Use `opencv-python`, never `opencv-python-headless`, so that the package matches what mmcv installs.

### 2.3 Light dev environment (one-time `[NET]`)

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

On the AWS instance, the pinned environment (`~/venvs/skier`) gets the same `pip install -e ".[dev]"`. The same tests run there, plus any `stack`/`gpu` tests.

If pip can't reach PyPI (for example behind a corporate proxy), install from the package mirror that machine is already configured for. With `uv` and `UV_INDEX_URL` set: `uv venv --python 3.10 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"`.

### 2.4 `requirements-pose.txt`

A record of the pins, not a one-shot install file: the install order and index URLs are in guide 1C.7.

```text
# Pinned GPU stack. Install in the order of guide 1C.7.
# --index-url https://download.pytorch.org/whl/cu121
torch==2.1.2
torchvision==0.16.2
# -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
mmcv==2.1.0
mmengine>=0.10,<0.11
mmpose==1.3.2
numpy<2
openmim
got10k
gdown
```

### 2.5 `.gitignore`

```gitignore
# large or machine-specific
data/raw/
data/processed/
*.pth
*.pt
*.mp4
*.avi
*.zip
__pycache__/
*.egg-info/
.venv/
.pytest_cache/
CLAUDE.local.md

# Never commit dataset images or labels (Ski 2DPose terms; SkiTB is CC BY-NC).
# Overlays, contact sheets, clips and predictions stay under $SKIER_RESULTS.
```

### 2.6 `src/skier/paths.py`

```python
def data_root() -> Path      # $SKIER_DATA, else ~/skier_data; expanduser; need not exist
def results_root() -> Path   # $SKIER_RESULTS, else ~/skier_results
def repo_root() -> Path      # Path(__file__).resolve().parents[2]; raises if no pyproject.toml there
def configs_dir() -> Path    # repo_root() / "configs"
```

`repo_root()` only works for the editable install; that's the only supported install.

Tests (`test_paths_io.py`):
- `SKIER_DATA` / `SKIER_RESULTS` overrides via `monkeypatch.setenv`.
- Defaults fall under `Path.home()` when the variables are unset.
- `repo_root() / "pyproject.toml"` exists.

### 2.7 `src/skier/io_utils.py`

```python
def ensure_dir(path) -> Path
def write_json(obj, path) -> None                      # indent=2, sort_keys=True, UTF-8
def read_json(path) -> Any
def read_json_with_duplicates(path) -> tuple[Any, list[str]]   # object_pairs_hook records repeated keys
def write_csv(rows: list[dict], path, columns: list[str]) -> None   # header; missing/None → ""
def read_csv(path) -> list[dict[str, str]]
def read_id_list(path) -> list[str]                    # one ID per line; '#' comments; blanks skipped
def sha256_file(path, chunk=1 << 20) -> str
```

Tests (also `test_paths_io.py`):
- JSON and CSV round-trips.
- `None` is written as an empty cell.
- A JSON text with `{"a": 1, "a": 2}` reports `["a"]`.
- `sha256_file` matches `hashlib.sha256(data).hexdigest()` on a small temp file.

### 2.8 Docs seeds

- `docs/scope.md`: copy section 1 of the main plan.
- `docs/decisions.md`: one entry per decision D1–D9 from main plan §2, in this format:

  ```markdown
  ## D1 — Learned component (2026-10-17)
  - Choice: pose fine-tuning (option A); tracker fixed.
  - Reason: …
  - Effect on evaluation: …
  ```

**Done when:** `pip install -e ".[dev]"` and `pytest` both run in a fresh clone.

---

## 3. 1B.1 — Landmark schema: `configs/ski24_schema.json` + `src/skier/schema.py`

Zero-based order from the official dataset page (verified 2026-10-17). "Right" means the skier's own right, not the right side of the image.

| Index | Name | Group | Side | Flip partner |
| ---: | --- | --- | --- | ---: |
| 0 | head | body | center | 0 |
| 1 | neck | body | center | 1 |
| 2 | shoulder_right | body | right | 6 |
| 3 | elbow_right | body | right | 7 |
| 4 | hand_right | body | right | 8 |
| 5 | pole_basket_right | pole | right | 9 |
| 6 | shoulder_left | body | left | 2 |
| 7 | elbow_left | body | left | 3 |
| 8 | hand_left | body | left | 4 |
| 9 | pole_basket_left | pole | left | 5 |
| 10 | hip_right | body | right | 13 |
| 11 | knee_right | body | right | 14 |
| 12 | ankle_right | body | right | 15 |
| 13 | hip_left | body | left | 10 |
| 14 | knee_left | body | left | 11 |
| 15 | ankle_left | body | left | 12 |
| 16 | ski_tip_right | ski | right | 20 |
| 17 | toes_right | ski | right | 21 |
| 18 | heel_right | ski | right | 22 |
| 19 | ski_tail_right | ski | right | 23 |
| 20 | ski_tip_left | ski | left | 16 |
| 21 | toes_left | ski | left | 17 |
| 22 | heel_left | ski | left | 18 |
| 23 | ski_tail_left | ski | left | 19 |

JSON layout (JSON object keys are strings; the loader converts them to `int`):

```json
{
  "version": 1,
  "names": ["head", "neck", "shoulder_right", "...24 total"],
  "groups": {"body": [0,1,2,3,4,6,7,8,10,11,12,13,14,15],
             "ski": [16,17,18,19,20,21,22,23], "pole": [5,9]},
  "side": ["center", "center", "right", "...24 total"],
  "flip_pairs": [[2,6],[3,7],[4,8],[5,9],[10,13],[11,14],[12,15],[16,20],[17,21],[18,22],[19,23]],
  "edges": [{"a": 0, "b": 1, "group": "body"}, "..."],
  "coco_shared": {"5": 6, "6": 2, "7": 7, "8": 3, "11": 13, "12": 10, "13": 14, "14": 11, "15": 15, "16": 12},
  "coco_approx": {"9": 8, "10": 4}
}
```

Edges (22 in total):
- Body (14): 0–1, 1–2, 1–6, 2–3, 3–4, 6–7, 7–8, 2–10, 6–13, 10–13, 10–11, 11–12, 13–14, 14–15.
- Ski (6): 16–17, 17–18, 18–19, 20–21, 21–22, 22–23.
- Pole (2): 4–5, 8–9.

`coco_shared` maps COCO index → Ski index; `coco_approx` maps COCO wrist → Ski hand and is reported separately.

```python
@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    group: str            # "body" | "ski" | "pole"

@dataclass(frozen=True)
class Schema:
    names: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    side: tuple[str, ...]
    flip_pairs: tuple[tuple[int, int], ...]   # (right, left)
    flip_index: tuple[int, ...]               # partner of i; i itself for center joints
    edges: tuple[Edge, ...]
    coco_shared: dict[int, int]
    coco_approx: dict[int, int]
    @property
    def num_keypoints(self) -> int: ...       # 24
    def group_of(self, i: int) -> str: ...

def load_schema(path: Path | None = None) -> Schema   # default configs_dir() / "ski24_schema.json"
```

`load_schema` validates the file and raises `ValueError` with a specific message if:
- names are not unique;
- the groups don't partition 0–23;
- a flip pair isn't (right, left) by `side`;
- a center joint isn't its own partner;
- an edge has an invalid index or group;
- a `coco_*` target isn't a body joint.

Tests (`test_schema.py`):
- The committed JSON loads; 24 unique names; group sizes 14 / 8 / 2.
- `flip_index` is an involution (`flip_index[flip_index[i]] == i`), every pair goes right → left, and center joints map to themselves.
- 22 edges with valid indices; 14 / 6 / 2 per group.
- `coco_shared` equals the table above (10 entries); `coco_approx` has 2.
- Spot checks: `names[16] == "ski_tip_right"`, `names[9] == "pole_basket_left"`.
- Corrupted copies in `tmp_path` raise `ValueError`: a duplicate name, a right → right flip pair, an edge to index 24.

---

## 4. Shared — raw label reader: `src/skier/data/ski2dpose_raw.py`

Labels JSON nesting (verified): `VideoID → SplitID → ImageID → {"annotation": [[x, y, visible] × 24], "frame_idx": i}`. Their "SplitID" is a camera sequence; we call it `segment_id`.

```python
# --- FILL IN AT 1E (code spec §13) ---------------------------------------
IMAGE_PATH_TEMPLATE: str | None = None    # e.g. "{video_id}/{segment_id}/{image_id}.jpg"; keys: video_id, segment_id, image_id

def info_for(info: Any, video_id: str, segment_id: str) -> dict:
    """Return {"discipline": str | None, "view_angle": str | None} for one sequence."""
    return {"discipline": None, "view_angle": None}
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RawLabel:
    video_id: str
    segment_id: str
    image_id: str
    annotation: Any        # as found in the JSON; None if the key or entry is missing
    frame_idx: Any         # None if the key is missing
    @property
    def sample_id(self) -> str: ...

def make_sample_id(video_id, segment_id, image_id) -> str
def safe_name(sample_id: str) -> str
def iter_raw_labels(labels: dict) -> Iterator[RawLabel]
def annotation_array(raw: RawLabel) -> np.ndarray | None   # float64 (24, 3), or None if it can't convert to that shape
def reliable_mask(ann: np.ndarray) -> np.ndarray            # bool[24], ann[:, 2] == 1
def to_pixels(xy_norm: np.ndarray, width: int, height: int) -> np.ndarray   # [..., 2]
def image_path_for(images_root: Path, video_id, segment_id, image_id, template: str | None = None) -> Path
def info_preview(info: Any, max_keys: int = 5, depth: int = 4) -> dict      # key names and types only, no values
```

- `iter_raw_labels` yields in sorted-key order at every level, so output is deterministic.
  - A non-dict at the video or segment level raises `ValueError`; the nesting is a verified fact, so a surprise there means something is badly wrong.
  - An image entry that isn't a dict yields `annotation=None, frame_idx=None`, and the audit reports it.
- `image_path_for` uses `template`, else `IMAGE_PATH_TEMPLATE`. If both are `None`, it raises `NotImplementedError("image path template not set: fill in at 1E (code spec §13)")`.
- `info_preview` exists so the audit can show the info JSON's structure in 1E, without guessing it.

Tests (`test_ski2dpose_raw.py`):
- Iteration order on a fixture with shuffled keys.
- `sample_id` and `safe_name` values; a missing `frame_idx` gives `None`.
- `to_pixels` on a 200 × 100 image: (0, 0) → (0, 0); (1, 1) → (200, 100); (0.25, 0.5) → (50, 50).
- `annotation_array` returns `None` for a (23, 3) list.
- `image_path_for` with a template; without one it raises `NotImplementedError` whose message mentions `1E`.
- `info_preview` contains keys but no leaf values.

---

## 5. Shared — reference box: `src/skier/geometry.py`

Step 2.1 adds the rest of this module. Write only these two functions now:

```python
def box_from_points(xy: np.ndarray, mask: np.ndarray) -> np.ndarray
    # tight xyxy box of xy[mask]; raises ValueError if fewer than 2 points

def reference_box(xy_px: np.ndarray, reliable: np.ndarray, schema: Schema) -> np.ndarray | None
    # box of reliable body + ski points (poles excluded); None if < 2 such points or the box diagonal < 1 px
```

The diagonal rule matches how the box is used later: its diagonal is the PCK normalizer. A thin box with a nonzero diagonal is kept.

Tests (in `test_ski2dpose_raw.py`):
- Hand-computed box; one masked point raises an error.
- A far-away pole point does not enlarge the reference box.
- An unreliable point is excluded.
- Two identical points → `None`.

---

## 6. 1B.2 — Label audit: `src/skier/data/audit.py`, `scripts/audit_pose_labels.py`

```text
python scripts/audit_pose_labels.py LABELS_JSON [--info INFO_JSON]
       [--images-root DIR] [--path-template TEMPLATE]
       [--inventory CSV]   (default: repo data/manifests/pose_inventory.csv)
       [--out DIR]
```

Never clamp, fix or drop anything; only report. Each problem below is reported **once** per affected record, as one row of `problems.csv` (`sample_id, problem, detail`). `detail` gives joint indices and counts, never coordinate values.

**Pass 1, structure (always runs):**

| `problem` | Condition | Notes |
| --- | --- | --- |
| `duplicate_key` | A key repeated within one JSON object (from `read_json_with_duplicates`) | `sample_id` empty; detail = key name |
| `missing_annotation` | Entry not a dict, or no `annotation` key | Skip the remaining checks for this record |
| `bad_shape` | `annotation_array` is `None` | Skip the shape-dependent checks |
| `non_finite` | Any NaN or inf | |
| `coord_out_of_range` | Any finite x or y outside [0, 1] | detail = joint indices |
| `bad_visibility` | Any visibility not in {0, 1} (`1.0` counts as 1) | detail = distinct values seen |
| `missing_frame_idx` | No `frame_idx` key | |
| `duplicate_frame_idx` | A non-missing `frame_idx` already used in the same (video, segment) | One row per later record; detail = the earlier `sample_id` |

**Pass 2, images (only with `--images-root`):** resolve paths with `image_path_for(..., template=--path-template)`. If no template is set at all, exit 1 with the fill-in message. Read width and height with `PIL.Image.open(p).size`, which reads only the header.

| `problem` | Condition |
| --- | --- |
| `label_without_image` | Resolved path doesn't exist |
| `unreadable_image` | PIL can't open it |
| `duplicate_image_path` | Two labels resolve to the same file (detail = the other `sample_id`) |
| `image_without_label` | A `.jpg/.jpeg/.png/.webp` under `--images-root` that no label resolves to (`sample_id` = relative path) |

**Outputs:**
- `label_summary.json`:
  - `n_videos`, `n_segments`, `n_images`, `images_per_video`, `segments_per_video`;
  - `visibility_values` (value → count), `reliable_per_joint` (24 counts);
  - `problem_counts`;
  - `expected = {"videos": 16, "segments": 158, "images": 1982}` and `matches_expected`;
  - `info_preview` (if `--info` is given);
  - the arguments used.
- `problems.csv`, sorted by (`problem`, `sample_id`).
- `--inventory` CSV with these columns in this order: `sample_id, video_id, segment_id, image_id, frame_idx, image_path, width, height, n_reliable_body, n_reliable_ski, n_reliable_pole, discipline, view_angle`.
  - One row per label record, including records with problems; unknown values are blank.
  - Sorted by `sample_id`.
  - `image_path` is **relative to `--images-root`**, with `/` separators. The manifest is committed and must not contain machine paths.
  - `discipline` and `view_angle` come from `info_for` and stay blank until it's filled in.
- Print a short summary to stdout.

Tests (`test_audit.py`), using `synth.make_labels` and `synth.make_blank_images` (§12):
- **Fixture:** 2 videos × 2 segments × 3 images = 12 records, with 64 × 48 PNGs at `{video_id}/{segment_id}/{image_id}.png`. Seeded defects, each on a different record:
  - shape (23, 3);
  - one x = 1.2;
  - one NaN;
  - one visibility 2;
  - missing `frame_idx`;
  - a repeated `frame_idx` within a segment;
  - one image file deleted;
  - one stray image with no label.
- Each seeded defect gives **exactly one** row with the expected `problem`; clean records give none; the total row count equals the number of seeded defects.
- The (23, 3) record gets only `bad_shape`.
- Counts are 2 videos, 4 segments, 12 images.
- The inventory has 12 rows sorted by `sample_id`, relative paths, and width/height 64/48 wherever the image exists.
- Without `--images-root`: no image problems and empty width/height.
- A JSON text with a duplicated image key reports `duplicate_key`.
- The labels file's SHA-256 is unchanged after the run.

---

## 7. 1B.3 — Ground-truth overlays: `src/skier/viz.py`, `src/skier/data/overlays.py`, `scripts/draw_gt_overlays.py`

`viz.py` holds only drawing; it is reused later for predictions, so it takes pixel coordinates and a mask, not label records. Sample selection and the CLI live in `data/overlays.py`.

```python
COLOR = {"right": (0, 140, 255), "left": (255, 255, 0), "center": (255, 255, 255)}   # BGR: orange, cyan, white
REF_BOX_COLOR = (0, 255, 255)                                                        # BGR yellow

def marker_radius(img) -> int                       # max(3, round(min(H, W) / 200))
def draw_points(img, xy_px, reliable, schema, label_indices=True) -> np.ndarray
def draw_edges(img, xy_px, reliable, schema) -> np.ndarray
def draw_box(img, xyxy, color, dashed=False, thickness=2) -> np.ndarray
def draw_dashed_line(img, p0, p1, color, thickness, dash=8, gap=6) -> np.ndarray     # dotted = dash 2, gap 4
def draw_header(img, lines: list[str]) -> np.ndarray
def contact_sheet(images: list[np.ndarray], cols=6, thumb_w=320) -> np.ndarray      # 16:9 thumbnails, letterboxed on black
def render_gt_overlay(img_bgr, ann_norm, schema, header_lines) -> np.ndarray         # to_pixels with the image's own size
```

Drawing rules (all functions draw in place and return the image):
- **Point color** comes from `schema.side`: skier's right is orange, left is cyan, head/neck white.
- **Point style:** a reliable point is a filled circle (`thickness=-1`); an occluded guess is a hollow circle (`thickness=2`).
- **Index labels:** each point's index is drawn offset by `(r + 2, -r - 2)` from the point.
- **Edges:** thickness `t = max(1, r // 2)`. Body edges use `t`, ski edges `3t`, pole edges are dotted (sparser, gap 8, when an endpoint is a guess). Any other edge with an unreliable endpoint is drawn dashed. Edge color is the endpoints' shared side color, or white if the sides differ.
- **Skipped points:** non-finite points are skipped, along with their edges.
- **Draw order:** reference box (dashed yellow) → edges → points → header, so points are never covered by a box or edge line. The header shows `sample_id`, `W×H`, discipline and view, with `?` when unknown. If there's no reference box, the header says `no ref box`.

Sample selection, in `select_overlay_samples(rows, pilot_videos, n, seed) -> list[str]`:
1. Keep inventory rows whose `video_id` is a pilot video and whose `width` and `n_reliable_body` are non-blank.
2. Group the rows by (`view_angle`, `video_id`, `segment_id`).
3. With `random.Random(seed)`, shuffle the group order, then the rows within each group.
4. Take rows round-robin across groups until `n` are taken. If fewer than `n` exist, take them all and print a warning.

```text
python scripts/draw_gt_overlays.py --inventory CSV --labels LABELS_JSON --images-root DIR
       --videos-file data/manifests/pilot_videos.txt [--n 30] [--seed 230] [--out DIR]
```

- `--videos-file` lists one `video_id` per line; `#` starts a comment.
- **Outputs:** `<safe_name(sample_id)>.png` at full resolution; `contact_sheet.png`; `overlay_ids.txt` (one `sample_id` per line, in selection order).

Tests (`test_viz.py`):
- **Point colors.** On a 200 × 100 black image, set all 24 points to (−100, −100) except three: index 2 reliable at (50, 50), index 6 unreliable at (150, 50), index 0 reliable at (100, 20). Call **`draw_points` only**, so no edge line can cover a hollow center. Expect pixel (y=50, x=50) to be orange, (50, 150) to still be black, (50, 150 + r) to be non-black (the ring is drawn), and (20, 100) to be white.
- **Real image size.** `render_gt_overlay` with normalized (0.25, 0.5) on a 200 × 100 image puts the marker at pixel (50, 50).
- **Dashed line.** A dashed horizontal line has both colored and black pixels along its row.
- **Contact sheet.** 7 images with `cols=6` give 2 rows, width `6 × 320`.
- **Selection.** The same seed gives the same list. Non-pilot videos are never returned. With `n` ≥ the number of groups, every group appears at least once.

---

## 8. 1B.4 — Video-frame alignment: `src/skier/data/alignment.py`, `scripts/check_alignment.py`

```text
python scripts/check_alignment.py --inventory CSV --images-root DIR --video-id ID --segment-id SEG
       [--videos-root DIR | --video FILE --start-frame N] [--labels LABELS_JSON]
       [--window 5] [--apply-offset 0] [--out DIR]
```

```python
# --- FILL IN AT 1E/1G (code spec §13) ------------------------------------
MIN_SCORE = 0.8      # starting values; adjust on the first real segment and record them
MIN_MARGIN = 0.02

def locate_segment(videos_root: Path, video_id: str, segment_id: str, info: Any = None) -> tuple[Path, int]:
    """Return (video file, decode index of the segment's frame_idx 0)."""
    raise NotImplementedError("video layout unknown: fill in at 1G (code spec §13)")
# --------------------------------------------------------------------------

def to_probe(img_bgr) -> np.ndarray     # grayscale, resize to 160×90 with INTER_AREA, float64, zero mean, unit std
def ncc(a, b) -> float                  # mean(a * b) of two probes; 0.0 if either is constant (no NaN)
def align_segment(video_path, start_frame, labeled, window) -> list[dict]
    # labeled: list of (image_id, frame_idx, image_path)
```

- **Which file and start frame:** `--video` + `--start-frame` override `locate_segment`. That override is how the tests run, and how 1G can work before `locate_segment` is filled in.
- **Decoding:** decode the video **sequentially** once, from frame 0 to the largest index any label needs. Keep probes only for frames inside some label's window `[t − window, t + window]`, where `t = start_frame + frame_idx`. Clip windows at 0 and at end of file; don't trust `CAP_PROP_FRAME_COUNT`.
- **Per labeled image:**
  - `best_decode_idx` = the highest-scoring frame;
  - `offset = best_decode_idx − t`;
  - `margin` = best − second-best score (empty if there was only one candidate);
  - `verified = best_score ≥ MIN_SCORE and margin ≥ MIN_MARGIN`;
  - if every candidate frame lies past end of file, `best_decode_idx` is empty and `verified` is false.
- **Resolution:** video resolution may differ from image resolution. Both are 16:9 and both go through the 160 × 90 probe.
- **Outputs:**
  - `alignment.csv` with `video_id, segment_id, image_id, label_frame_idx, best_decode_idx, offset, best_score, margin, verified`.
  - `alignment_summary.json` with `n_labeled`, `n_verified`, offset counts, and `consistent` (all verified, one offset).
  - `gt_clip.mp4` (only with `--labels`), from frame `min(t) − window` to `max(t) + window`:
    - codec `mp4v`, at the source FPS (25 if the source reports 0), with the frame number drawn in a corner;
    - labels are drawn with `render_gt_overlay` **only** at decode index `t + apply_offset`, using the **frame's** size for `to_pixels`;
    - the default `--apply-offset 0` shows the labels where they are claimed to be; rerun with the measured offset to confirm it.

Tests (`test_alignment.py`), using `synth.make_moving_square_video` (§12):
- **Synthetic video:** 60 frames of 640 × 360 MJPG `.avi`, mid-gray background, a 40 × 40 white square starting at x = 20 and moving +8 px per frame horizontally, with no wrap-around. Frames 10, 30 and 50 are saved as PNGs from the generated arrays (not re-decoded).
- Inventory with `frame_idx` 10 / 30 / 50 and `--start-frame 0`: every offset is 0, every row verified, and `consistent` is true.
- With labels shifted to 12 / 32 / 52: every offset is −2.
- A label at `frame_idx` 2 with window 5 doesn't crash (window clipped at 0).
- A label at `frame_idx` 70 gives an empty `best_decode_idx` and `verified` false.
- `ncc(p, p) == 1 ± 1e-6`; `ncc(p, -p) == -1 ± 1e-6`; a constant image gives 0 with no NaN.
- With `--labels`, `gt_clip.mp4` opens with `cv2.VideoCapture` and has the expected frame count.

---

## 9. 1B.5 — Environment check: `src/skier/envcheck.py`, `scripts/check_env.py`

```text
python scripts/check_env.py [--strict] [--out DIR]      → DIR/env.json
```

`collect() -> dict` never raises. Missing packages are recorded as `null`.

- `timestamp_utc`, `os` (`platform.platform()`), `python` (`sys.version`), `git_commit` (`git rev-parse HEAD`, `null` on failure).
- `packages`: `importlib.metadata.version` for `torch, torchvision, mmcv, mmengine, mmpose, numpy, opencv-python, pillow`.
- If torch imports: `torch_cuda_build` (`torch.version.cuda`), `cuda_available`, `gpu_name`, `vram_gb` (`total_memory / 2**30`), `cudnn`. Without torch these are `null`; with torch but no GPU, `cuda_available` is `false`.
- `checks`: a list of `{name, ok, detail}` records, where `ok` is `true`, `false` or `null` (not applicable):
  - `numpy_lt_2`;
  - `torch_present`;
  - `cuda_available`;
  - `mmcv_ops_import` (`import mmcv.ops`, catching any exception and recording its message);
  - `mmpose_present`;
  - `gpu_conv` (a 3 × 3 `conv2d` on a `1×3×8×8` CUDA tensor, followed by `torch.cuda.synchronize()`).

Exit codes:
- Without `--strict`: always 0, so the check can run in the light environment.
- With `--strict` (used on AWS): 1 if any check has `ok` false or `null`.

Use a pure helper `numpy_ok(version: str) -> bool` (major version < 2).

Tests (`test_envcheck.py`):
- `main(["--out", tmp])` exits 0 in the light environment and writes `env.json` with all the keys above.
- Monkeypatch `importlib.metadata.version` to raise for `mmpose` → `packages.mmpose` is `null` and `mmpose_present.ok` is false.
- `--strict` with a failing check → exit 1.
- `numpy_ok("1.26.4")` is true; `numpy_ok("2.0.1")` is false.
- If `torch.cuda.is_available()` isn't true on the test machine, `cuda_available` is `false` or `null`.

---

## 10. 1B.6 — Pose smoke test: `src/skier/pose/smoke.py`, `scripts/smoke_pose.py`

```text
python scripts/smoke_pose.py --config CFG --checkpoint PTH --image IMG --box x1 y1 x2 y2
       [--device cuda:0] [--warmup 5] [--runs 20] [--out DIR]
```

All torch and mmpose imports happen inside `run()`. This script tests the install, so it uses MMPose's own pipeline; our pipeline is cross-checked against it in Step 3.

1. `model = mmpose.apis.init_model(config, checkpoint, device=device)`.
2. `res = inference_topdown(model, img, bboxes=np.array([box]), bbox_format="xyxy")`. Pass the skier's **tight** box: MMPose adds its own 1.25 padding.
3. From `res[0].pred_instances`, take `keypoints[0]` (17, 2) and `keypoint_scores[0]` (17,). If the shapes are wrong or any value is non-finite, exit 1.
4. **Timing:**
   - Run the warm-up calls first.
   - Then call `torch.cuda.reset_peak_memory_stats()` and time `--runs` calls with `time.perf_counter()`, calling `torch.cuda.synchronize()` before and after each call when on CUDA.
   - Peak memory = `torch.cuda.max_memory_allocated()`.
5. **Outputs:**
   - `generic_pose_prediction.npz` (keypoints, scores, box);
   - `smoke_pose.json` (config, checkpoint filename + SHA-256, image filename, box, device, warm-up and run counts, ms/image mean and median, peak memory in MB, torch and mmpose versions);
   - `smoke_pose_overlay.png`, drawn with the **COCO 17-point** skeleton constant below, never the 24-point one.

```python
COCO_SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                 (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
```

Tests (in `test_cli.py`; there's no offline test of `run()`):
- `build_parser()` parses `--box` as 4 floats.
- The lazy-import check (§12) passes for `skier.pose.smoke`.

---

## 11. 1B.7 — Tracker smoke test: `src/skier/tracking/smoke.py`, `scripts/smoke_tracker.py`

```text
python scripts/smoke_tracker.py --siamfc-repo DIR --weights PTH --video FILE
       --init-frame N --init-box x1 y1 x2 y2 [--max-frames 300] [--out DIR]
```

```python
class BoxTracker(Protocol):
    def init(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> None: ...
    def update(self, frame_bgr: np.ndarray) -> np.ndarray: ...       # xyxy

def xyxy_to_xywh1(box) -> np.ndarray     # (x1 + 1, y1 + 1, x2 - x1, y2 - y1)
def xywh1_to_xyxy(box) -> np.ndarray     # inverse

class SiamFCTracker:                      # implements BoxTracker; imports lazily
    def __init__(self, siamfc_repo: Path, weights: Path): ...   # upstream picks cuda:0 itself when available

def run_on_video(tracker, video_path, init_frame, init_box_xyxy, max_frames) -> list[tuple]
def write_boxes_csv(rows, path) -> None   # columns: frame_index, x1, y1, x2, y2 (3 decimals)
def read_boxes_csv(path) -> list[tuple]
def render_track(video_path, rows, out_path) -> None   # viz.draw_box + frame number; mp4v; source FPS
```

- **`SiamFCTracker`** inserts `siamfc_repo` at the front of `sys.path`, imports the tracker class, and loads the weights.
  - All conversions happen at this boundary: BGR → RGB, and `xyxy` ↔ `xywh1`.
  - Upstream `siamfc-pytorch` (huanglianghua) takes RGB images and 1-based `[x, y, w, h]` boxes in `init(img, box)` / `update(img)`. The class name, constructor and conventions **must be confirmed against the fork the SkiTB README points to** (§13).
- **`run_on_video`** decodes sequentially from frame 0.
  - It calls `init` once at `init_frame`, which gives the first row (the init box), then `update` on each following frame until `max_frames` rows exist or the video ends. It never re-initializes.
  - If `init_frame` is past the end of the file, it raises `ValueError`.
- **Outputs:** `boxes.csv` and `track.mp4`. The ±1 px ambiguity in the `xywh1` conversion is far below tracker error.

Tests (`test_tracker_smoke.py`):
- A `FakeTracker` shifts the box by (+3, +1) on every `update`. On a 20-frame synthetic video with `init_frame=5, max_frames=10`: rows cover frames 5–14, the first row is the init box, and the last is the init box + 9 × (3, 1).
- `init_frame` past the end of the file raises `ValueError`.
- `boxes.csv` round-trips to 3 decimals.
- The `xyxy` ↔ `xywh1` round trip is exact; (10, 20, 30, 60) ↔ (11, 21, 20, 40).
- `render_track` writes a video with one frame per row.

---

## 12. Tests: shared helpers and matrix

**`tests/synth.py`** (a plain helper module; pytest's default import mode puts `tests/` on `sys.path`):

```python
def make_annotation(rng, n_unreliable=2) -> list[list[float]]   # 24 × [x, y, v], x/y in [0.2, 0.8], all v=1 except n_unreliable 0s
def make_labels(videos=2, segments=2, images=3, seed=230) -> dict  # real nesting; IDs like "v0", "s0", "i0"
def write_labels(labels, path) -> Path
def make_blank_images(root, labels, size=(64, 48), template="{video_id}/{segment_id}/{image_id}.png") -> None
def make_moving_square_video(path, n=60, size=(640, 360), step=8, square=40, save_frames=(), frames_dir=None) -> list[Path]
```

Every test passes `--out` (and, for the audit, `--inventory`) under `tmp_path`, so tests never write to `$SKIER_RESULTS` or the repo.

**`tests/conftest.py`:** skip `stack`-marked tests if `mmpose` isn't importable, and `gpu`-marked tests if torch is missing or CUDA isn't available.

**Lazy-import check** (in `test_cli.py`): for each tool module, run `subprocess.run([sys.executable, "-c", "import sys, <module>; print('torch' in sys.modules)"])` and expect `False`. Also call `main(["--help"])` on all six tools and expect `SystemExit(0)`.

| Test file | Covers | Needs | Marker |
| --- | --- | --- | --- |
| `test_paths_io.py` | §2.6–2.7 | light | — |
| `test_schema.py` | §3 | light | — |
| `test_ski2dpose_raw.py` | §4–5 | light | — |
| `test_audit.py` | §6 | light | — |
| `test_viz.py` | §7 (drawing, selection, CLI end to end) | light | — |
| `test_alignment.py` | §8 | light (OpenCV video I/O) | — |
| `test_envcheck.py` | §9 | light (extra assertions when torch is present) | — |
| `test_tracker_smoke.py` | §11 | light | — |
| `test_cli.py` | `--help` ×6, lazy imports, §10 parser | light | — |

Step 1 has no `stack` or `gpu` tests. The first ones arrive with Step 3.1, using `pytest.importorskip("mmpose")`.

---

## 13. Fill in later (with real data or cloned code)

Each fill-in is a small commit, noted in `docs/data_audit.md` (data facts) or `docs/decisions.md` (code/tool facts).

| What | Where | How to find it | Guide step | Status |
| --- | --- | --- | --- | --- |
| `IMAGE_PATH_TEMPLATE` | `data/ski2dpose_raw.py` | Example loader + `unzip -l` of the images archive | 1E.1 | **Done 2026-10-17:** `Images_jpg/{video_id}/{segment_id}/{image_id}.jpg` |
| `info_for` (discipline and view-angle keys) | `data/ski2dpose_raw.py` | `info_preview` in `label_summary.json` | 1E.2 | **Done 2026-10-17:** `video_info.type`; view flags joined, e.g. `front+side` |
| `locate_segment` | `data/alignment.py` | File count and `ffprobe` durations in `videos/` | 1G.1 | **Done 2026-10-17:** `Videos/{video_id}/{segment_id}.mp4`, start 0 (confirm in 1G) |
| `MIN_SCORE`, `MIN_MARGIN` | `data/alignment.py` | First real segment's scores and margins | 1G.3 | Open |
| SiamFC import path, constructor, box and color conventions, `SCORE_ATTR` | `tracking/siamfc_adapter.py` (moved from `smoke.py` in Step 3) | Read the fork's tracker source | 1D / 1H.3 | Open (AWS) |
| SkiTB folder names, split-JSON shape, `frame_file()` (frames.txt entry → image file) | `data/skitb.py` | The real SkiTB files and toolkit README | When SkiTB arrives | Open |

---

## 14. Acceptance (this is guide 1B's "Done when")

In a fresh clone with only the light environment:

```bash
pip install -e ".[dev]"
pytest                                   # all pass; skips allowed only for stack/gpu markers
for s in audit_pose_labels draw_gt_overlays check_alignment check_env smoke_pose smoke_tracker; do
  python scripts/$s.py --help > /dev/null || echo "FAIL $s"
done
python scripts/check_env.py --out /tmp/skier_envcheck     # non-strict; exits 0
git status                               # tests wrote nothing into the repo
```
