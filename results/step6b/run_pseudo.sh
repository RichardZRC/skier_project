#!/bin/bash
# Step 6b (D29): pseudo-labels on SkiTB training races ("noisy student lite"), a labeled post-freeze extra.
# Stages, in order, inside Ubuntu on the GPU machine (branch pseudo):
#   bash results/step6b/run_pseudo.sh calibrate   # the filter, on Ski 2DPose val labels -> results/step6b/calibration.json
#   bash results/step6b/run_pseudo.sh labels      # pseudo-labels on SkiTB training frames (D29 + calibration committed)
#   bash results/step6b/run_pseudo.sh train       # the student PL, seeds 230 and 231 (about 2.2 h)
#   bash results/step6b/run_pseudo.sh val         # PL vs. E2 on val (clean, moderate crops, tracker crops)
#   bash results/step6b/run_pseudo.sh test        # ONCE, after val, from a pushed commit: test + follow-cam
# Large outputs: $SKIER_RESULTS/pseudo/. Text evidence only: results/step6b/ (no images, no labels).
set -euo pipefail
STAGE=${1:?usage: run_pseudo.sh calibrate|labels|train|val|test}
source ~/venvs/skier/bin/activate
export SKIER_DATA=${SKIER_DATA:-$HOME/skier_data} SKIER_RESULTS=${SKIER_RESULTS:-$HOME/skier_results}
cd ~/skier_project
P=$SKIER_RESULTS/pseudo
E=results/step6b
HR=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
COCO=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth
D=$SKIER_RESULTS/pose/d21
PS=data/manifests/pose_split_v1.json
CE=data/manifests/crop_error_set_v1.json
TEACH=(--teacher E2-s230=$D/E2-s230/best.pt --teacher E2-s231=$D/E2-s231/best.pt)
mkdir -p $P $E
log() { echo "=== $(date '+%F %T') $*"; }
refuse() { echo "refusing: $*"; exit 1; }
grep -q '^## D29' docs/decisions.md || refuse "D29 must be recorded in docs/decisions.md first"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = pseudo ] || refuse "run on branch pseudo"

treesha() {  # one hash over every file under $1 (relative path + content)
  python -I - "$1" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
h = hashlib.sha256()
for p in sorted(q for q in root.rglob("*") if q.is_file()):
    h.update(str(p.relative_to(root)).encode() + b"\0")
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
print(h.hexdigest())
PY
}

freeze() {  # $1 output json, $2 partition, $3 frozen root, $4 its tree hash before
  python -I - "$@" <<PY
import hashlib, json, subprocess, sys
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
out, part, root, before = sys.argv[1:5]
files = {"pose_split": "$PS", "crop_error_set": "$CE", "hrnet_config": "$HR", "coco_checkpoint": "$COCO",
         "pose_config_e2": "configs/pose/e2.yaml", "integration_config": "configs/integration/default.yaml",
         "teacher_E2-s230": "$D/E2-s230/best.pt", "teacher_E2-s231": "$D/E2-s231/best.pt",
         "student_PL-s230": "$P/PL-s230/best.pt", "student_PL-s231": "$P/PL-s231/best.pt",
         "pseudo_labels": "$P/labels/pseudo_labels.json", "calibration": "$E/calibration.json"}
if Path(f"{root}/freeze.json").is_file():
    files["frozen_freeze"] = f"{root}/freeze.json"
if part != "followcam":
    files["usable_segments"] = f"{root}/propagation/usable_segments.json"
freeze = {"partition": part, "experiment": "D29 pseudo-label student PL (post-freeze extra)", "branch": "$BRANCH",
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "split_content_sha256": json.load(open("$PS"))["content_sha256"], "frozen_root": root,
          "frozen_root_tree_sha256_before": before,
          "files": {k: {"path": v, "sha256": sha(v)} for k, v in files.items()}}
Path(out).write_text(json.dumps(freeze, indent=1))
print("freeze.json:", len(freeze["files"]), "files hashed, commit", freeze["git_commit"][:7])
PY
}

read KPOSE RS RK < <(python -I -c "import sys; sys.path.insert(0, 'src'); from skier.integration.config import IntegrationConfig as C; c = C.from_yaml('configs/integration/default.yaml').require('k_pose', 'refine_min_score', 'refine_k'); print(c.k_pose, c.refine_min_score, c.refine_k)")

pose_eval() {  # $1 partition, $2 output root, then extra flags
  local part=$1 out=$2; shift 2
  for s in 230 231; do
    python scripts/eval_pose.py --checkpoint $P/PL-s$s/best.pt --hrnet-config $HR --split $PS --partition $part \
      --experiment PL-s$s --out $out/pose/PL-s$s "$@" 2>&1 | grep -v -i warn | tail -1
  done
}

prop_eval() {  # $1 partition, $2 frozen root (usable segments + SiamFC tracks), $3 output root, then extra flags
  local part=$1 root=$2 out=$3; shift 3
  python scripts/eval_propagation.py --split $PS --partition $part --usable $root/propagation/usable_segments.json \
    --track-dir $root/tracking/ski2dpose --checkpoint PL-s230=$P/PL-s230/best.pt --checkpoint PL-s231=$P/PL-s231/best.pt \
    --recipe PL --k-pose $KPOSE --refine $RS $RK --hrnet-config $HR --out $out/propagation "$@" 2>&1 \
    | grep -v -i warn | tail -4
}

case $STAGE in
calibrate)
  [ -e $P/calibration ] && refuse "$P/calibration exists"
  log "calibrate at $(git rev-parse --short HEAD)"
  python scripts/make_pseudo_labels.py --calibrate "${TEACH[@]}" --hrnet-config $HR --split $PS \
    --out $P/calibration 2>&1 | grep -v -i warn | tee $P/calibrate.log
  cp $P/calibration/calibration.json $E/calibration.json
  ;;
labels)
  [ -e $P/labels ] && refuse "$P/labels exists"
  cmp -s $E/calibration.json $P/calibration/calibration.json || refuse "$E/calibration.json differs from the run's"
  git ls-files --error-unmatch $E/calibration.json >/dev/null 2>&1 && \
    git diff --quiet HEAD -- $E/calibration.json docs/decisions.md || \
    refuse "commit D29 and $E/calibration.json first (checkpoint C1: the rule is dated before any label)"
  log "labels at $(git rev-parse --short HEAD)"
  python scripts/make_pseudo_labels.py "${TEACH[@]}" --hrnet-config $HR --split $PS --filter $E/calibration.json \
    --out $P/labels 2>&1 | grep -v -i warn | tee $P/labels.log
  python3 - $P/labels/summary.json $P/labels/pseudo_labels.json $E/pseudo_summary.json <<'PY'
import hashlib, json, sys
s = json.load(open(sys.argv[1]))
s["pseudo_labels_sha256"] = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
json.dump(s, open(sys.argv[3], "w"), indent=1, sort_keys=True)
PY
  ;;
train)
  L=$P/labels/pseudo_labels.json
  [ -f $L ] || refuse "no pseudo-labels at $L"
  for s in 230 231; do [ -e $P/PL-s$s ] && refuse "$P/PL-s$s exists"; done
  for s in 230 231; do
    log "train PL seed $s at $(git rev-parse --short HEAD)"
    python scripts/train_pose.py --split $PS --hrnet-config $HR --checkpoint $COCO --config configs/pose/e2.yaml \
      --experiment PL --seed $s --pseudo-labels $L --out $P/PL-s$s 2>&1 | grep -v -i warn | tail -4
    cp $P/PL-s$s/log.csv $E/PL-s${s}_log.csv
  done
  log "train done"
  ;;
val)
  for s in 230 231; do [ -f $P/PL-s$s/best.pt ] || refuse "no $P/PL-s$s/best.pt"; done
  V=$P/eval/val
  A=$SKIER_RESULTS/final/val
  [ -e $V ] && refuse "$V exists"
  for f in $A/freeze.json $A/propagation/usable_segments.json $A/propagation/propagation.json; do
    [ -f $f ] || refuse "$f is missing (the frozen val run)"
  done
  [ -n "$(git status --porcelain)" ] && echo "warning (val only): the tree has uncommitted changes"
  mkdir -p $V
  exec > >(tee -a $V/run.log) 2>&1
  BEFORE=$(treesha $A)
  log "val at $(git rev-parse --short HEAD); k_pose $KPOSE; refine s_min $RS k_refine $RK"
  freeze $V/freeze.json val $A $BEFORE
  log "pose (clean + crop-error set)"; pose_eval val $V --crop-error-set $CE
  log "propagation (frozen SiamFC tracks)"; prop_eval val $A $V
  log "compare"
  python scripts/compare_pseudo.py --partition val --e2 $D/E2-s230/eval/val --e2 $D/E2-s231/eval/val \
    --pl $V/pose/PL-s230 --pl $V/pose/PL-s231 --levels clean moderate --prop-e2 $A/propagation \
    --prop-pl $V/propagation --k-pose $KPOSE --out $V/compare | grep -E '^\*\*|^\| (PL|E2|Δ)'
  [ "$(treesha $A)" = "$BEFORE" ] || refuse "$A changed during the run"
  mkdir -p $E/val && cp $V/compare/compare.json $V/compare/compare.md $V/freeze.json $V/run.log $E/val/
  log "val done; $A unchanged"
  ;;
test)
  [ -f $P/eval/val/compare/compare.json ] || refuse "run val first (D29)"
  for d in $P/eval/test $P/eval/followcam; do [ -e $d ] && refuse "$d exists (test is run once, D29)"; done
  git fetch -q origin
  [ -z "$(git status --porcelain)" ] || refuse "the tree has uncommitted changes"
  [ "$(git rev-parse HEAD)" = "$(git rev-parse -q --verify origin/$BRANCH || true)" ] || refuse "HEAD is not origin/$BRANCH"
  A=$SKIER_RESULTS/final/test
  FCA=$SKIER_RESULTS/final/followcam
  for f in $A/freeze.json $A/propagation/usable_segments.json $A/propagation/propagation.json $FCA/pose/E2-s230/metrics.json; do
    [ -f $f ] || refuse "$f is missing (the frozen test run)"
  done
  V=$P/eval/test
  FV=$P/eval/followcam
  mkdir -p $V $FV
  exec > >(tee -a $V/run.log) 2>&1
  BEFORE=$(treesha $A)
  FBEFORE=$(treesha $FCA)
  log "TEST (once) at $(git rev-parse --short HEAD); k_pose $KPOSE; refine s_min $RS k_refine $RK"
  freeze $V/freeze.json test $A $BEFORE
  freeze $FV/freeze.json followcam $FCA $FBEFORE
  log "pose (test)"; pose_eval test $V --final-test
  log "pose (followcam)"; pose_eval followcam $FV --final-test
  log "propagation (test, frozen SiamFC tracks)"; prop_eval test $A $V --final-test
  log "compare (test: same event / other events)"
  python scripts/compare_pseudo.py --partition test --e2 $A/pose/E2-s230 --e2 $A/pose/E2-s231 \
    --pl $V/pose/PL-s230 --pl $V/pose/PL-s231 --prop-e2 $A/propagation --prop-pl $V/propagation --k-pose $KPOSE \
    --out $V/compare | grep -E '^\*\*|^\| (PL|E2|Δ)'
  log "compare (followcam)"
  python scripts/compare_pseudo.py --partition followcam --e2 $FCA/pose/E2-s230 --e2 $FCA/pose/E2-s231 \
    --pl $FV/pose/PL-s230 --pl $FV/pose/PL-s231 --out $FV/compare | grep -E '^\*\*|^\| (PL|E2|Δ)'
  [ "$(treesha $A)" = "$BEFORE" ] || refuse "$A changed during the run"
  [ "$(treesha $FCA)" = "$FBEFORE" ] || refuse "$FCA changed during the run"
  mkdir -p $E/test $E/followcam
  cp $V/compare/compare.json $V/compare/compare.md $V/freeze.json $V/run.log $E/test/
  cp $FV/compare/compare.json $FV/compare/compare.md $FV/freeze.json $E/followcam/
  log "test done; $A and $FCA unchanged"
  ;;
*)
  refuse "unknown stage $STAGE"
  ;;
esac
