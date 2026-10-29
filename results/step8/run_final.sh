#!/bin/bash
# Step 8: the frozen evaluation run (plan §13; D23-D25). Usage, inside Ubuntu on the GPU machine:
#   bash results/step8/run_final.sh val    # the cheap trial: the identical chain on val
#   bash results/step8/run_final.sh test   # ONCE: test + the follow-cam set (D11); refuses a second run
# Preflight: clean tree and HEAD == origin/main; freeze.json (hashes of everything that decides a number) is
# written before any output. Output: $SKIER_RESULTS/final/<PART>/ (and final/followcam/ for test).
set -euo pipefail
PART=${1:?usage: run_final.sh val|test}
[ "$PART" = val ] || [ "$PART" = test ] || { echo "PART must be val or test"; exit 1; }
source ~/venvs/skier/bin/activate
export SKIER_DATA=${SKIER_DATA:-$HOME/skier_data} SKIER_RESULTS=${SKIER_RESULTS:-$HOME/skier_results}
cd ~/skier_project
F=$SKIER_RESULTS/final/$PART
FC=$SKIER_RESULTS/final/followcam
for d in $F $( [ $PART = test ] && echo $FC ); do
  if [ -e $d ]; then echo "refusing: $d exists (each set is evaluated once; move it aside only with a recorded reason)"; exit 1; fi
done
git fetch -q origin
dirty=""
[ -n "$(git status --porcelain)" ] && dirty="the tree has uncommitted changes"
[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ] && dirty="${dirty:+$dirty; }HEAD is not origin/main"
if [ -n "$dirty" ]; then
  if [ $PART = test ]; then echo "refusing: $dirty"; exit 1; fi
  echo "warning (val trial only): $dirty"
fi

FT=(); [ $PART = test ] && FT=(--final-test)
V=$SKIER_DATA/vendor
SK=$SKIER_DATA/skitb
SPLIT=$SK/AL/AL_train_val_test_athlete_60-40.json
HR=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
COCO=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth
SIAM=(--siamfc-repo $V/siamfc_pytorch --weights $V/siamfc_pytorch/pretrained/siamfc_alexnet_e50.pth)
D=$SKIER_RESULTS/pose/d21
PS=data/manifests/pose_split_v1.json
RUNS=(E1-s230 E1-s231 E2-s230 E2-s231)
KPOSE=$(python -I -c "import sys; sys.path.insert(0, 'src'); from skier.integration.config import IntegrationConfig as C; print(C.from_yaml('configs/integration/default.yaml').require('k_pose', 'conf_signal', 'replay_uses_status').k_pose)")
mkdir -p $F
LOG=$F/run.log
exec > >(tee -a $LOG) 2>&1
echo "=== $(date '+%F %T') run_final $PART at $(git rev-parse --short HEAD); k_pose $KPOSE"

python -I - "$F/freeze.json" "$PART" <<PY
import hashlib, json, subprocess, sys
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
files = {"pose_split": "$PS", "skitb_split": "$SPLIT", "fps_table": "data/manifests/skitb_fps_v1.csv",
         "integration_config": "configs/integration/default.yaml", "replay_config": "configs/replay/default.yaml",
         "pose_config_e1": "configs/pose/e1.yaml", "pose_config_e2": "configs/pose/e2.yaml", "hrnet_config": "$HR",
         "coco_checkpoint": "$COCO", "siamfc_weights": "$V/siamfc_pytorch/pretrained/siamfc_alexnet_e50.pth",
         **{f"checkpoint_{r}": f"$D/{r}/best.pt" for r in "${RUNS[*]}".split()}}
freeze = {"partition": sys.argv[2], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "split_content_sha256": json.load(open("$PS"))["content_sha256"],
          "files": {k: {"path": v, "sha256": sha(v)} for k, v in files.items()}}
Path(sys.argv[1]).write_text(json.dumps(freeze, indent=1))
print("freeze.json:", len(freeze["files"]), "files hashed, commit", freeze["git_commit"][:7])
PY

pose() {  # $1 partition, $2 output folder
  for r in "${RUNS[@]}"; do
    python scripts/eval_pose.py --checkpoint $D/$r/best.pt --hrnet-config $HR --split $PS --partition $1 "${FT[@]}" \
      --experiment $r --out $2/pose/$r 2>&1 | grep -v -i warn | tail -1
  done
  python scripts/eval_generic_pose.py --config $HR --checkpoint $COCO --split $PS --partition $1 "${FT[@]}" \
    --out $2/pose/E0 2>&1 | grep -v -i warn | tail -1
}
echo "=== pose ($PART)"; pose $PART $F
if [ $PART = test ]; then mkdir -p $FC; echo "=== pose (followcam)"; pose followcam $FC; fi

echo "=== SkiTB tracking ($PART)"
python scripts/run_tracker.py "${SIAM[@]}" --skitb-root $SK --split-json $SPLIT --partition $PART "${FT[@]}" \
  --out $F/tracking/skitb 2>&1 | grep -v -i warn | tail -1

echo "=== Ski 2DPose: alignment, tracking, propagation (k_pose $KPOSE)"
python scripts/eval_propagation.py --split $PS --partition $PART "${FT[@]}" --check-alignment --out $F/propagation
python scripts/run_tracker.py "${SIAM[@]}" --ski2dpose-partition $PART "${FT[@]}" --out $F/tracking/ski2dpose 2>&1 \
  | grep -v -i warn | tail -1
CK=(); for r in "${RUNS[@]}"; do CK+=(--checkpoint "$r=$D/$r/best.pt"); done
python scripts/eval_propagation.py --split $PS --partition $PART "${FT[@]}" --usable $F/propagation/usable_segments.json \
  --track-dir $F/tracking/ski2dpose "${CK[@]}" --recipe E2 --k-pose $KPOSE --hrnet-config $HR --out $F/propagation 2>&1 \
  | grep -v -i warn | tail -2

echo "=== replay ($PART)"
python scripts/make_replay.py --evaluate --track-dir $F/tracking/skitb --skitb-root $SK --split-json $SPLIT \
  --ski2dpose-track-dir $F/tracking/ski2dpose --partition $PART "${FT[@]}" --out $F/replay

echo "=== tables and figures"
AUG=(); [ $PART = val ] && AUG=(--d21-dir $D)
python scripts/make_tables.py --root $F --partition $PART --skitb-root $SK --skitb-split $SPLIT "${AUG[@]}"
TABLES=($F/tables/tables.json)
if [ $PART = test ]; then
  python scripts/make_tables.py --root $FC --partition followcam
  TABLES+=($FC/tables/tables.json)
  [ -f $SKIER_RESULTS/final/val/tables/tables.json ] && TABLES=($SKIER_RESULTS/final/val/tables/tables.json "${TABLES[@]}")
fi
python scripts/make_figures.py --tables "${TABLES[@]}" --d21-dir $D --out $F/figures
echo "=== $(date '+%F %T') run_final $PART done → $F"
