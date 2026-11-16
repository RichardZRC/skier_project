#!/bin/bash
# Step 6b (D29): pseudo-labels on SkiTB training races ("noisy student lite"), a labeled post-freeze extra.
# Stages, in order, inside Ubuntu on the GPU machine (branch pseudo):
#   bash results/step6b/run_pseudo.sh calibrate   # the filter, on Ski 2DPose val labels -> results/step6b/calibration.json
#   bash results/step6b/run_pseudo.sh labels      # pseudo-labels on SkiTB training frames (D29 + calibration committed)
#   bash results/step6b/run_pseudo.sh train       # the student PL, seeds 230 and 231 (about 2.6 h)
#   bash results/step6b/run_pseudo.sh val         # PL vs. E2 on val            (added with compare_pseudo)
#   bash results/step6b/run_pseudo.sh test        # ONCE, from a pushed commit  (added with compare_pseudo)
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
TEACH=(--teacher E2-s230=$D/E2-s230/best.pt --teacher E2-s231=$D/E2-s231/best.pt)
mkdir -p $P $E
log() { echo "=== $(date '+%F %T') $*"; }
refuse() { echo "refusing: $*"; exit 1; }
grep -q '^## D29' docs/decisions.md || refuse "D29 must be recorded in docs/decisions.md first"
[ "$(git rev-parse --abbrev-ref HEAD)" = pseudo ] || refuse "run on branch pseudo"

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
*)
  refuse "stage $STAGE is not implemented yet"
  ;;
esac
