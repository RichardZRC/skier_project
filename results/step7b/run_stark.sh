#!/bin/bash
# Step 7b (D28): the STARK-ST50 chain, a labeled post-freeze extra next to SiamFC's Step 8 run. Usage, inside Ubuntu
# on the GPU machine, after the val confidence check has filled configs/integration/stark.yaml:
#   bash results/step7b/run_stark.sh val    # the val comparison
#   bash results/step7b/run_stark.sh test   # ONCE, after val, from a pushed commit; refuses a second run
# Only the tracker-dependent steps run: SkiTB and Ski 2DPose tracking, propagation, replay. Pose and the alignment
# check don't depend on the tracker; the alignment's usable_segments.json is copied from SiamFC's run of the same
# partition (its sha256 is in freeze.json). Output: $SKIER_RESULTS/final_stark/<PART>/, then compare/ against
# $SKIER_RESULTS/final/<PART>/, whose file tree must be unchanged at the end.
set -euo pipefail
PART=${1:?usage: run_stark.sh val|test}
[ "$PART" = val ] || [ "$PART" = test ] || { echo "PART must be val or test"; exit 1; }
source ~/venvs/skier/bin/activate
export SKIER_DATA=${SKIER_DATA:-$HOME/skier_data} SKIER_RESULTS=${SKIER_RESULTS:-$HOME/skier_results}
cd ~/skier_project
F=$SKIER_RESULTS/final_stark/$PART
A=$SKIER_RESULTS/final/$PART
if [ -e $F ]; then echo "refusing: $F exists (STARK is evaluated once per set, D28)"; exit 1; fi
for f in $A/freeze.json $A/propagation/usable_segments.json $A/propagation/propagation.json $A/replay/smoothed.csv; do
  [ -f $f ] || { echo "refusing: $f is missing (SiamFC's run of $PART, the comparison's baseline)"; exit 1; }
done
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin
dirty=""
[ -n "$(git status --porcelain)" ] && dirty="the tree has uncommitted changes"
[ "$(git rev-parse HEAD)" != "$(git rev-parse -q --verify origin/$BRANCH || true)" ] && dirty="${dirty:+$dirty; }HEAD is not origin/$BRANCH"
if [ -n "$dirty" ]; then
  if [ $PART = test ]; then echo "refusing: $dirty"; exit 1; fi
  echo "warning (val only): $dirty"
fi

FT=(); [ $PART = test ] && FT=(--final-test)
SK=$SKIER_DATA/skitb
SPLIT=$SK/AL/AL_train_val_test_athlete_60-40.json
HR=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
STARK=$SKIER_DATA/vendor/Stark
STARK_COMMIT=162e1d37a38f92251cb380e90ccdb59608f2df3c
W=$STARK/checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar
W_SHA=23c8a677d9c2163e96e5225c1648ac4803e43232aef7c79044a367bde4a86b95
TRK=(--tracker stark --stark-repo $STARK --weights $W)
IC=configs/integration/stark.yaml
D=$SKIER_RESULTS/pose/d21
PS=data/manifests/pose_split_v1.json
RUNS=(E1-s230 E1-s231 E2-s230 E2-s231)
[ "$(git -C $STARK rev-parse HEAD)" = $STARK_COMMIT ] || { echo "refusing: $STARK is not at $STARK_COMMIT (D28)"; exit 1; }
echo "$W_SHA  $W" | sha256sum -c --quiet || { echo "refusing: $W is not the D28 checkpoint"; exit 1; }
read KPOSE RS RK TRACKER < <(python -I -c "import sys; sys.path.insert(0, 'src'); from skier.integration.config import IntegrationConfig as C; c = C.from_yaml('$IC').require('k_pose', 'conf_signal', 'replay_uses_status', 'pose_refine', 'refine_min_score', 'refine_k'); print(c.k_pose, c.refine_min_score, c.refine_k, c.tracker)")
[ "${TRACKER:-}" = stark ] || { echo "refusing: $IC must be filled in and say tracker: stark"; exit 1; }

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
A_BEFORE=$(treesha $A)
mkdir -p $F/logs
LOG=$F/run.log
exec > >(tee -a $LOG) 2>&1
echo "=== $(date '+%F %T') run_stark $PART at $(git rev-parse --short HEAD) ($BRANCH); STARK ${STARK_COMMIT:0:7}; k_pose $KPOSE; refine s_min $RS k_refine $RK"

python -I - "$F/freeze.json" "$PART" "$A_BEFORE" <<PY
import hashlib, json, subprocess, sys
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
files = {"pose_split": "$PS", "skitb_split": "$SPLIT", "fps_table": "data/manifests/skitb_fps_v1.csv",
         "integration_config": "$IC", "replay_config": "configs/replay/default.yaml", "hrnet_config": "$HR",
         "stark_weights": "$W", "siamfc_freeze": "$A/freeze.json", "usable_segments": "$A/propagation/usable_segments.json",
         **{f"checkpoint_{r}": f"$D/{r}/best.pt" for r in "${RUNS[*]}".split()}}
freeze = {"partition": sys.argv[2], "tracker": "stark", "stark_commit": "$STARK_COMMIT", "branch": "$BRANCH",
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "split_content_sha256": json.load(open("$PS"))["content_sha256"], "siamfc_root": "$A",
          "siamfc_root_tree_sha256_before": sys.argv[3],
          "files": {k: {"path": v, "sha256": sha(v)} for k, v in files.items()}}
Path(sys.argv[1]).write_text(json.dumps(freeze, indent=1))
print("freeze.json:", len(freeze["files"]), "files hashed, commit", freeze["git_commit"][:7])
PY

step() {  # $1 log name, then the command; the full output goes to logs/, the last lines to the run log
  local name=$1; shift
  "$@" > $F/logs/$name.log 2>&1 || { tail -20 $F/logs/$name.log; echo "step $name failed"; exit 1; }
  grep -v -i warn $F/logs/$name.log | tail -${TAILN:-1}
}

echo "=== SkiTB tracking ($PART, STARK)"
step skitb_tracking python scripts/run_tracker.py "${TRK[@]}" --skitb-root $SK --split-json $SPLIT --partition $PART \
  "${FT[@]}" --out $F/tracking/skitb

echo "=== Ski 2DPose: tracking, propagation (k_pose $KPOSE; refined crop D26; status from $IC)"
mkdir -p $F/propagation
cp $A/propagation/usable_segments.json $F/propagation/usable_segments.json
step ski2dpose_tracking python scripts/run_tracker.py "${TRK[@]}" --ski2dpose-partition $PART "${FT[@]}" \
  --out $F/tracking/ski2dpose
CK=(); for r in "${RUNS[@]}"; do CK+=(--checkpoint "$r=$D/$r/best.pt"); done
TAILN=10 step propagation python scripts/eval_propagation.py --split $PS --partition $PART "${FT[@]}" \
  --usable $F/propagation/usable_segments.json --track-dir $F/tracking/ski2dpose "${CK[@]}" --recipe E2 \
  --k-pose $KPOSE --refine $RS $RK --config $IC --hrnet-config $HR --out $F/propagation

echo "=== replay ($PART)"
step replay python scripts/make_replay.py --evaluate --track-dir $F/tracking/skitb --skitb-root $SK --split-json $SPLIT \
  --ski2dpose-track-dir $F/tracking/ski2dpose --partition $PART "${FT[@]}" --integration-config $IC --out $F/replay

echo "=== tables, comparison with $A, figures"
step tables python scripts/make_tables.py --root $F --partition $PART --skitb-root $SK --skitb-split $SPLIT \
  --integration-config $IC
step compare python scripts/compare_trackers.py --a $A --b $F --config-b $IC --partition $PART --skitb-root $SK \
  --skitb-split $SPLIT --out $F/compare
CMP=($F/compare/compare.json)
[ $PART = test ] && [ -f $SKIER_RESULTS/final_stark/val/compare/compare.json ] && \
  CMP=($SKIER_RESULTS/final_stark/val/compare/compare.json "${CMP[@]}")
step figures python scripts/make_figures.py --tables $F/tables/tables.json --compare "${CMP[@]}" --out $F/figures

A_AFTER=$(treesha $A)
[ "$A_AFTER" = "$A_BEFORE" ] || { echo "ERROR: $A changed during the run"; exit 1; }
echo "$A unchanged (tree sha256 ${A_AFTER:0:12})"
echo "=== $(date '+%F %T') run_stark $PART done → $F"
