#!/bin/bash
# SiamFC-vs-STARK demo set (D28 extra; val only): clips picked by results/step7b/demo_set.py's fixed rules
# (2 both good, 1 both bad, 2 STARK better; clips of at least 5 s; one clip per sequence). Each clip is rendered from the val chains' own
# track CSVs (--track-csv), so the demos show exactly the evaluated tracks; same system otherwise (E2-s230,
# refinement on, each tracker's integration config). Videos show dataset frames and stay under $SKIER_RESULTS.
# Usage, inside Ubuntu on the GPU machine:  bash results/step7b/make_demo_set.sh
set -euo pipefail
cd ~/skier_project
source ~/venvs/skier/bin/activate
export SKIER_DATA=${SKIER_DATA:-$HOME/skier_data} SKIER_RESULTS=${SKIER_RESULTS:-$HOME/skier_results}
SK=$SKIER_DATA/skitb; SPLIT=$SK/AL/AL_train_val_test_athlete_60-40.json
HR=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
CK=$SKIER_RESULTS/pose/d21/E2-s230/best.pt
O=$SKIER_RESULTS/demo_set
[ -e $O ] && { echo "refusing: $O exists"; exit 1; }
mkdir -p $O
python results/step7b/demo_set.py --n-good 2 --n-bad 1 --n-better 2 --min-seconds 5 --out $O/demo_set.json \
  | grep -E '^(PICK|wrote|[0-9]+ of)'
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
lab() { echo "drawtext=fontfile=$FONT:text='$1':x=20:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"; }
i=0
python3 -c "import json; d=json.load(open('$O/demo_set.json'))['picks']; [print(c, t['clip'], t['siamfc'], t['stark']) for c in ('both_good','both_bad','stark_better') for t in d[c]]" |
while read CAT CLIP IA IB; do
  i=$((i + 1)); NAME=${i}_${CAT}_${CLIP}; SEQ=${CLIP%%__*}; CAM=${CLIP##*__}
  echo "=== $(date '+%T') $NAME (IoU siamfc $IA, stark $IB)"
  for T in siamfc stark; do
    if [ $T = siamfc ]; then TC=$SKIER_RESULTS/final/val/tracking/skitb/$CLIP.csv; IC=configs/integration/default.yaml
    else TC=$SKIER_RESULTS/final_stark/val/tracking/skitb/$CLIP.csv; IC=configs/integration/stark.yaml; fi
    python scripts/process_clip.py --skitb-clip $SEQ/$CAM --skitb-root $SK --split-json $SPLIT --partition val \
      --track-csv $TC --config $IC --checkpoint $CK --hrnet-config $HR --out $O/$NAME/$T < /dev/null 2>&1 \
      | grep -v -i warn | tail -1
    for v in replay debug; do
      ffmpeg -v error -y -i $O/$NAME/$T/$v.mp4 -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p \
        -movflags +faststart $O/$NAME/$T/${v}_h264.mp4 < /dev/null
    done
  done
  for v in debug replay; do
    ffmpeg -v error -y -i $O/$NAME/siamfc/${v}_h264.mp4 -i $O/$NAME/stark/${v}_h264.mp4 -filter_complex \
      "[0:v]scale=960:-2,$(lab "SiamFC  (clip IoU $IA)")[a];[1:v]scale=960:-2,$(lab "STARK-ST50  (clip IoU $IB)")[b];[a][b]hstack=inputs=2[v]" \
      -map "[v]" -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart $O/$NAME/${v}_side_by_side.mp4 < /dev/null
  done
done
ls $O/*/*_side_by_side.mp4
echo "=== $(date '+%T') done"
