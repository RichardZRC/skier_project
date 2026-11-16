#!/bin/bash
# Stills from the STARK demo of val clip AL0073 cam 7 (D28): exact frames of the side-by-side videos
# (SiamFC left, STARK-ST50 right), at full resolution. They show dataset frames (SkiTB, CC BY-NC), so the output
# folder is gitignored: upload to Overleaf by hand, like qualitative_val.png.
# Needs ffmpeg and the demo videos (made 2026-11-15; see results/step7b/demo_AL0073_7.json).
# Usage: bash final_report/figures/stark_demo_stills.sh [OUT_DIR]   (default: final_report/figures/stark_demo/)
set -euo pipefail
SKIER_RESULTS=${SKIER_RESULTS:-$HOME/skier_results}
V=$SKIER_RESULTS/final_stark/demos/AL0073__7_siamfc_vs_stark
OUT=${1:-$(dirname "$0")/stark_demo}
mkdir -p "$OUT"
still() {  # $1 view (debug|replay), $2 frame index (30 fps); select by frame number, not by time
  local f; f=$(printf "%03d" "$2")
  ffmpeg -v error -y -i "$V/$1_side_by_side.mp4" -vf "select=eq(n\,$2)" -frames:v 1 "$OUT/AL0073_7_$1_f$f.png"
}
# debug view (full frame; yellow: replay window, green: tracker box)
still debug 45    # 1.5 s: both trackers have lost the skier (fence and spectators)
still debug 120   # 4.0 s: STARK has found the skier again; SiamFC has not
still debug 195   # 6.5 s: STARK on the skier (box on the lower body); SiamFC on the fence
still debug 270   # 9.0 s: after the view changes, both lose the skier
# replay view (the centered 16:9 replay with the skeleton)
still replay 138  # 4.6 s: STARK frames the skier, but its flag says low confidence, so no skeleton is drawn
still replay 195  # 6.5 s: SiamFC draws a skeleton on the fence; STARK frames the skier (low confidence, no skeleton)
ls -la "$OUT"
