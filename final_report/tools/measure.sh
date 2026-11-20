#!/bin/bash
# Page-limit diagnostics for the final report (scratch builds only; the repo is not touched):
#   1. where the main text ends (in pages), each main-text float's height, room on page 1 below the Introduction;
#   2. last-line fill of every paragraph (short last lines are the cheapest cuts);
#   3. prose words per section.
# Needs TeX Live and python3. Usage: bash final_report/tools/measure.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
compile() {  # $1 = tex basename without .tex
  pdflatex -interaction=nonstopmode "$1.tex" > "$1.l1" 2>&1; bibtex "$1" > "$1.b" 2>&1
  pdflatex -interaction=nonstopmode "$1.tex" > "$1.l2" 2>&1; pdflatex -interaction=nonstopmode "$1.tex" > "$1.l3" 2>&1
  grep -E '^!' "$1.log" | head -3
  return 0
}
B="$(mktemp -d)"
cp -r "$REPO/final_report/." "$B/"
cd "$B" || exit 1
perl -pi -e 's/\r$//' main.tex references.bib
python3 "$HERE/placeholders.py" figures > /dev/null
python3 "$HERE/measure_layout.py" mark main.tex m.tex && compile m && python3 "$HERE/measure_layout.py" read m.aux
echo
python3 "$HERE/para_fill.py" mark main.tex p.tex && compile p && python3 "$HERE/para_fill.py" read p.aux p.tex
echo
python3 "$HERE/words.py" main.tex
