#!/bin/bash
# Build the final report in a temporary copy; write final_report_preview.pdf at the repo root (gitignored: it shows
# dataset frames). Needs TeX Live (pdflatex, bibtex) and python3. The two dataset-frame figures are gitignored; when
# they are missing, same-size gray placeholders are drawn, so page count and float positions are unchanged.
# Usage: bash final_report/tools/build.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
B="$(mktemp -d)"
cp -r "$REPO/final_report/." "$B/"
cd "$B" || exit 1
perl -pi -e 's/\r$//' main.tex references.bib   # the Windows clone checks out CRLF
python3 "$HERE/placeholders.py" figures
pdflatex -interaction=nonstopmode main.tex > l1.txt 2>&1
bibtex main > b.txt 2>&1
pdflatex -interaction=nonstopmode main.tex > l2.txt 2>&1
pdflatex -interaction=nonstopmode main.tex > l3.txt 2>&1
echo "== errors and warnings (none listed = clean)"
grep -E '^!|Warning: (Reference|Citation)|undefined' main.log | sort -u | head -20
grep -iE 'warning|error' main.blg | grep -v 'warning\$ -- 0' | head -5
grep -E 'Output written' main.log
cp main.pdf "$REPO/final_report_preview.pdf" && echo "preview: $REPO/final_report_preview.pdf"
echo "build dir: $B"
