"""Where does the main text end, how tall is each main-text float, and how much room is left on page 1 below the
Introduction? Used for the page-limit trim (main text = everything before Contributions; limit 5 pages).
Usage: measure_layout.py mark IN.tex OUT.tex   (then compile OUT.tex)
       measure_layout.py read OUT.aux
Geometry is the NeurIPS 2026 style: 11in paper, 9in text block below a 1in top margin (pdf y counted from the bottom).
"""
import re
import sys
from pathlib import Path

BS = chr(92)
TOP, BOT = 720.0, 72.0


def mark(inp, outp):
    src = Path(inp).read_text()
    macro = (BS + "makeatletter\n"
             + BS + "newcommand{" + BS + "possaved}[3]{}\n"
             + BS + "newcommand{" + BS + "possave}[1]{" + BS + "pdfsavepos" + BS + "write" + BS + "@auxout{"
             + BS + "string" + BS + "possaved{#1}{" + BS + "thepage}{" + BS + "the" + BS + "pdflastypos}}}\n"
             + BS + "makeatother\n")
    src = src.replace(BS + "begin{document}", macro + BS + "begin{document}", 1)
    end_main = src.index(BS + "section{Contributions}")
    pat = re.compile("(" + re.escape(BS) + r"begin\{(figure|table)\}(\[[^\]]*\])?)(.*?)(" + re.escape(BS)
                     + r"label\{([^}]*)\})", re.S)
    out, i = [], 0
    for m in pat.finditer(src):
        if m.start() > end_main:
            break
        lab = m.group(6)
        out.append(src[i:m.start()])
        out.append(m.group(1) + BS + "possave{top:" + lab + "}" + m.group(4) + m.group(5) + BS + "par"
                   + BS + "possave{bot:" + lab + "}")
        i = m.end()
    out.append(src[i:])
    src = "".join(out)
    c = src.index(BS + "section{Contributions}")
    src = src[:c].rstrip() + BS + "possave{endmain}\n\n" + src[c:]
    src = re.sub(r"not shared with another\s+class\.", lambda m: m.group(0) + BS + "possave{endintro}", src, count=1)
    Path(outp).write_text(src)


def read(aux):
    text = Path(aux).read_text()
    pat = re.escape(BS) + r"possaved\{([^}]*)\}\{(\d+)\}\{(-?\d+)\}"
    pos = {m.group(1): (int(m.group(2)), int(m.group(3)) / 65536) for m in re.finditer(pat, text)}
    p, y = pos["endmain"]
    print(f"main text ends on page {p}, {100 * (TOP - y) / (TOP - BOT):.0f}% down its text block "
          f"-> {p - 1 + (TOP - y) / (TOP - BOT):.2f} pages")
    total = 0.0
    for k in pos:
        if k.startswith("top:"):
            lab = k[4:]
            (p1, y1), (p2, y2) = pos[k], pos["bot:" + lab]
            h = (y1 - y2) / (TOP - BOT) if p1 == p2 else float("nan")
            total += h
            print(f"  {lab:16s} page {p1}: {h:.2f} page incl. caption (+ about 0.02 float gap)")
    print(f"  floats total: {total:.2f} pages")
    if "endintro" in pos:
        p, y = pos["endintro"]
        print(f"intro ends on page {p}; room left below it: {y - BOT:.0f} pt = {(y - BOT) / (TOP - BOT):.2f} page")


if __name__ == "__main__":
    mark(*sys.argv[2:4]) if sys.argv[1] == "mark" else read(sys.argv[2])
