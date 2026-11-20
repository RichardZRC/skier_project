"""How full is the last line of each main-text paragraph? A paragraph whose last line holds only a few words loses a
whole line when 2-3 words are cut: the cheapest trims. (Cutting words elsewhere often saves nothing.)
Usage: para_fill.py mark IN.tex OUT.tex   (then compile OUT.tex)
       para_fill.py read OUT.aux OUT.tex
"""
import re
import sys
from pathlib import Path

BS = chr(92)
LEFT, WIDTH = 108.0, 396.0  # NeurIPS: 5.5in text block centred on 8.5in paper


def span(src):
    return src.index(BS + "section{Introduction}"), src.index(BS + "section{Contributions}")


def mark(inp, outp):
    src = Path(inp).read_text()
    macro = (BS + "makeatletter\n"
             + BS + "newcommand{" + BS + "paraxy}[4]{}\n"
             + BS + "newcommand{" + BS + "parasave}[1]{" + BS + "pdfsavepos" + BS + "write" + BS + "@auxout{"
             + BS + "string" + BS + "paraxy{#1}{" + BS + "thepage}{" + BS + "the" + BS + "pdflastxpos}{"
             + BS + "the" + BS + "pdflastypos}}}\n"
             + BS + "makeatother\n")
    src = src.replace(BS + "begin{document}", macro + BS + "begin{document}", 1)
    a, b = span(src)
    parts = src[a:b].split("\n\n")
    n = 0
    for i, p in enumerate(parts):
        s = p.rstrip()
        last = s.splitlines()[-1].strip() if s else ""
        if (not s or last.startswith("%") or last.startswith(BS + "end{") or last.startswith(BS + "label")
                or last.startswith(BS + "section") or last.startswith(BS + "subsection")
                or not re.search("[a-z]", last)):
            continue
        n += 1
        parts[i] = s + BS + "parasave{" + str(n) + "}" + p[len(s):]
    Path(outp).write_text(src[:a] + "\n\n".join(parts) + src[b:])


def read(aux, tex):
    text = Path(aux).read_text()
    pat = re.escape(BS) + r"paraxy\{(\d+)\}\{(\d+)\}\{(-?\d+)\}\{(-?\d+)\}"
    pos = {int(m.group(1)): (int(m.group(2)), int(m.group(3)) / 65536) for m in re.finditer(pat, text)}
    src = Path(tex).read_text()
    a, b = span(src)
    body = src[a:b]
    first = {}
    for m in re.finditer(re.escape(BS) + r"parasave\{(\d+)\}", body):
        par = body[:m.start()].split("\n\n")[-1]
        par = "\n".join(l for l in par.splitlines() if not l.lstrip().startswith("%"))
        words = [w for w in re.split(r"\s+", re.sub(re.escape(BS) + r"[a-z]+\*?(\{[^}]*\})?", " ", par))
                 if re.search("[A-Za-z]", w)]
        first[int(m.group(1))] = " ".join(words[:9])
    for k in sorted(pos):
        p, x = pos[k]
        fill = (x - LEFT) / WIDTH
        flag = "  <-- short last line: cut 2-3 words here to save a line" if fill < 0.25 else ""
        print(f"p{p} last line {fill:4.0%} full | {first.get(k, '')[:60]}{flag}")


if __name__ == "__main__":
    mark(*sys.argv[2:4]) if sys.argv[1] == "mark" else read(sys.argv[2], sys.argv[3])
