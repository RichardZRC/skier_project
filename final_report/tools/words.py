"""Prose words per section of final_report/main.tex (floats, comments, citations and labels left out).
Usage: python3 words.py final_report/main.tex"""
import re
import sys
from pathlib import Path

BS = chr(92)
src = Path(sys.argv[1]).read_text()
src = src[:src.index(BS + "section*{References}")]
src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
heads = [(m.start(), m.group(1), m.group(2))
         for m in re.finditer(re.escape(BS) + r"(section|subsection)\{([^}]*)\}", src)]
a0, a1 = src.find(BS + "begin{abstract}"), src.find(BS + "end{abstract}")
chunks = [("Abstract", src[a0:a1])]
for n, (pos, kind, title) in enumerate(heads):
    end = heads[n + 1][0] if n + 1 < len(heads) else len(src)
    if kind == "section" and n + 1 < len(heads) and heads[n + 1][1] == "subsection" and heads[n + 1][0] - pos < 80:
        continue  # a section whose text starts in its first subsection
    chunks.append((title, src[pos:end]))
total = 0
for title, body in chunks:
    text = re.sub(re.escape(BS) + r"begin\{(table|figure)\}.*?" + re.escape(BS) + r"end\{\1\}", " ", body, flags=re.S)
    text = re.sub(re.escape(BS) + r"(cite|ref|label|section|subsection)\{[^}]*\}", " ", text)
    text = re.sub(re.escape(BS) + r"(begin|end)\{abstract\}", " ", text)
    words = len([w for w in re.split(r"\s+", text) if re.search(r"[A-Za-z0-9]", w)])
    if title not in ("Contributions", "AI Disclosure"):
        total += words
    print(f"{words:5d}  {title}")
print(f"{total:5d}  main text (abstract to conclusion); about 750 words fill one page here")
