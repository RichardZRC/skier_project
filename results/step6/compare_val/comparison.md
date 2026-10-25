# Experiment comparison: val

| Run | Body @0.1 | Body (shared joints) @0.1 | Ski @0.1 | Ski @0.05 | Ski @0.1, mild crops | Ski @0.1, moderate crops | Ski @0.1, severe crops | Ski cut off, severe | Δ body vs E1 | Δ ski vs E1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | — | 95.1% (2517/2647) | — | — | — | — | — | — | — | — |
| E1 | 96.9% (3642/3759) | 97.1% (2570/2647) | 89.5% (1568/1752) | 75.5% (1323/1752) | 90.0% | 89.0% | 82.6% | 9.4% | +0.0 | +0.0 |
| E2 | 96.1% (3612/3759) | 96.5% (2554/2647) | 69.3% (1215/1752) | 52.0% (911/1752) | 69.7% | 69.9% | 66.8% | 9.4% | -0.8 | -20.1 |
| E3 | 97.9% (3679/3759) | 98.1% (2597/2647) | 77.4% (1356/1752) | 60.6% (1061/1752) | 77.7% | 77.2% | 72.7% | 9.4% | +1.0 | -12.1 |
| E4 | 97.7% (3673/3759) | 97.9% (2591/2647) | 77.6% (1360/1752) | 61.2% (1072/1752) | 78.1% | 77.6% | 73.1% | 9.4% | +0.8 | -11.9 |

⚠ = body PCK@0.1 more than 2 points below E1.

**Rule's suggestion: E1** (no candidate beats E1 on clean ski PCK@0.1: negative result, keep E1).
Rule: D16: drop candidates whose clean body PCK@0.1 falls > 2 points below E1; take the highest clean ski PCK@0.1; if the top two are within 1 point, take the higher moderate crop-error ski PCK@0.1; if none beats E1, keep E1 (negative result). Ruichao makes the final choice; record it in docs/decisions.md.
Paired bootstrap intervals over source videos come in Step 8.
