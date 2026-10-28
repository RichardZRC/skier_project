# Experiment comparison: val

| Run | Body @0.1 | Body (shared joints) @0.1 | Ski @0.1 | Ski @0.05 | Ski @0.1, mild crops | Ski @0.1, moderate crops | Ski @0.1, severe crops | Ski cut off, severe | Δ body vs E1 | Δ ski vs E1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 95.6% (3593/3759) | 95.8% (2535/2647) | 92.2% (1615/1752) | 83.1% (1456/1752) | 91.6% | 90.5% | 85.0% | 9.4% | +0.0 | +0.0 |
| E2 | 95.1% (3573/3759) | 95.2% (2521/2647) | 92.5% (1621/1752) | 84.4% (1478/1752) | 93.4% | 92.3% | 88.6% | 9.4% | -0.5 | +0.3 |
| E3 | 96.6% (3632/3759) | 96.7% (2560/2647) | 91.8% (1608/1752) | 81.8% (1434/1752) | 92.0% | 90.6% | 84.9% | 9.4% | +1.0 | -0.4 |
| E4 | 97.1% (3651/3759) | 97.3% (2576/2647) | 90.8% (1591/1752) | 79.2% (1387/1752) | 90.7% | 89.3% | 85.2% | 9.4% | +1.5 | -1.4 |

⚠ = body PCK@0.1 more than 2 points below E1.

**Rule's suggestion: E2** (highest clean ski PCK@0.1).
Rule: D16: drop candidates whose clean body PCK@0.1 falls > 2 points below E1; take the highest clean ski PCK@0.1; if the top two are within 1 point, take the higher moderate crop-error ski PCK@0.1; if none beats E1, keep E1 (negative result). Ruichao makes the final choice; record it in docs/decisions.md.
Paired bootstrap intervals over source videos come in Step 8.
