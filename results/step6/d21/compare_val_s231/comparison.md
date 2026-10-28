# Experiment comparison: val

| Run | Body @0.1 | Body (shared joints) @0.1 | Ski @0.1 | Ski @0.05 | Ski @0.1, mild crops | Ski @0.1, moderate crops | Ski @0.1, severe crops | Ski cut off, severe | Δ body vs E1 | Δ ski vs E1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 94.8% (3562/3759) | 95.2% (2519/2647) | 91.6% (1605/1752) | 82.6% (1448/1752) | 91.2% | 90.4% | 84.5% | 9.4% | +0.0 | +0.0 |
| E2 | 95.9% (3605/3759) | 96.3% (2549/2647) | 93.7% (1642/1752) | 84.3% (1477/1752) | 93.5% | 93.4% | 88.5% | 9.4% | +1.1 | +2.1 |
| E3 | 97.1% (3651/3759) | 97.4% (2578/2647) | 92.3% (1617/1752) | 82.2% (1440/1752) | 92.6% | 91.2% | 84.8% | 9.4% | +2.4 | +0.7 |
| E4 | 97.0% (3647/3759) | 97.1% (2571/2647) | 91.1% (1596/1752) | 78.5% (1375/1752) | 90.9% | 89.7% | 85.1% | 9.4% | +2.3 | -0.5 |

⚠ = body PCK@0.1 more than 2 points below E1.

**Rule's suggestion: E2** (highest clean ski PCK@0.1).
Rule: D16: drop candidates whose clean body PCK@0.1 falls > 2 points below E1; take the highest clean ski PCK@0.1; if the top two are within 1 point, take the higher moderate crop-error ski PCK@0.1; if none beats E1, keep E1 (negative result). Ruichao makes the final choice; record it in docs/decisions.md.
Paired bootstrap intervals over source videos come in Step 8.
