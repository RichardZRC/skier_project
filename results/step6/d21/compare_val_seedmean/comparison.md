# Experiment comparison: val, means over seeds 230 and 231 (D21)

| Run | Body @0.1 | Body (shared joints) @0.1 | Ski @0.1 | Ski @0.1 per seed (230 / 231) | Ski @0.05 | Ski @0.1, mild crops | Ski @0.1, moderate crops | Ski @0.1, severe crops | Ski cut off, severe | Δ body vs E1 | Δ ski vs E1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 95.2% (7155/7518) | 95.5% (5054/5294) | 91.9% (3220/3504) | 92.2% / 91.6% | 82.9% (2904/3504) | 91.4% | 90.4% | 84.8% | 9.4% | +0.0 | +0.0 |
| E2 | 95.5% (7178/7518) | 95.8% (5070/5294) | 93.1% (3263/3504) | 92.5% / 93.7% | 84.3% (2955/3504) | 93.4% | 92.9% | 88.6% | 9.4% | +0.3 | +1.2 |
| E3 | 96.9% (7283/7518) | 97.1% (5138/5294) | 92.0% (3225/3504) | 91.8% / 92.3% | 82.0% (2874/3504) | 92.3% | 90.9% | 84.9% | 9.4% | +1.7 | +0.1 |
| E4 | 97.1% (7298/7518) | 97.2% (5147/5294) | 91.0% (3187/3504) | 90.8% / 91.1% | 78.8% (2762/3504) | 90.8% | 89.5% | 85.1% | 9.4% | +1.9 | -0.9 |

⚠ = body PCK@0.1 more than 2 points below E1.
Each pooled run sums the hits and totals of its seeds, so every percentage is the seed mean; the rule applies to these means (D21).

**Rule's suggestion: E2** (highest clean ski PCK@0.1).
Rule: D16: drop candidates whose clean body PCK@0.1 falls > 2 points below E1; take the highest clean ski PCK@0.1; if the top two are within 1 point, take the higher moderate crop-error ski PCK@0.1; if none beats E1, keep E1 (negative result). Ruichao makes the final choice; record it in docs/decisions.md.
Paired bootstrap intervals over source videos come in Step 8.
