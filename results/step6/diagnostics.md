# Step 6 diagnostics: why E2–E4 lose ski points (2026-10-24, laptop)

All numbers on **val** or **training** images; test untouched. The scripts are in this folder; they read the runs
under `$SKIER_RESULTS/pose/`.

## 1. The loss is in two ski points, not general (`compare/val/per_joint.csv`)

Val PCK@0.1, clean crops:

| Joint | n | E1 | E2 | E3 | E4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ski_tip_left | 250 | 93.6 | **66.0** | 92.4 | 93.6 |
| toes_left | 238 | 93.7 | 92.0 | 94.5 | 92.4 |
| heel_left | 190 | 94.2 | 88.4 | 88.9 | 88.4 |
| ski_tail_left | 138 | 64.5 | **18.1** | **13.8** | **18.1** |
| ski_tip_right | 261 | 90.8 | **63.6** | 92.0 | 93.9 |
| toes_right | 259 | 93.8 | 93.4 | 94.6 | 94.6 |
| heel_right | 226 | 96.0 | 92.0 | 90.3 | 89.4 |
| ski_tail_right | 190 | 76.8 | **11.6** | **12.1** | **11.1** |
| pole_basket_left | 219 | 4.6 | 0.9 | 1.8 | 1.4 |
| pole_basket_right | 247 | 6.1 | 2.0 | 2.0 | 2.4 |

Body joints, toes and heels are unaffected. Poles are poor in every run (supplementary only, D7).

## 2. Where the wrong predictions land (`tail_where.py`, val, clean)

- **Tails (E2–E4) land at the boot:** the nearest labelled point is the same ski's toes (49–62%) or a heel; the median error
  is about 0.2 reference-box diagonals. They are not swapped with the tips.
- **E2's tips split between the two skis:** nearest label is the own tip 51–53% and the *other* ski's tip 42–44% (E1: 84% / 11–12%).

## 3. The training targets are the same (`supervision_check.py`, epoch 5, 300 training images)

- **E3 vs E1** (pixel-only augmentation): crop matrices, weights and heatmap targets are **identical** (max |diff| 0.0).
  **E4 vs E2:** identical crops and weights.
- Share of reliable labels supervised (inside the crop): tails 89.7 / 89.6% in E1 and E3, 87.6 / 86.6% in E2 and E4; tips
  93–96% in all. Jitter removes only 2–3 points of supervision coverage.

So the label pipeline is not the cause.

## 4. Underfitting, not a generalization gap (`train_fit.py`, best.pt on all 1,194 training images, no augmentation)

| Joint | E1 train | E2 train | E3 train | E4 train | E1 val | E2 val | E3 val | E4 val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ski_tail_right | 73.2 | 20.2 | 20.7 | 20.9 | 76.8 | 11.6 | 12.1 | 11.1 |
| ski_tail_left | 78.0 | 19.8 | 20.5 | 21.1 | 64.5 | 18.1 | 13.8 | 18.1 |
| ski_tip_right | 95.9 | 63.7 | 97.2 | 97.0 | 90.8 | 63.6 | 92.0 | 93.9 |
| ski_tip_left | 98.4 | 63.6 | 97.7 | 97.9 | 93.6 | 66.0 | 92.4 | 93.6 |
| toes_right | 100.0 | 100.0 | 99.4 | 99.5 | 93.8 | 93.4 | 94.6 | 94.6 |
| heel_right | 99.3 | 97.9 | 95.0 | 95.1 | 96.0 | 92.0 | 90.3 | 89.4 |
| all ski | 93.9 | 72.5 | 81.1 | 81.2 | 89.5 | 69.3 | 77.4 | 77.6 |

E2–E4 never learned the tails, **even on their own training images**.

## Reading

- **Whether a run learns the ski ends within the schedule varies from run to run.** E2 and E4 have identical crops and
  jitter, yet E2's tips fail and E4's don't. The 32-image overfit showed the same plateau (ski about 61% for epochs 10–50,
  then 100% by epoch 200). The schedule (50 fine-tune epochs, ×0.1 at epochs 30 and 42) can freeze a run before it escapes.
- **The current E1–E4 table is therefore confounded by convergence.** D16's rule picks E1 ("negative result"), which is the
  rule applied correctly, but it does not show that augmentation hurts ski accuracy.
- Not yet known: how often E1 itself fails to learn tails (one run so far), and whether a longer schedule fixes E2–E4.

## 5. The D20 trial (2026-10-25; `trial_d20/`, `trial_eval.py`)

Two diagnostic runs, with the reading fixed in advance (D20; "learns the tails" = training-image tail PCK@0.1 ≥ 60% on both sides):

| Joint | E1 train | E1 val | **E1-seed231** train | val | E3 train | E3 val | **E3-ft150** train | val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ski_tail_right | 73.2 | 76.8 | **19.8** | 11.6 | 20.7 | 12.1 | **95.0** | 88.9 |
| ski_tail_left | 78.0 | 64.5 | **19.6** | 15.2 | 20.5 | 13.8 | **79.6** | 60.9 |
| ski_tip_right | 95.9 | 90.8 | 65.6 | 67.0 | 97.2 | 92.0 | 100.0 | 93.5 |
| ski_tip_left | 98.4 | 93.6 | 60.1 | 63.2 | 97.7 | 92.4 | 99.8 | 94.0 |
| all ski | 93.9 | 89.5 | 72.2 | 69.4 | 81.1 | 77.4 | 97.2 | **91.8** |
| body | 99.9 | 96.9 | 100.0 | 96.7 | 99.9 | 97.9 | 100.0 | 96.6 |
| best epoch | | 47 | | 34 | | 49 | | 144 |

- **E1-seed231 fails on the tails** (about 20%), and its tips split between the skis, the same failure as E2. So E1's original
  success was partly luck: **plain E1 fails too under another seed.**
- **E3-ft150 learns the tails** (95 / 80% on training images). Its val ski PCK climbs from 81.8% (epoch 60) to 89.3% (epoch 80),
  before its first ×0.1 step at epoch 93. In the 50-epoch schedule the first step comes at epoch 33, before that climb.
  Best val ski **91.8%**, above E1's 89.5% (diagnostic only; not part of the Step 6 table).
- **D20's outcome row: "E1-seed231 fails, E3-ft150 learns"** → rerun E1–E4 with the longer schedule, with 2+ seeds per recipe.
